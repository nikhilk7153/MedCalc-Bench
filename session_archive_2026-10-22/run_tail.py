def is_openai(model_name):
    return model_name.split('/')[0].lower() == "openai"


def max_length_for(model_name):
    name = model_name.lower()
    if "mixtral" in name or "mistral" in name:
        return 32768
    elif "llama-3" in name:
        return 8192
    elif "meditron-70b" in name:
        return 4096
    elif "pmc_llama" in name:
        return 2048
    raise ValueError(f"Unknown model: {model_name}")


def load_tokenizer(model_name):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    name = model_name.lower()
    if "mixtral" in name or "mistral" in name:
        tokenizer.chat_template = open('../templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
    elif "meditron-70b" in name:
        tokenizer.chat_template = open('../templates/meditron.jinja').read().replace('    ', '').replace('\n', '')
    elif "pmc_llama" in name:
        tokenizer.chat_template = open('../templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
    return tokenizer


def truncate(tokenizer, text, n):
    return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:n])


# Where the opening instruction of a system prompt ends: the one-shot example (one_shot, one_shot_meditron) or
# the demonstration (zero_shot_meditron, direct_answer_meditron)
EXAMPLE_STARTS = ["\n\n###User:\nHere is an example patient note:", "Here is an example patient note:", " Here is a demonstration"]


def insert_time_point(system, time_point):
    """Put the time-point instruction right after the opening instruction, before any example or demonstration."""
    starts = [system.find(s) for s in EXAMPLE_STARTS if s in system]
    if not starts:
        return system + " " + time_point
    i = min(starts)
    return system[:i] + " " + time_point + ("" if system[i] in " \n" else " ") + system[i:]


def build_messages(row, prompt_style, model_name, tokenizer, one_shot_json):
    name = model_name.lower()
    if "meditron" in name:
        zero_shot_fn, direct_answer_fn, one_shot_fn = zero_shot_meditron, direct_answer_meditron, one_shot_meditron
    else:
        zero_shot_fn, direct_answer_fn, one_shot_fn = zero_shot, direct_answer, one_shot

    patient_note = row["Patient Note"]
    question = row["Question"]
    calculator_id = str(row["Calculator ID"])

    if "pmc_llama" in name:
        patient_note = truncate(tokenizer, patient_note, 256)
    if prompt_style == "zero_shot":
        system, user = zero_shot_fn(patient_note, question)
    elif prompt_style == "one_shot":
        example = copy.deepcopy(one_shot_json[calculator_id])
        if "meditron" in name:
            example["Patient Note"] = truncate(tokenizer, example["Patient Note"], 512)
            example["Response"]["step_by_step_thinking"] = truncate(tokenizer, example["Response"]["step_by_step_thinking"], 512)
        elif "pmc_llama" in name:
            example["Patient Note"] = truncate(tokenizer, example["Patient Note"], 256)
            example["Response"]["step_by_step_thinking"] = truncate(tokenizer, example["Response"]["step_by_step_thinking"], 256)
        system, user = one_shot_fn(patient_note, question, one_shot_example_question(calculator_id, question), example["Patient Note"], {"step_by_step_thinking": example["Response"]["step_by_step_thinking"], "answer": example["Response"]["answer"]})
    elif prompt_style == "direct_answer":
        system, user = direct_answer_fn(patient_note, question)
    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    time_point = row.get("Time Point Instruction")
    if isinstance(time_point, str) and time_point.strip():
        system = insert_time_point(system, time_point.strip())

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    return patient_note, messages


def score(row, patient_note, answer, prompt_style):
    calculator_id = str(row["Calculator ID"])
    outputs = {
        "Row Number": int(row["Row Number"]),
        "Calculator Name": row["Calculator Name"],
        "Calculator ID": calculator_id,
        "Category": row["Category"],
        "Note ID": str(row["Note ID"]),
        "Patient Note": patient_note,
        "Question": row["Question"],
        "Ground Truth Answer": row["Ground Truth Answer"],
        "Ground Truth Explanation": row["Ground Truth Explanation"],
        "LLM Raw Output": answer,
    }
    try:
        answer_value, explanation = extract_answer(answer, int(calculator_id))
        correctness = check_correctness(answer_value, row["Ground Truth Answer"], calculator_id, row["Upper Limit"], row["Lower Limit"])
        outputs["LLM Answer"] = answer_value
        outputs["LLM Explanation"] = explanation
        outputs["Result"] = "Correct" if correctness else "Incorrect"
    except Exception as e:
        outputs["LLM Answer"] = str(e)
        outputs["LLM Explanation"] = str(e)
        outputs["Result"] = "Incorrect"
    if prompt_style == "direct_answer":
        outputs["LLM Explanation"] = "N/A"
    return outputs


def run_vllm(args, model_name, df, one_shot_json, output_paths):
    from vllm import LLM, SamplingParams

    prompt_styles = [p for p in output_paths if not os.path.exists(os.path.join("outputs", output_paths[p]))]
    if not prompt_styles:
        print("All outputs already exist.")
        return []

    max_length = max_length_for(model_name)
    tokenizer = load_tokenizer(model_name)
    stop_token_ids = [tokenizer.eos_token_id]
    if "llama-3" in model_name.lower():
        stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    stop = ["###", "User:", "\n\n\n"] if "meditron" in model_name.lower() else None

    jobs = []
    for prompt_style in prompt_styles:
        for index in range(len(df)):
            row = df.iloc[index]
            patient_note, messages = build_messages(row, prompt_style, model_name, tokenizer, one_shot_json)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            # llm_inference.py: max_length=min(max_length, len(encode(prompt, add_special_tokens=True)) + 4096)
            total = min(max_length, len(tokenizer.encode(prompt, add_special_tokens=True)) + 4096)
            jobs.append({"prompt_style": prompt_style, "row": row, "patient_note": patient_note, "ids": ids,
                         "max_tokens": total - len(ids)})

    too_long = [j for j in jobs if j["max_tokens"] < 1]
    if too_long:
        print(f"{len(too_long)} prompts do not fit in the {max_length}-token context; they are scored Incorrect.")
    runnable = [j for j in jobs if j["max_tokens"] >= 1]

    # vLLM adds the eos ids of the model's generation_config.json to the stop tokens even with generation_config="vllm"
    # (PMC_LLaMA_13B's file lists eos_token_id 1, which is <s>), so point it at a config holding only the eos ids
    # llm_inference.py passed explicitly
    generation_config_dir = tempfile.mkdtemp()
    with open(os.path.join(generation_config_dir, "generation_config.json"), "w") as f:
        json.dump({"eos_token_id": stop_token_ids}, f)
    llm = LLM(model=model_name, tensor_parallel_size=args.tp, max_model_len=max_length, dtype="bfloat16", gpu_memory_utilization=0.9,
              generation_config=generation_config_dir)
    sampling = [SamplingParams(temperature=0.0, max_tokens=j["max_tokens"], stop=stop, include_stop_str_in_output=True,
                               stop_token_ids=stop_token_ids) for j in runnable]
    generations = llm.generate([{"prompt_token_ids": j["ids"]} for j in runnable], sampling)
    for j, g in zip(runnable, generations):
        o = g.outputs[0]
        j["answer"] = re.sub(r"\s+", " ", o.text)
        j["finish"] = {"Finish Reason": o.finish_reason, "Stop Reason": o.stop_reason, "Output Tokens": len(o.token_ids)}
    for j in too_long:
        j["answer"] = f"Prompt exceeds the {max_length}-token context"
        j["finish"] = {"Finish Reason": "prompt_too_long", "Stop Reason": None, "Output Tokens": 0}

    for prompt_style in prompt_styles:
        with open(os.path.join("outputs", output_paths[prompt_style]), "w") as f:
            for j in jobs:
                if j["prompt_style"] == prompt_style:
                    f.write(json.dumps({**score(j["row"], j["patient_note"], j["answer"], prompt_style), **j["finish"]}) + "\n")
    return prompt_styles


def openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key and os.path.exists(os.path.expanduser("~/.openai_key")):
        key = open(os.path.expanduser("~/.openai_key")).read().strip()
    if not key:
        raise SystemExit("Set OPENAI_API_KEY or put the key in ~/.openai_key")
    return key


def run_openai(args, model_name, df, one_shot_json, output_paths):
    # Rows are appended as they finish, so an interrupted run resumes where it stopped.
    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key(), max_retries=8)
    api_model = model_name.split('/', 1)[1]

    def call(row, prompt_style):
        patient_note, messages = build_messages(row, prompt_style, model_name, None, one_shot_json)
        response = client.chat.completions.create(model=api_model, messages=messages)
        outputs = score(row, patient_note, re.sub(r"\s+", " ", response.choices[0].message.content), prompt_style)
        outputs["LLM Model Version"] = response.model
        return outputs

    finished = []
    for prompt_style, output_path in output_paths.items():
        full_path = os.path.join("outputs", output_path)
        done = set()
        if os.path.exists(full_path):
            done = {json.loads(line)["Row Number"] for line in open(full_path)}
        todo = [df.iloc[i] for i in range(len(df)) if int(df.iloc[i]["Row Number"]) not in done]

        failed = 0
        with ThreadPoolExecutor(args.workers) as ex, open(full_path, "a") as f:
            futures = [ex.submit(call, row, prompt_style) for row in todo]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=prompt_style):
                try:
                    f.write(json.dumps(future.result()) + "\n")
                    f.flush()
                except Exception as e:
                    failed += 1
                    print(f"API error ({prompt_style}): {e}")

        if failed:
            print(f"{prompt_style}: {failed} rows failed; run again to retry them.")
            continue
        rows = sorted((json.loads(line) for line in open(full_path)), key=lambda r: r["Row Number"])
        with open(full_path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        finished.append(prompt_style)
    return finished


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run MedCalc-Bench')
    parser.add_argument('--model', type=str, required=True, help='Specify which model you are using. Options are OpenAI/gpt-4, OpenAI/gpt-3.5-turbo (any OpenAI/<model>), mistralai/Mistral-7B-Instruct-v0.2, mistralai/Mixtral-8x7B-Instruct-v0.1, meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Meta-Llama-3-70B-Instruct, epfl-llm/meditron-70b, axiong/PMC_LLaMA_13B')
    parser.add_argument('--prompt', type=str, nargs='+', default=["direct_answer", "zero_shot", "one_shot"], help='Specify one or more prompt types. Options are direct_answer, zero_shot, one_shot')
    parser.add_argument('--data', type=str, default="../final_files_2026-10-16/test_data_final.csv")
    parser.add_argument('--one_shot', type=str, default="../final_files_2026-10-16/one_shot_finalized_explanation.json")
    parser.add_argument('--tag', type=str, default="test_data_final", help='Subfolder of outputs/ and suffix of the results file')
    parser.add_argument('--tp', type=int, default=1, help='vLLM tensor parallel size (number of GPUs)')
    parser.add_argument('--workers', type=int, default=16, help='Parallel OpenAI requests')
    parser.add_argument('--limit', type=int, default=None, help='Only run the first N rows (for a test)')
    args = parser.parse_args()

    model_name = args.model
    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)
    with open(args.one_shot, "r") as file:
        one_shot_json = json.load(file)

    os.makedirs(os.path.join("outputs", args.tag), exist_ok=True)
    output_paths = {p: os.path.join(args.tag, f"{model_name.replace('/', '_')}_{p}.jsonl") for p in args.prompt}

    if is_openai(model_name):
        finished = run_openai(args, model_name, df, one_shot_json, output_paths)
    else:
        finished = run_vllm(args, model_name, df, one_shot_json, output_paths)

    for prompt_style in finished:
        stats = compute_overall_accuracy(output_paths[prompt_style], model_name, f"{prompt_style}_{args.tag}")
        print(model_name, prompt_style, json.dumps(stats))
