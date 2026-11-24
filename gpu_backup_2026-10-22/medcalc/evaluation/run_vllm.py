'''
Batched vLLM version of run.py for the open-weight models. Prompts, chat templates, note truncation,
answer extraction and scoring are the same as run.py / llm_inference.py; generation is greedy.
Differences from run.py: all prompts of a model are generated in one batch, the time-point instruction
of the test set (if any) is appended to the system prompt, and answers are extracted from the model's
completion only (the HF pipeline also returned the prompt).
As in transformers 4.41.2's text-generation pipeline, the templated prompt is tokenized without adding special tokens
(the chat template supplies any BOS), and only the eos ids llm_inference.py passed explicitly stop generation.
'''

import os
import re
import copy
import json
import argparse
import tempfile
import pandas as pd

import run
from run import extract_answer
from evaluate import check_correctness
from table_stats import compute_overall_accuracy


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


def build_messages(row, prompt_style, model_name, tokenizer, one_shot_json):
    name = model_name.lower()
    if "meditron" in name:
        zero_shot, direct_answer, one_shot = run.zero_shot_meditron, run.direct_answer_meditron, run.one_shot_meditron
    else:
        zero_shot, direct_answer, one_shot = run.zero_shot, run.direct_answer, run.one_shot

    patient_note = row["Patient Note"]
    question = row["Question"]
    calculator_id = str(row["Calculator ID"])

    if "pmc_llama" in name:
        patient_note = truncate(tokenizer, patient_note, 256)
    if prompt_style == "zero_shot":
        system, user = zero_shot(patient_note, question)
    elif prompt_style == "one_shot":
        example = copy.deepcopy(one_shot_json[calculator_id])
        if "meditron" in name:
            example["Patient Note"] = truncate(tokenizer, example["Patient Note"], 512)
            example["Response"]["step_by_step_thinking"] = truncate(tokenizer, example["Response"]["step_by_step_thinking"], 512)
        elif "pmc_llama" in name:
            example["Patient Note"] = truncate(tokenizer, example["Patient Note"], 256)
            example["Response"]["step_by_step_thinking"] = truncate(tokenizer, example["Response"]["step_by_step_thinking"], 256)
        system, user = one_shot(patient_note, question, run.one_shot_example_question(calculator_id, question), example["Patient Note"], {"step_by_step_thinking": example["Response"]["step_by_step_thinking"], "answer": example["Response"]["answer"]})
    elif prompt_style == "direct_answer":
        system, user = direct_answer(patient_note, question)
    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    time_point = row.get("Time Point Instruction")
    if isinstance(time_point, str) and time_point.strip():
        system += " " + time_point.strip()

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


if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser(description='Run MedCalc-Bench with vLLM')
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model id, e.g. meta-llama/Meta-Llama-3-70B-Instruct')
    parser.add_argument('--prompts', type=str, nargs='+', default=["direct_answer", "zero_shot", "one_shot"])
    parser.add_argument('--data', type=str, default="../final_files_2026-10-16/test_data_final.csv")
    parser.add_argument('--one_shot', type=str, default="../final_files_2026-10-16/one_shot_finalized_explanation.json")
    parser.add_argument('--tag', type=str, default="test_data_final", help='Subfolder of outputs/ and suffix of the results file')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel size (number of GPUs)')
    args = parser.parse_args()

    model_name = args.model
    max_length = max_length_for(model_name)
    df = pd.read_csv(args.data)
    with open(args.one_shot, "r") as file:
        one_shot_json = json.load(file)

    os.makedirs(os.path.join("outputs", args.tag), exist_ok=True)
    output_paths = {p: os.path.join(args.tag, f"{model_name.replace('/', '_')}_{p}.jsonl") for p in args.prompts}
    prompt_styles = [p for p in args.prompts if not os.path.exists(os.path.join("outputs", output_paths[p]))]
    if not prompt_styles:
        print("All outputs already exist.")
        raise SystemExit

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
        stats = compute_overall_accuracy(output_paths[prompt_style], model_name, f"{prompt_style}_{args.tag}")
        print(model_name, prompt_style, json.dumps(stats))
