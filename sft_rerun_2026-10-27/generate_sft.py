"""
Generates test-set responses with a fine-tuned model (or a base model, for the baseline), for score_sft.py.

  python generate_sft.py --model <Hugging Face model dir or hub id> --out outputs/<name>.jsonl [--tp 1]

Prompts: data/test_chatml.jsonl (system and user turns), written out in the chatML format the models were trained on
(<|im_start|>role\\ncontent<|im_end|>\\n ... <|im_start|>assistant\\n), so a base model gets exactly the same text as its
fine-tuned version. Greedy decoding (temperature 0), as in every other run; generation stops at <|im_end|> or the
end-of-sequence token. Output budget as in evaluation/run.py: up to 4,096 new tokens within the model's context
(Llama-2: 4,096 in all). A prompt that leaves no room gets an empty output (scored Incorrect) and is counted.
Writes one line per test row: {"Row Number", "Output", "Finish Reason", "Output Tokens"}. Needs vllm and a GPU.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_NEW_TOKENS = 4096


def chatml(conversation):
    """The prompt text: every turn as Megatron-LLM's preprocess_instruct_data.py formats it, then the assistant header."""
    turns = "".join(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n" for turn in conversation)
    return turns + "<|im_start|>assistant\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts", default=os.path.join(HERE, "data", "test_chatml.jsonl"))
    parser.add_argument("--tp", type=int, default=1, help="number of GPUs")
    args = parser.parse_args()

    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    context = AutoConfig.from_pretrained(args.model).max_position_embeddings
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16", max_model_len=context)
    tokenizer = llm.get_tokenizer()

    rows = [json.loads(line) for line in open(args.prompts)]
    prompt_ids = [tokenizer.encode(chatml(row["conversations"])) for row in rows]
    budgets = [min(MAX_NEW_TOKENS, context - len(ids)) for ids in prompt_ids]
    runnable = [i for i, budget in enumerate(budgets) if budget >= 1]
    print(f"{len(rows)} prompts; {len(rows) - len(runnable)} do not fit in the {context}-token context")

    sampling = [SamplingParams(temperature=0.0, max_tokens=budgets[i], stop=["<|im_end|>"]) for i in runnable]
    generations = llm.generate([{"prompt_token_ids": prompt_ids[i]} for i in runnable], sampling)
    results = {i: generation.outputs[0] for i, generation in zip(runnable, generations)}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for i, row in enumerate(rows):
            out = results.get(i)
            f.write(json.dumps({
                "Row Number": row["Row Number"],
                "Output": out.text if out else "",
                "Finish Reason": out.finish_reason if out else "prompt_too_long",
                "Output Tokens": len(out.token_ids) if out else 0,
            }) + "\n")
    print("wrote", args.out)
