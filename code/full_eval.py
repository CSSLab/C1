"""Evaluate a full HF model dir on the chess test set (greedy pass@1,
FINAL_ANSWER exact-match UCI — same protocol as verl's in-training validation).

One model per process — loop over checkpoints in bash, one invocation each,
so vllm memory is fully released.
"""
from vllm import LLM, SamplingParams
import os
import json
import pandas as pd
import argparse
from pathlib import Path
from transformers import AutoTokenizer

from utils import convert_numpy_types
from chess_reward_function import compute_score


def main(cfg):
    print("Configuration:")
    for key, value in vars(cfg).items():
        print(f"{key}: {value}")
    print()

    df = pd.read_parquet(cfg.test_data_path)
    print(f"Loaded {len(df)} records for testing")
    messages_list = df['prompt'].values.tolist()
    gts = df["reward_model"].values.tolist()
    extra_infos = df["extra_info"].values.tolist()

    tokenizer_path = cfg.tokenizer_path or cfg.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    llm = LLM(
        model=cfg.model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        dtype="auto",
        max_num_seqs=cfg.max_num_seqs,
    )
    sampling_params = SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        n=cfg.n,
    )

    formatted_prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    outputs = llm.generate(formatted_prompts, sampling_params)

    results = []
    all_accuracies = []
    all_lengths = []
    clipped_count = 0
    for i, output in enumerate(outputs):
        gt = gts[i]['ground_truth']
        # pass@1 over n samples: average score across samples
        sample_scores = []
        for o in output.outputs:
            sample_scores.append(compute_score("chess_puzzles", o.text, gt)["score"])
        score = sum(sample_scores) / len(sample_scores)
        pred = output.outputs[0].text
        all_accuracies.append(score)
        response_length = len(tokenizer.encode(pred))
        all_lengths.append(response_length)
        if response_length >= cfg.max_tokens:
            clipped_count += 1
        results.append({
            "input": messages_list[i][0]["content"] if len(messages_list[i]) else "",
            "output": pred,
            "gts": gt,
            "score": score,
            "num_tokens": response_length,
            **convert_numpy_types(extra_infos[i])
        })

    accuracy = sum(all_accuracies) / len(all_accuracies)
    avg_length = sum(all_lengths) / len(all_lengths)
    print(f"\n{'='*60}")
    print(f"Model: {cfg.model_path}")
    print(f"Sampling: temp={cfg.temperature} top_p={cfg.top_p} n={cfg.n}")
    print(f"Accuracy: {accuracy:.4f} ({sum(all_accuracies):.1f}/{len(all_accuracies)})")
    print(f"Average Response Length: {avg_length:.2f} tokens")
    print(f"Clipped Responses: {clipped_count}/{len(all_lengths)} ({clipped_count/len(all_lengths):.2%})")
    print(f"{'='*60}")

    if cfg.output_path:
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        with open(cfg.output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Saved {len(results)} results to {cfg.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default="../data/test.parquet")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=1)
    cfg = parser.parse_args()
    main(cfg)
