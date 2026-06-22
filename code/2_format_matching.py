import argparse
import json
import os
import pandas as pd
import tqdm
from utils import make_pre_move, get_minimal_prompt


def fen_key(fen):
    return ' '.join(fen.split()[:4])


def load_test_fen_keys(test_path):
    df = pd.read_parquet(test_path)
    return set(df['extra_info'].apply(lambda x: fen_key(x['fen'])))


def format_sft_data(csv_path, json_path, test_fen_keys):
    df = pd.read_csv(csv_path)
    records, skipped, excluded = [], 0, 0
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="SFT data"):
        fen_after, cot = row['fen_after'], row['cot']
        if not isinstance(fen_after, str) or not isinstance(cot, str) or not cot.strip():
            skipped += 1
            continue
        if fen_key(fen_after) in test_fen_keys:
            excluded += 1
            continue
        records.append({
            "instruction": get_minimal_prompt(fen_after),
            "input": "",
            "output": cot
        })

    print(f"Skipped {skipped} invalid rows, excluded {excluded} rows overlapping test set")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"Saved {len(records)} samples to {json_path}")


def format_rl_data(csv_path, parquet_path, test_fen_keys):
    df = pd.read_csv(csv_path)
    records, excluded = [], 0
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="RL data"):
        fen_after, ground_truth = make_pre_move(row)
        if fen_key(fen_after) in test_fen_keys:
            excluded += 1
            continue
        records.append({
            'data_source': 'chess_puzzles',
            'prompt': [{'role': 'user', 'content': get_minimal_prompt(fen_after)}],
            'ability': 'chess',
            'reward_model': {'ground_truth': ground_truth, 'style': 'rule'},
            'extra_info': {
                'fen': fen_after,
                'puzzle_id': row['PuzzleId'],
                'rating': row['Rating'],
                'themes': eval(row['themes_list'])
            }
        })

    print(f"Excluded {excluded} rows overlapping test set")
    out_df = pd.DataFrame(records)
    out_df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(out_df)} records to {parquet_path}")


def register_dataset(dataset_info_path, dataset_name, file_name):
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            datasets = json.load(f)
    else:
        datasets = {}

    datasets[dataset_name] = {"file_name": file_name}
    with open(dataset_info_path, 'w') as f:
        json.dump(datasets, f, indent=2)
    print(f"Registered {dataset_name} -> {file_name}")


def main():
    args = parse_args()

    test_fen_keys = load_test_fen_keys(args.test_path)
    print(f"Loaded {len(test_fen_keys)} test FENs for exclusion")

    print("\n=== SFT Format ===")
    format_sft_data(args.sft_input_path, args.sft_output_path, test_fen_keys)
    register_dataset(args.dataset_info_path, args.dataset_name, os.path.basename(args.sft_output_path))

    print("\n=== RL Format ===")
    format_rl_data(args.rl_input_path, args.rl_output_path, test_fen_keys)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_input_path", type=str, default="../data/train_sft_cot.csv")
    parser.add_argument("--sft_output_path", type=str, default="../data/train_sft_cot.json")
    parser.add_argument("--rl_input_path", type=str, default="../data/train_rl.csv")
    parser.add_argument("--rl_output_path", type=str, default="../data/train_rl.parquet")
    parser.add_argument("--test_path", type=str, default="../data/test.parquet")
    parser.add_argument("--dataset_info_path", type=str, default="../data/dataset_info.json")
    parser.add_argument("--dataset_name", type=str, default="sft_data_cot")
    return parser.parse_args()


if __name__ == "__main__":
    main()
