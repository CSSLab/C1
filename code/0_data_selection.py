import argparse
import json
import os
from collections import Counter
import pandas as pd
import tqdm
from datasets import load_dataset
from utils import get_minimal_prompt


def get_rare_themes(data, n_themes):
    theme_freq = data['Themes'].str.split().explode().value_counts(ascending=True)
    return theme_freq.index[:n_themes].tolist(), theme_freq


def construct_test_set(cache_dir, hf_dataset_name, output_path_test):
    os.makedirs(cache_dir, exist_ok=True)
    data = load_dataset(hf_dataset_name, "short_tactics", cache_dir=cache_dir)
    data = pd.DataFrame(data['train']).values

    ret = []
    for line in data:
        puzzle_type = line[1].split("_")[-1]
        fen = line[3]
        prompt = get_minimal_prompt(fen)
        move = line[6]
        metadata = json.loads(line[8])

        item = {
            'data_source': 'chess_puzzles',
            'prompt': [{'role': 'user', 'content': prompt}],
            'ability': 'chess',
            'reward_model': {
                'ground_truth': move,
                'style': 'rule'
            },
            'extra_info': {
                'fen': fen,
                'type': puzzle_type,
                **metadata
            }
        }

        ret.append(item)

    df = pd.DataFrame(ret)
    df.to_parquet(output_path_test, index=False)
    print(f"Saved {len(df)} samples for testing to {output_path_test}")


def prepare_themes_list(data):
    data = data.copy()
    data['themes_list'] = data['Themes'].str.split()
    return data


def load_reserved_data(cache_dir, puzzle_dataset_path, seed):
    sft_path = os.path.join(cache_dir, "reserve_sft.csv")
    rl_path = os.path.join(cache_dir, "reserve_rl.csv")

    if os.path.exists(sft_path) and os.path.exists(rl_path):
        sft_data = pd.read_csv(sft_path)
        rl_data = pd.read_csv(rl_path)
        print(f"Loaded reserved SFT: {sft_data.shape}, RL: {rl_data.shape}", flush=True)
    else:
        print("Creating new reserved split...", flush=True)
        all_data = pd.read_csv(puzzle_dataset_path)
        all_data = all_data.sample(frac=1, random_state=seed).reset_index(drop=True)
        mid = len(all_data) // 2
        sft_data, rl_data = all_data.iloc[:mid].copy(), all_data.iloc[mid:].copy()
        sft_data.to_csv(sft_path, index=False)
        rl_data.to_csv(rl_path, index=False)
        print(f"Created SFT: {sft_data.shape}, RL: {rl_data.shape}", flush=True)

    return sft_data, rl_data


def fill_samples_by_theme(data, rare_themes, n_per_theme):
    seen_ids, ret_idx, found = set(), [], {theme: 0 for theme in rare_themes}

    for theme in tqdm.tqdm(rare_themes, desc="Filling themes"):
        if found[theme] >= n_per_theme:
            continue

        mask = data['themes_list'].apply(lambda themes: theme in themes)
        for idx, row in data[mask].iterrows():
            if found[theme] >= n_per_theme:
                break
            if row['PuzzleId'] in seen_ids:
                continue

            seen_ids.add(row['PuzzleId'])
            ret_idx.append(idx)
            found[theme] += 1

    return data.loc[ret_idx].reset_index(drop=True), found


def select_balanced(data, n_per_theme, n_themes):
    data = prepare_themes_list(data)
    rare_themes, theme_freq = get_rare_themes(data, n_themes)

    print(f"Balanced: {len(theme_freq)} themes, targeting {n_per_theme} per theme")
    print(f"Themes: {rare_themes[:10]}{'...' if len(rare_themes) > 10 else ''}")

    result, found = fill_samples_by_theme(data, rare_themes, n_per_theme)

    for theme, count in found.items():
        if count < n_per_theme:
            print(f"  Warning: '{theme}' only has {count} samples (target: {n_per_theme})")

    return result


def print_stats(df, name):
    print(f"\n=== {name} ===")
    print(f"Samples: {len(df)}")

    themes_flat = [t for sublist in df['Themes'].str.split().tolist() for t in sublist]
    theme_counts = Counter(themes_flat)
    print(f"Unique themes: {len(theme_counts)}")
    for theme, count in sorted(theme_counts.items(), key=lambda x: -x[1]):
        print(f"  {theme}: {count}")

    ratings = df['Rating'].tolist()
    print(f"Rating: {int(min(ratings))} - {int(max(ratings))}, avg {sum(ratings) / len(ratings):.1f}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Constructing test set")
    print("=" * 60)
    construct_test_set(args.cache_dir, args.hf_dataset_name, args.test_parquet_path)

    sft_reserve, rl_reserve = load_reserved_data(args.cache_dir, args.puzzle_dataset_path, args.seed)

    print("\n" + "=" * 60)
    print("Constructing RL data (balanced)")
    print("=" * 60)

    rl_data = select_balanced(rl_reserve, args.N_rl_per_theme, args.N_rl_themes)
    rl_data = rl_data.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    rl_path = os.path.join(args.output_dir, "train_rl.csv")
    rl_data.to_csv(rl_path, index=False)
    print(f"Saved {len(rl_data)} to {rl_path}")
    print_stats(rl_data, "RL Balanced")

    print("\n" + "=" * 60)
    print("Constructing SFT data (balanced)")
    print("=" * 60)

    sft_data = select_balanced(sft_reserve, args.N_sft_per_theme, args.N_sft_themes)
    sft_data = sft_data.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sft_path = os.path.join(args.output_dir, "train_sft.csv")
    sft_data.to_csv(sft_path, index=False)
    print(f"Saved {len(sft_data)} to {sft_path}")
    print_stats(sft_data, "SFT Balanced")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="../data/raw")
    parser.add_argument("--puzzle_dataset_path", type=str, default="../data/raw/lichess_db_puzzle.csv")
    parser.add_argument("--test_parquet_path", type=str, default="../data/test.parquet")
    parser.add_argument("--output_dir", type=str, default="../data")
    parser.add_argument("--hf_dataset_name", type=str, default="wieeii/ChessQA-Benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N_rl_per_theme", type=int, default=800)
    parser.add_argument("--N_rl_themes", type=int, default=50)
    parser.add_argument("--N_sft_per_theme", type=int, default=800)
    parser.add_argument("--N_sft_themes", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    main()
