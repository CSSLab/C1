import pandas as pd
import os
import argparse
import json
from utils import get_piece_arrangement, load_api_key, make_pre_move
from chess_reward_function import compute_score
import chess
import tqdm
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed




def get_distillation_prompt(row, fen, move):
    board = chess.Board(fen)
    side_to_move = 'White' if fen.split()[1] == 'w' else 'Black'
    legal_moves = ', '.join(board.san(m) for m in board.legal_moves)

    prompt = f"""You are a chess grandmaster generating training data for teaching small LLMs to solve chess puzzles.

CRITICAL: The small model only sees the FEN string. Your analysis must show how to go from FEN → piece positions → tactical relationships → solution. Ground everything in explicit square names.

CONTEXT (use this to ground your analysis; present everything as your own reasoning, never as something you were told):
FEN: {fen}
Board:
{board}
Pieces: {get_piece_arrangement(fen)}
Side to Move: {side_to_move}
Opponent's Last Move: {row['Moves'].split()[0]}
Solution: {move}
Rating: {row['Rating']}
Themes: {row['Themes'].replace(' ', ', ')}
Legal Moves: {legal_moves}"""

    if 'mateIn1' not in row['Themes']:
        prompt += f"\nPV: {' '.join(row['Moves'].split()[1:])}"

    prompt += f"""

TASK: Write natural chain-of-thought analysis arriving at {move}.

Your analysis should cover (in natural prose, vary the structure):
- Where key pieces are (use square names: "the queen on h5", "king on g1")
- What tactical relationship exists (attacks, pins, weak squares, defender counts)
- Why the move works (what it threatens, why opponent can't respond adequately)

Style:
- Objective voice, no "I see/notice"
- Standard notation with brief clarification when helpful: "Qxh7+ (queen takes h7 with check)"
- Justify the move through the key tactical idea and main threat; keep any deeper continuation brief and mention only follow-up moves you are certain are legal. Present everything as your own reasoning - never reference the provided fields (rating, themes, solution, PV, legal moves) as sources, and never imply you were given the answer
- 4-10 sentences, scaled to complexity

End with exactly: FINAL_ANSWER: {move}"""

    return prompt


def process_api_requests_concurrent(df, api_key, cfg):
    output_path = cfg.output_path
    target_count = cfg.max_samples if cfg.max_samples is not None else len(df)
    df_target = df.iloc[:target_count]

    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        done_ids = set(df_existing['PuzzleId'])
        print(f"Found {len(df_existing)} rows covering {len(done_ids)} completed PuzzleIds")
        df_remaining = df_target[~df_target['PuzzleId'].isin(done_ids)]
        if len(df_remaining) == 0:
            print(f"All {len(df_target)} target puzzles already completed")
            return df_existing
        print(f"Processing {len(df_remaining)} remaining puzzles")
    else:
        df_remaining = df_target
        print(f"Processing {len(df_remaining)} samples")

    all_items = df_remaining.to_dict('records')
    results = []

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {executor.submit(process_single_row, row, api_key, cfg): idx
                  for idx, row in enumerate(all_items)}

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="API calls"):
            try:
                result = future.result()
                results.append(result)

                if len(results) % cfg.save_interval == 0:
                    append_results(results, output_path)
                    print(f"Saved checkpoint: {len(results)} samples")
                    results = []
            except Exception as e:
                print(f"Error processing item: {e}")
                pass

    return save_results(results, output_path)


def append_results(results, output_path):
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)


def save_results(results, output_path):
    df_result = pd.DataFrame(results)
    if os.path.exists(output_path):
        df_result.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df_result.to_csv(output_path, index=False)
    return df_result


def process_single_row(row, api_key, cfg):
    fen_after, ground_truth = make_pre_move(row)
    prompt = get_distillation_prompt(row, fen_after, ground_truth)

    retry_params = {
        "max_retries": cfg.max_retries,
        "initial_delay": cfg.initial_delay,
        "exponential_base": cfg.exponential_base,
        "max_delay": cfg.max_delay
    }

    response_data = call_api_with_retry(
        user_prompt=prompt,
        model_name=cfg.model,
        max_tokens=cfg.max_tokens,
        api_key=api_key,
        retry_params=retry_params,
        ground_truth=ground_truth
    )

    reasoning_text = response_data.get("reasoning_details", {}).get("text", "")

    result = dict(row)
    result['fen_after'] = fen_after
    result['cot'] = response_data["content"]
    result['reasoning'] = reasoning_text
    result['best_move'] = ground_truth
    result['total_tokens'] = response_data["total_tokens"]
    result['prompt_tokens'] = response_data["prompt_tokens"]
    result['completion_tokens'] = response_data["completion_tokens"]
    result['cost'] = response_data["cost"]
    result['correct'] = int(response_data.get("correctness_score", 0))

    return result


def make_api_request(user_prompt, model_name, max_tokens, api_key):
    api_params = {
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_prompt}],
        "usage": {"include": True},
        "reasoning": {"enabled": True}
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=api_params,
        timeout=600
    )

    response.raise_for_status()

    try:
        result = response.json()
    except json.JSONDecodeError as e:
        raise requests.exceptions.ConnectionError(f"Invalid JSON response: {e}")

    if "choices" in result and len(result["choices"]) > 0:
        message = result["choices"][0].get("message", {})
        content = message.get("content", "")
        if content:
            usage = result.get("usage", {})
            return {
                "content": content,
                "reasoning_details": message.get("reasoning_details", [{}])[0] if message.get("reasoning_details") else {},
                "total_tokens": usage.get("total_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "cost": usage.get("cost", 0.0)
            }

    if "error" in result:
        error_msg = result["error"].get("message", str(result["error"]))
        if "rate" in error_msg.lower() or "limit" in error_msg.lower():
            raise requests.exceptions.HTTPError(response=response)
        raise requests.exceptions.ConnectionError(f"API Error: {error_msg}")

    raise requests.exceptions.ConnectionError("Empty response from API")


def call_api_with_retry(user_prompt, model_name, max_tokens, api_key, retry_params, ground_truth=None):
    delay = retry_params["initial_delay"]
    last_exception = None

    for attempt in range(retry_params["max_retries"] + 1):
        try:
            response_data = make_api_request(user_prompt, model_name, max_tokens, api_key)

            if ground_truth is not None:
                result = compute_score("chess_puzzles", response_data["content"], ground_truth)
                response_data["correctness_score"] = result["score"]

                if result["score"] < 1.0:
                    last_exception = ValueError(
                        f"Incorrect response (score={result['score']})"
                    )
                    if attempt == retry_params["max_retries"]:
                        break
                    print(f"Incorrect response (score={result['score']}), "
                          f"retry {attempt + 1}/{retry_params['max_retries']} in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * retry_params["exponential_base"], retry_params["max_delay"])
                    continue

            return response_data

        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                last_exception = e
                if attempt == retry_params["max_retries"]:
                    break
                print(f"Rate limit hit, retry {attempt + 1}/{retry_params['max_retries']} in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * retry_params["exponential_base"], retry_params["max_delay"])
            else:
                raise e
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError, requests.exceptions.Timeout) as e:
            last_exception = e
            if attempt == retry_params["max_retries"]:
                break
            print(f"Connection error, retry {attempt + 1}/{retry_params['max_retries']} in {delay:.1f}s...")
            time.sleep(delay)
            delay = min(delay * retry_params["exponential_base"], retry_params["max_delay"])
        except Exception as e:
            raise e

    raise last_exception


def main():
    cfg = parse_args()
    print("Configuration:")
    for key, value in vars(cfg).items():
        print(f"{key}: {value}")

    api_key = load_api_key(cfg)
    print(f"Using model: {cfg.model}")

    df = pd.read_csv(cfg.input_path)
    print(f"Loaded {len(df)} samples, distilling CoT with model {cfg.model}")
    print(f"Output will be saved to {cfg.output_path}")

    df_result = process_api_requests_concurrent(df, api_key, cfg)

    df_all = pd.read_csv(cfg.output_path)
    print(f"\nSaved {len(df_all)} samples to {cfg.output_path}")

    correct_count = sum(1 for v in df_all['correct'] if v == 1)
    wrong_count = sum(1 for v in df_all['correct'] if v == 0)
    total_cost = df_all['cost'].sum()
    total_tokens = df_all['total_tokens'].sum()

    print(f"\nStats:")
    print(f"  Correct: {correct_count}/{len(df_all)} ({100*correct_count/len(df_all):.1f}%)")
    print(f"  Wrong: {wrong_count}/{len(df_all)} ({100*wrong_count/len(df_all):.1f}%)")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Total tokens: {total_tokens:,}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../data/train_sft.csv")
    parser.add_argument("--output_path", type=str, default="../data/train_sft_cot.csv")
    parser.add_argument("--api_key_file", type=str, default="../api_keys.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=1024 * 32)
    parser.add_argument("--model", type=str, default="google/gemini-3-flash-preview")
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument("--initial_delay", type=float, default=2.0)
    parser.add_argument("--exponential_base", type=float, default=1.5)
    parser.add_argument("--max_delay", type=float, default=60.0)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--save_interval", type=int, default=50)

    return parser.parse_args()


if __name__ == "__main__":
    main()
