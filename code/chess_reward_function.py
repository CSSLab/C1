import re
from typing import Optional

# All castling moves in UCI notation
CASTLING_MOVES = {"e1g1", "e1c1", "e8g8", "e8c8"}


def extract_chess_move_from_response(response: str) -> Optional[str]:
    """Extract UCI move from model response using strict FINAL_ANSWER format only.
    
    Only looks for: FINAL_ANSWER: e2e4
    
    Args:
        response: Model's text response
        
    Returns:
        UCI move string (e.g., "e2e4") or None if not found
    """
    # Pattern: FINAL_ANSWER: format (strict)
    best_move_pattern = r"FINAL_ANSWER:\s*([a-h][1-8][a-h][1-8][qrbn]?)"
    match = re.search(best_move_pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    
    return None


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float | dict:
    """Compute reward score for chess puzzle solution.

    Returns a dict with:
        - score: binary training reward (0.0 or 1.0)
        - binary_score: binary evaluation metric (0.0 or 1.0)

    Scoring:
    - Full correct (exact match): score=1.0, binary_score=1.0
    - Wrong move or invalid format: score=0.0, binary_score=0.0

    Args:
        data_source: Dataset name (e.g., "chess_puzzles")
        solution_str: Model's generated response text
        ground_truth: Correct UCI move (e.g., "e2e4")
        extra_info: Additional info from dataset

    Returns:
        Dict with 'score' (for RL training) and 'binary_score' (for evaluation)
    """
    try:
        # Extract predicted move from response
        predicted_move = extract_chess_move_from_response(solution_str)

        if predicted_move is None:
            return {"score": 0.0, "binary_score": 0.0}

        # Compare with ground truth (case-insensitive)
        pred_lower = predicted_move.lower()
        gt_lower = ground_truth.lower()

        # Exact match - full credit
        if pred_lower == gt_lower:
            return {"score": 1.0, "binary_score": 1.0}

        # No match (including partial matches, castling, etc.)
        return {"score": 0.0, "binary_score": 0.0}

    except Exception as e:
        print(f"Error in chess reward function: {e}")
        print(f"Ground truth: {ground_truth}")
        print(f"Solution: {solution_str}")
        return {"score": 0.0, "binary_score": 0.0}


def is_correct_response(response: str, ground_truth: str) -> bool:
    """Check if a response is fully correct (score == 1.0).

    Args:
        response: Model's generated response text
        ground_truth: Correct UCI move (e.g., "e2e4")

    Returns:
        True if response is fully correct (score == 1.0), False otherwise
    """
    result = compute_score("chess_puzzles", response, ground_truth)
    return result["score"] == 1.0


if __name__ == "__main__":

    test_cases = [
        # Exact match tests
        {
            "solution": "FINAL_ANSWER: e2e4",
            "ground_truth": "e2e4",
            "expected_score": 1.0,
            "expected_binary": 1.0,
            "desc": "exact match"
        },
        {
            "solution": "FINAL_ANSWER: d7d5",
            "ground_truth": "d7d5",
            "expected_score": 1.0,
            "expected_binary": 1.0,
            "desc": "exact match 2"
        },
        # No match tests
        {
            "solution": "FINAL_ANSWER: g1f3",
            "ground_truth": "e2e4",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "no square match"
        },
        {
            "solution": "The best move is e2e4 to control the center.",
            "ground_truth": "e2e4",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "no FINAL_ANSWER format"
        },
        # Partial matches now return 0.0 (no partial credit)
        {
            "solution": "FINAL_ANSWER: h1h8",
            "ground_truth": "h1h3",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "partial: from-square match"
        },
        {
            "solution": "FINAL_ANSWER: e2e3",
            "ground_truth": "e2e4",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "partial: from-square match 2"
        },
        {
            "solution": "FINAL_ANSWER: f2e4",
            "ground_truth": "e2e4",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "partial: to-square match"
        },
        {
            "solution": "FINAL_ANSWER: a1h8",
            "ground_truth": "e2e4",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "no match at all"
        },
        # Castling
        {
            "solution": "FINAL_ANSWER: e1g1",
            "ground_truth": "e1c1",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "castling: different castling"
        },
        {
            "solution": "FINAL_ANSWER: e1c1",
            "ground_truth": "e1g1",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "castling: different castling 2"
        },
        {
            "solution": "FINAL_ANSWER: e8g8",
            "ground_truth": "e8c8",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "castling: black kingside vs queenside"
        },
        {
            "solution": "FINAL_ANSWER: e1f1",
            "ground_truth": "e1g1",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "castling: non-castling vs castling"
        },
        # Pawn promotions
        {
            "solution": "FINAL_ANSWER: e7e8q",
            "ground_truth": "e7e8r",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "promotion: squares match, wrong piece"
        },
        {
            "solution": "FINAL_ANSWER: e7e8r",
            "ground_truth": "e7e8q",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "promotion: squares match, wrong piece 2"
        },
        {
            "solution": "FINAL_ANSWER: e7f8q",
            "ground_truth": "e7e8q",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "promotion: from-square match"
        },
        {
            "solution": "FINAL_ANSWER: d7e8q",
            "ground_truth": "e7e8q",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "promotion: to-square match"
        },
        {
            "solution": "FINAL_ANSWER: d7f8q",
            "ground_truth": "e7e8q",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "promotion: no square match"
        },
        # En passant
        {
            "solution": "FINAL_ANSWER: e5d6",
            "ground_truth": "e5d6",
            "expected_score": 1.0,
            "expected_binary": 1.0,
            "desc": "en passant: exact match"
        },
        {
            "solution": "FINAL_ANSWER: e5d6",
            "ground_truth": "e5f6",
            "expected_score": 0.0,
            "expected_binary": 0.0,
            "desc": "en passant: from-square match"
        },
    ]

    print("Testing chess reward function:")
    print("-" * 100)
    all_pass = True
    for i, test in enumerate(test_cases):
        result = compute_score("chess_puzzles", test["solution"], test["ground_truth"])
        score = result["score"]
        binary = result["binary_score"]
        score_ok = score == test['expected_score']
        binary_ok = binary == test['expected_binary']
        passed = score_ok and binary_ok
        all_pass = all_pass and passed
        status = "✓" if passed else "✗"
        print(f"{status} Test {i+1:2d}: {test['desc']}")
        print(f"           pred={test['solution'].split('FINAL_ANSWER:')[-1].strip()[:10]:10s} "
              f"gt={test['ground_truth']:10s} | score={score}, binary={binary}")
    print("-" * 100)
    print(f"All tests passed: {all_pass}")