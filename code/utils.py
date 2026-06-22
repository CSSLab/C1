from typing import Tuple
import numpy as np
import pandas as pd
import chess
import json


def make_pre_move(row: pd.Series) -> Tuple[str, str]:
    
    fen_before = row['FEN']
    moves = row['Moves'].split(' ')
    board = chess.Board(fen_before)
    pre_move = chess.Move.from_uci(moves[0])
    board.push(pre_move)
    fen_after = board.fen()
    
    return fen_after, moves[1]


def get_piece_arrangement(fen):

    board = chess.Board(fen)
    piece_order = ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn"]
    colors = ["White", "Black"]

    pieces = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = "White" if piece.color == chess.WHITE else "Black"
            names = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
            key = f"{color} {names[piece.piece_type]}"
            if key not in pieces:
                pieces[key] = []
            pieces[key].append(chess.square_name(square))

    for key in pieces:
        pieces[key].sort()
    arrangement_parts = []
    for color in colors:
        for piece_type in piece_order:
            key = f"{color} {piece_type}"
            if key in pieces:
                arrangement_parts.append(f"{key}: {pieces[key]}")

    return ", ".join(arrangement_parts)


def get_minimal_prompt(fen):
    """Generate minimal chess puzzle prompt from FEN."""
    prompt = f"You are given a chess position in FEN: {fen}.\n"
    pieces = get_piece_arrangement(fen)
    prompt += f"Piece positions: {pieces}\n"
    board = chess.Board(fen)
    legal_moves = [move.uci() for move in board.legal_moves]
    prompt += f"Legal moves: {', '.join(legal_moves)}\n"
    prompt += "Find the best move for the side to play.\n"
    prompt += "Analyze step by step and explain your reasoning.\n"
    prompt += "Finish with a single line formatted EXACTLY as:\n"
    prompt += "FINAL_ANSWER: <answer>\n"
    prompt += "Use UCI notation (e.g., e2e4, c2b1q) for the final answer."
    return prompt


def load_api_key(cfg):

    with open(cfg.api_key_file, 'r') as f:
        api_keys = json.load(f)
        api_key = api_keys.get('openrouter', {}).get('api_key')

    return api_key


def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj