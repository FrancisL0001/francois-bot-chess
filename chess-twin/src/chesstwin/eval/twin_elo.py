"""Metric 4 — Twin playing Elo, measured by games vs strength-capped Stockfish."""

import chess
import chess.engine
import torch
import json
from pathlib import Path

from chesstwin.config import load_config
from chesstwin.device import get_device
from chesstwin.model import build_model
from chesstwin.eval.engine import open_engine
from chesstwin.encoding.board import board_to_tensor
from chesstwin.encoding.moves import index_to_move
from chesstwin.encoding.mask import legal_move_mask
from chesstwin.eval.blunder_rate import sample_twin_move

root = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = root / "configs" / "default.yaml"
artifacts_dir = root / "artifacts"

tracker_dir = root / "data" / "eval"
tracker_dir.mkdir(parents=True, exist_ok=True)

# Short Stockfish limit: at capped strength, a long think fights the Elo cap
# and wastes hours. This is plenty.
STOCKFISH_LIMIT = chess.engine.Limit(time=0.01)
# Safety cap so a degenerate twin game can't loop forever.
MAX_PLIES = 400

@torch.no_grad()
def argmax_twin_move(model, board, device):
    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    mask = torch.from_numpy(legal_move_mask(board)).to(device)
    logits = model(x).squeeze(0).masked_fill(~mask, -1e9)
    idx = logits.argmax().item()
    return index_to_move(idx, board)


def load_twin():
    """Load the final twin once (not per-Elo-level)."""
    config = load_config(CONFIG_PATH)
    device = get_device()
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(artifacts_dir / "best_model_parameters.pt", map_location=device))
    model.eval()
    return model, device, config


def play_one_game(model, device, engine, twin_color: chess.Color) -> float:
    """Play a single game. Returns the twin's score: 1.0 win, 0.5 draw, 0.0 loss."""
    board = chess.Board()
    plies = 0
    while not board.is_game_over() and plies < MAX_PLIES:
        if board.turn == twin_color:
            move = argmax_twin_move(model, board, device)
        else:
            move = engine.play(board, STOCKFISH_LIMIT).move 
        board.push(move)
        plies += 1

    if plies >= MAX_PLIES:               # hit the safety cap → treat as draw
        return 0.5
    winner = board.outcome().winner
    if winner is None:
        return 0.5 
    return 1.0 if winner == twin_color else 0.0


def evaluate_gameplay(model, device, config, elo: int, num_games: int) -> dict:
    """Play num_games vs Stockfish capped at `elo`, alternating twin color.
    Returns wins/draws/losses and the standard score (W + 0.5D)/N."""
    wins = draws = losses = 0

    with open_engine(config.eval_.stockfish_path) as engine:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

        for g in range(num_games):
            twin_color = chess.WHITE if g % 2 == 0 else chess.BLACK
            score = play_one_game(model, device, engine, twin_color)
            if score == 1.0:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1

            if (g + 1) % 5 == 0:
                running = (wins + 0.5 * draws) / (g + 1)
                print(f"  Elo {elo}: {g+1}/{num_games}  score={running:.3f} "
                      f"(W{wins} D{draws} L{losses})")

    score = (wins + 0.5 * draws) / num_games
    result = {"elo": elo, "games": num_games, "wins": wins,
              "draws": draws, "losses": losses, "score": score}

    with open(tracker_dir / "elo_results.jsonl", "a") as f:   # one line per completed level
        json.dump(result, f)
        f.write("\n")
    return result