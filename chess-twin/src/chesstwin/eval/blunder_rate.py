"""Metric 3 — blunder-rate match between player and twin."""

import chess
import torch
from pathlib import Path

from chesstwin.config import load_config
from chesstwin.device import get_device
from chesstwin.training.dataset import split_records, flatten_games
from chesstwin.training.train import load_records
from chesstwin.model import build_model
from chesstwin.encoding.board import board_to_tensor
from chesstwin.encoding.moves import index_to_move
from chesstwin.encoding.mask import legal_move_mask
from chesstwin.eval.engine import open_engine, evaluate_position

root = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = root / "configs" / "default.yaml"
artifacts_dir = root / "artifacts"


def centipawn_loss(engine, board: chess.Board, move: chess.Move, depth: int) -> int:
    """How many centipawns the mover loses by playing `move`.
    Positive = the move worsened the mover's position."""
    eval_before = evaluate_position(engine, board, depth)   # mover's POV
    board.push(move)
    # now it's the opponent's turn; their relative eval, negated, is the mover's eval
    eval_after = -evaluate_position(engine, board, depth)
    board.pop()
    return eval_before - eval_after                          # drop from mover's POV


@torch.no_grad()
def sample_twin_move(model, board: chess.Board, device) -> chess.Move:
    """Sample a move from the twin's masked probability distribution.
    Sampling (not argmax) reproduces human variability — the point of a twin."""
    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    mask = torch.from_numpy(legal_move_mask(board)).to(device)
    logits = model(x).squeeze(0).masked_fill(~mask, -1e9)
    probs = torch.softmax(logits, dim=0)
    idx = torch.multinomial(probs, num_samples=1).item()
    return index_to_move(idx, board)

@torch.no_grad()
def top_twin_move(model, board: chess.Board, device) -> chess.Move:
    """Sample a move from the twin's masked probability distribution.
    Sampling (not argmax) reproduces human variability — the point of a twin."""
    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    mask = torch.from_numpy(legal_move_mask(board)).to(device)
    logits = model(x).squeeze(0).masked_fill(~mask, -1e9)
    idx = logits.argmax()
    return index_to_move(idx, board)


def compute_blunder_rates(max_positions: int | None = None) -> dict:
    config = load_config(CONFIG_PATH)
    device = get_device()
    depth = config.eval_.analysis_depth
    threshold = config.eval_.blunder_threshold_cp

    records = load_records(Path(config.data.processed_dir) / "games.jsonl")
    _train, _val, test = split_records(records, config.data.val_fraction, config.data.test_fraction)
    test = flatten_games(test)
    if max_positions:
        test = test[:max_positions]      # cap for a faster first run

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(artifacts_dir / "best_model_parameters.pt", map_location=device))
    model.eval()

    player_blunders = twin_blunders = twin_blunders_top = total = 0

    with open_engine(config.eval_.stockfish_path) as engine:
        for i, rec in enumerate(test):
            board = chess.Board(rec["fen"])

            # player's actual move
            player_move = chess.Move.from_uci(rec["move_uci"])
            if centipawn_loss(engine, board, player_move, depth) >= threshold:
                player_blunders += 1

            # twin's sampled move from the SAME position
            twin_move = sample_twin_move(model, board, device)
            if centipawn_loss(engine, board, twin_move, depth) >= threshold:
                twin_blunders += 1

            # twin's top move from the SAME position
            twin_top_move = top_twin_move(model, board, device)
            if centipawn_loss(engine, board, twin_top_move, depth) >= threshold:
                twin_blunders_top += 1

            total += 1
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(test)}  player={player_blunders/total:.3f}  twin={twin_blunders/total:.3f} twin_top1={twin_blunders_top/total:.3f}")

    return {
        "player_blunder_rate": player_blunders / total,
        "twin_blunder_rate": twin_blunders / total,
        "twin_blunder_rate_top1": twin_blunders_top / total,
        "positions": total,
        "depth": depth,
        "threshold_cp": threshold,
    }