"""
Metric 2 — Opening-repertoire overlap.

Measures how often the twin's top (argmax) move matches the player's actual
move in the opening, computed separately for White and Black on the held-out
TEST set. Reported as a position-weighted agreement rate.

Unit note: `ply` in the records is the absolute half-move index within a game
(0 = first move). "n full moves" therefore means ply < 2*n. We report at a
single canonical window (OPENING_FULL_MOVES) for the model card.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from chesstwin.config import load_config
from chesstwin.device import get_device
from chesstwin.training.dataset import flatten_games, split_records, ChessPositionDataset
from chesstwin.training.train import load_records
from chesstwin.model import build_model

root = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = root / "configs" / "default.yaml"
artifacts_dir = root / "artifacts"

# Canonical opening window for the model card: first 5 full moves = first 10 plies.
OPENING_FULL_MOVES = 5


def games_by_color(records: list[dict]) -> dict[str, list[dict]]:
    """Split position records into white-turn and black-turn lists."""
    games = {"white": [], "black": []}
    for pos in records:
        games[pos["player_color"]].append(pos)
    return games


def extract_opening_positions(positions: list[dict], full_moves: int) -> list[dict]:
    """Keep only positions within the first `full_moves` full moves.
    One full move = 2 plies, so we keep ply < 2 * full_moves."""
    ply_limit = 2 * full_moves
    return [pos for pos in positions if pos["ply"] < ply_limit]


@torch.no_grad()
def _agreement_rate(model, records: list[dict], device, batch_size: int) -> tuple[int, int]:
    """Returns (correct, total): how many of these positions the twin's argmax
    move matches the player's actual move. Position-weighted, not batch-averaged."""
    if not records:
        return 0, 0

    loader = DataLoader(
        ChessPositionDataset(records),
        batch_size=batch_size,
        num_workers=0,
    )

    correct = 0
    total = 0
    for X, y, mask in loader:
        X, y, mask = X.to(device), y.to(device), mask.to(device)
        logits = model(X).masked_fill(~mask, -1e9)   # legal-mask before argmax
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct, total


def determine_overlap() -> dict:
    """Computes White and Black opening-overlap rates on the held-out test set."""
    config = load_config(CONFIG_PATH)
    device = get_device()

    # Load the SAME test split used everywhere — third return value, never trained on.
    records = load_records(Path(config.data.processed_dir) / "games.jsonl")
    _train, _val, test = split_records(records, config.data.val_fraction, config.data.test_fraction)
    test = flatten_games(test)

    # Restrict to the opening window, then split by color.
    opening = extract_opening_positions(test, OPENING_FULL_MOVES)
    by_color = games_by_color(opening)
    print(f"Test opening positions (first {OPENING_FULL_MOVES} full moves): "
          f"{len(by_color['white'])} white / {len(by_color['black'])} black")

    # Load the final twin in eval mode (BatchNorm uses running stats).
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(artifacts_dir / "best_model_parameters.pt", map_location=device))
    model.eval()

    w_correct, w_total = _agreement_rate(model, by_color["white"], device, config.training.batch_size)
    b_correct, b_total = _agreement_rate(model, by_color["black"], device, config.training.batch_size)

    return {
        "opening_full_moves": OPENING_FULL_MOVES,
        "white_overlap": w_correct / w_total if w_total else 0.0,
        "white_positions": w_total,
        "black_overlap": b_correct / b_total if b_total else 0.0,
        "black_positions": b_total,
    }