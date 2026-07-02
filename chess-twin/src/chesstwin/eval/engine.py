"""Shared Stockfish bridge for evaluation metrics."""

import chess
import chess.engine
from contextlib import contextmanager


@contextmanager
def open_engine(stockfish_path: str):
    """Context manager: guarantees the engine subprocess is closed."""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        yield engine
    finally:
        engine.quit()


def evaluate_position(engine : chess.engine.SimpleEngine, board: chess.Board, depth: int) -> int:
    """Centipawn score from the perspective of the side to move.
    Mate scores are clamped to a large finite centipawn value."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].relative          # relative = from side-to-move's POV
    return score.score(mate_score=10000)    # mate → ±10000 cp, never None