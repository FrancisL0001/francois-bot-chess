import chess
import chess.pgn
from pathlib import Path
from datasets import load_dataset
import io

from chesstwin.data.lichess_stream import passes_pretrain_filter, positions_from_game_all_plies, dict_to_clean_pgn
from chesstwin.training.train import load_records


def test_pretrain_data_loader_loads_valid_data():
    data = load_dataset(
        "Lichess/standard-chess-games",
        data_files=f"data/year=2025/month=06/*.parquet",  # VERIFY this path against the repo file listing
        split="train", streaming=True
    )

    lo, hi = 2200, 2600

    sample = []
    count = 0
    for game in data:
        parsed_game = chess.pgn.read_game(io.StringIO(dict_to_clean_pgn(game)))
        if passes_pretrain_filter(
            parsed_game.headers.get("WhiteElo", 0), 
            parsed_game.headers.get("BlackElo", 0), 
            parsed_game.headers.get("TimeControl", None), 
            [lo, hi]
        ):
            
            sample += positions_from_game_all_plies(parsed_game, f"game_{count}")
            count += 1
        if count > 100:
            break

    for rec in sample:
        assert lo <= rec.elo <= hi # filter was applied
        board = chess.Board(rec.fen)
        assert chess.Move.from_uci(rec.move_uci) in board.legal_moves # all labels are legal
    assert {rec.player_color for rec in sample} == {"white", "black"}
