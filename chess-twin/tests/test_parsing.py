import chess
from chesstwin.data.parsing import positions_from_game
from pathlib import Path

def test_all_positions_are_players_turn_and_legal():
    import chess.pgn
   
    with open(Path(__file__).parent / "data" / "sample.pgn") as pgn:
        pgn = chess.pgn.read_game(pgn)
        records = positions_from_game(pgn, "ZenHustler237", "test_0")
        assert records, "expected at least one record"
        for rec in records:
            board = chess.Board(rec.fen)
            assert (board.turn == chess.WHITE) == (rec.player_color == "white")
            assert chess.Move.from_uci(rec.move_uci) in board.legal_moves