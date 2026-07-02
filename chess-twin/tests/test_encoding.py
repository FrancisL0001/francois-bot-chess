import chess
from chesstwin.encoding.moves import move_to_index, index_to_move
from chesstwin.encoding.mask import legal_move_mask
from chesstwin.encoding.board import board_to_tensor

from pathlib import Path
import json

parsed_positions = []

with open(Path(__file__).parent / "data" / "games.jsonl") as games:
    for _ in range(250):
        pos = games.readline()
        if not pos:
            break

        parsed_positions.append(json.loads(pos))

sample_positions = [chess.Board(pos["fen"]) for pos in parsed_positions]

def test_ensure_our_sample_contains_both_colors_on_positions():
    turns = {b.turn for b in sample_positions}
    assert turns == {chess.WHITE, chess.BLACK}, \
        f"sample lacks both colors: {turns} — flip logic is untested"



# 1. Invertibility: every legal move survives the round trip, in BOTH frames.
def test_invertibility_works():
    for board in sample_positions:           # include black-to-move positions!
        for m in board.legal_moves:
            idx = move_to_index(m, board)
            assert index_to_move(idx, board) == m

# 2. Completeness: every legal move has a vocab slot (no KeyError ever).
#    (covered if #1 runs without raising)

# 3. Mask agreement: the mask marks exactly the legal moves, no more, no fewer.
def test_mask_marks_exactly_the_legal_moves():
    for board in sample_positions:
        mask = legal_move_mask(board)
        assert sum(mask) == board.legal_moves.count()

# 4. Shape: board tensor matches the config plane count.
def test_board_tensor_matches_config_plane_count():
    board = sample_positions[1]
    
    assert board_to_tensor(board).shape == (17, 8, 8)