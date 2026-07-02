import chess
import numpy as np

# piece type (1..6) → plane offset 0..5
_PIECE_TO_PLANE = {pt: i for i, pt in enumerate(chess.PIECE_TYPES)}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Canonical board encoding: always from the side-to-move's perspective.
    Returns float32 array of shape (17, 8, 8)."""
    b = board if board.turn == chess.WHITE else board.mirror()
    # after this, b.turn is always WHITE → white = "mine", black = "theirs"
    planes = np.zeros((17, 8, 8), dtype=np.float32)

    for square, piece in b.piece_map().items():
        rank, file = divmod(square, 8)
        offset = _PIECE_TO_PLANE[piece.piece_type]
        plane = offset if piece.color == chess.WHITE else offset + 6
        planes[plane, rank, file] = 1.0

    planes[12, :, :] = float(b.has_kingside_castling_rights(chess.WHITE))
    planes[13, :, :] = float(b.has_queenside_castling_rights(chess.WHITE))
    planes[14, :, :] = float(b.has_kingside_castling_rights(chess.BLACK))
    planes[15, :, :] = float(b.has_queenside_castling_rights(chess.BLACK))

    if b.ep_square is not None:
        r, f = divmod(b.ep_square, 8)
        planes[16, r, f] = 1.0
    return planes

"""

    Planes go from 
    white(Peon -> Knight -> Bishop -> Rook -> Queen -> King) 
        -> black(Peon -> Knight -> Bishop -> Rook -> Queen -> King)
            -> white(king-side-castling -> queen-side-castling)
                -> black(king-side-castling -> queen-side-castling)
                    -> en-passant square

"""