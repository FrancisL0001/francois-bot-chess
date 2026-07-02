import chess

def _to_canonical(move: chess.Move, turn: bool) -> chess.Move:
    """Map an absolute move into the canonical (side-to-move-up) frame."""
    if turn == chess.WHITE:
        return move
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )

def _from_canonical(move: chess.Move, turn: bool) -> chess.Move:
    """Inverse of _to_canonical — square_mirror is its own inverse."""
    return _to_canonical(move, turn)


def build_moves_vocab():
    """ Returns the vocab_list with all possible moves and the move_to_idx dict that maps every move to it's index in the vocab_list. """
    vocab_list = []

    for from_sq in range(64):
        for to_sq in range(64):
            vocab_list.append((from_sq, to_sq, None))

    for file in range(8):
        source = file + 48

        if file < 7 and file > 0:
            possible_targets = [file - 1 + 56, file + 56, file + 1 + 56]
        elif file == 0:
            possible_targets =  [file + 56, file + 1 + 56]
        else:
            possible_targets = [file - 1 + 56, file + 56]

        for target in possible_targets:
            for promo in (
                chess.QUEEN,
                chess.ROOK,
                chess.BISHOP,
                chess.KNIGHT,
            ):
                vocab_list.append((source, target, promo))

    move_to_idx = {
        key : idx
        for idx, key in enumerate(vocab_list)
    }

    return (vocab_list, move_to_idx)


VOCAB_LIST, MOVE_TO_IDX = build_moves_vocab()

def move_to_index(move : chess.Move, board : chess.Board):
    """ Returns the index associated to a certain move according to our vocab list"""
    canonical_move =_to_canonical(move, board.turn)
    key = (
        canonical_move.from_square,
        canonical_move.to_square,
        canonical_move.promotion
    )

    return MOVE_TO_IDX[key]

def index_to_move(idx : int, board : chess.Board):
    """ Returns the original move associated with a canonicalized move according to our dict. """

    from_sq, to_sq, prom = VOCAB_LIST[idx]
    canonical_move = chess.Move(
        from_sq,
        to_sq,
        promotion=prom
    )

    return _from_canonical(canonical_move, board.turn)