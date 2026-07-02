import chess
from .moves import move_to_index
import numpy as np

def legal_move_mask(board : chess.Board):
    """ Returns a 4184 size boolean array that tells whether a move is valid from all the moves in vocab_list. """
    mask = np.array([False]*4184, dtype=bool)

    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True

    return mask