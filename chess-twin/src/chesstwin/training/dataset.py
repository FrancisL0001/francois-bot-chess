import json, torch
from pathlib import Path
from torch.utils.data import Dataset
import chess
from datetime import datetime
import random

from chesstwin.encoding.board import board_to_tensor
from chesstwin.encoding.moves import move_to_index
from chesstwin.encoding.mask import legal_move_mask


class ChessPositionDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records            # already-split list of dicts

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        board = chess.Board(rec["fen"])
        x = torch.from_numpy(board_to_tensor(board))            # (17,8,8) float32
        y = move_to_index(chess.Move.from_uci(rec["move_uci"]), board)
        mask = torch.from_numpy(legal_move_mask(board))         # (4184,) bool
        return x, y, mask
    


def split_records(records : list[dict], val_fraction : float, test_fraction : float):
    """ This function splits our games into training, validation, and testing set by date and by game. """
    """ Most recent games are used for testing and next most recent games are used for validation. """

    # Grouping records by game_id
    records_by_game_id = {}
    for rec in records:
        if rec["game_id"] in records_by_game_id:
            records_by_game_id[rec["game_id"]].append(rec)
        else:
            records_by_game_id[rec["game_id"]] = [rec]

    # Sorting games by date
    games_by_date = sorted(
        records_by_game_id.items(),
        key = lambda item: datetime.strptime(item[1][0]["date"], "%Y.%m.%d"),
        reverse=True
    )

    # Get the most recent test_fraction as test and next val_fraction as validation
    games_by_date = [game[1] for game in games_by_date]
    number_of_games = len(games_by_date)

    n_test, n_val= int(test_fraction * number_of_games), int(val_fraction * number_of_games)

    return games_by_date[n_test + n_val:], games_by_date[n_test:n_val+n_test], games_by_date[:n_test]


def random_split_by_game(records: list[dict], val_fraction: float, seed: int = 42):
    """Random split by game_id — for the general pretrain corpus, where
    recency is meaningless. Whole games stay intact in one split."""
    by_game = {}
    for rec in records:
        by_game.setdefault(rec["game_id"], []).append(rec)

    game_ids = list(by_game.keys())
    random.Random(seed).shuffle(game_ids)        # seeded → reproducible

    n_val = int(val_fraction * len(game_ids))
    val_ids = set(game_ids[:n_val])

    train = [by_game[g] for g in game_ids if g not in val_ids]
    val   = [by_game[g] for g in game_ids if g in val_ids]
    return train, val          # lists of games; flatten_games as usual

def flatten_games(games : list[list[dict]]):
    """ Processes the returned train, test, and val splits into a single list of positions. """
    return [pos for game in games for pos in game]
