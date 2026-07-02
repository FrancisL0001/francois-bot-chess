from chesstwin.training.dataset import split_records, flatten_games
from pathlib import Path
import json
from datetime import datetime

GAMES_FILE = Path(__file__).parent / "data" / "games.jsonl"

GAMES = []

with open(GAMES_FILE) as games:
    line = games.readline()

    while line:
        GAMES.append(json.loads(line))
        line = games.readline()





def test_split_returns_appropriate_values():
    train, val, test = split_records(GAMES, 0.1, 0.1)
    print(f"Train: {len(train)} \n Test: {len(test)}\n Val: {len(val)}\n")

    assert len(test) == len(val)
    latest_train = max(
        datetime.strptime(game[0]["date"], "%Y.%m.%d")
        for game in train
    )

    earliest_val = min(
        datetime.strptime(game[0]["date"], "%Y.%m.%d")
        for game in val
    )

    earliest_test = min(
        datetime.strptime(game[0]["date"], "%Y.%m.%d")
        for game in test
    )

    assert latest_train <= earliest_val
    assert earliest_val <= earliest_test

def test_no_game_id_overlap_between_sets():
    train, val, test = split_records(GAMES, 0.1, 0.1)
    train = flatten_games(train)
    test = flatten_games(test)
    val = flatten_games(val)

    train_ids = {}
    test_ids = {}
    val_ids = {}

    for pos in train:
        if not pos["game_id"] in train_ids:
            train_ids[pos["game_id"]] = 1

    for pos in test:
        if not pos["game_id"] in test_ids:
            test_ids[pos["game_id"]] = 1

    for pos in val:
        if not pos["game_id"] in val_ids:
            val_ids[pos["game_id"]] = 1


    train_ids = set(train_ids)
    test_ids = set(test_ids)
    val_ids = set(val_ids)

    assert len(train_ids & test_ids) == 0
    assert len(train_ids & val_ids) == 0
    assert len(val_ids & test_ids) == 0

    
        
