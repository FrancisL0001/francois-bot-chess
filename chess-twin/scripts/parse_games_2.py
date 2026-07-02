"""

Parses the raw PGN files present in data/raw into JSONL files in data/processed,
using the parsing functions defined in src/chesstwin/data/parsing.py. 
The output JSONL files will be used by the model training pipeline.

"""

import chess
import chess.pgn
from chesstwin.data.parsing import positions_from_game, passes_filters
from chesstwin.data.records import PositionRecord
from chesstwin.config import Config, load_config
import json
from dataclasses import asdict

from pathlib import Path 
import os
import shutil

import multiprocessing as mp


def merge_shards(shards_dir, output_file):
    shard_files = sorted(Path(shards_dir).glob("chess_*.jsonl"))

    with open(output_file, "w") as out_f:
        for shard in shard_files:
            with open(shard) as in_f:
                shutil.copyfileobj(in_f, out_f)


def processing_worker(args):
    """ This worker processes the games in a PGN and returns the list of positions from these games into the queue. """
    output_dir, file_name, config = args
    num_positions = 0

    with open(file_name) as pgn:
        num_games = 0

        game = chess.pgn.read_game(pgn) # read the first game in the PGN file

        while game:
            if passes_filters(game, config):
                num_games += 1
                game_id = f"{file_name}_{num_games}" # Assigning a name to each game
                positions = positions_from_game(game, config.player.username, game_id) # get all the positions from the game
            
            with open(output_dir/f"{Path(file_name).stem}.jsonl", "w") as out_f:
                for pos in positions:

                    board = chess.Board(pos.fen)
                    assert (board.turn == chess.WHITE) == (pos.player_color == "white")
                    assert chess.Move.from_uci(pos.move_uci) in board.legal_moves

                    json.dump(asdict(pos), out_f)
                    out_f.write("\n")
                    num_positions += 1

            game = chess.pgn.read_game(pgn) # load the next game

    print(f"Wrote {num_games} games from process {os.getpid()} into the queue from file {file_name}. \n")

    return num_positions


if __name__ == "__main__":

    config = load_config("/Users/francoisleutou/Desktop/CodingSpace/chess-bot-amea/server/chess-twin/configs/default.yaml")

    files_dir = Path(config.data.raw_dir)
    output_dir = Path(config.data.processed_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = list(files_dir.rglob("*.pgn"))

    print(f"We have {len(filenames)} pgns")

    # start the processing workers
    n_workers = 5
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(
            processing_worker,
            [(output_dir, f, config) for f in filenames]
        )

    # All processes have completed

    total_positions = sum(results)

    print(f"Wrote {total_positions} positions to the JSONL file.")

    # Merge all the shard files

    merge_shards(output_dir, output_dir / f"games_2.jsonl")



                    