"""

Parses the games gotten from lichess used for pretraining
"""

from pathlib import Path 
from collections import Counter
import shutil
import chess.pgn
import json
from dataclasses import asdict
from datasets import load_dataset
import io 
import os

from chesstwin.data.lichess_stream import passes_pretrain_filter, positions_from_game_all_plies, dict_to_clean_pgn
from chesstwin.data.parsing import classify_time_control 
from chesstwin.config import load_config



def merge_shards(file_root : str, shards_dir : Path, output_file : str):
    """ Merges the jsonl shards obtained from parse_lichess_games. """

    seen = set()
    with open(shards_dir / output_file, "w") as out_f:
        for shard in sorted(shards_dir.glob(f"{file_root}*.jsonl")):
            with open(shard) as in_f:
                for line in in_f:
                    gid = json.loads(line)["game_id"]
                    if gid not in seen:        # collapse any cross-run dupes
                        out_f.write(line)
            seen.add(gid)  # NOTE: see below — this line is wrong on purpose

def parse_lichess_games(pretrain_target_positions = 10000, output_dir = Path(__file__).parent.parent / "data" / "pretrain"):
    """ Loads and processes the lichess games. """
    CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"
    config = load_config(CONFIG_PATH)
    N = 50

    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = list(output_dir.glob("pretrain_*.jsonl"))

    if written_files:
        with open(written_files[-1]) as last:
            game_ids = {json.loads(r)["game_id"] for r in last}
        if len(game_ids) < N:          # partial → discard and re-bank it
            os.remove(written_files[-1])
            written_files.pop()

    total_positions = 0

    for shard in written_files:
        with open(shard) as sh:
            rows = sh.readlines()
            total_positions += len(rows)

    num_games = 0
    total_seen = 0
    tc_counter = Counter()


    N_counter = 0
    file_id = len(written_files) 

    yr = 2025
    month = 6                           # check hugging face to see where target data dates start, ideally start from most recents

    while total_positions < pretrain_target_positions:
        data = load_dataset("Lichess/standard-chess-games",
                    data_files=f"data/year={yr}/month={month:02d}/*.parquet",  # VERIFY this path against the repo file listing
                    split="train", streaming=True
                )
        

        id = 0
        for game in data: 
            tc_counter[classify_time_control(game.get("TimeControl"))] += 1
            total_seen += 1

            if passes_pretrain_filter(
                game.get("WhiteElo", 0), 
                game.get("BlackElo", 0), 
                game.get("TimeControl", None), 
                config.transfer.pretrain_rating_band
            ):
                parsed_game = chess.pgn.read_game(io.StringIO(dict_to_clean_pgn(game)))
                N_counter += 1
                num_games += 1
                positions = positions_from_game_all_plies(parsed_game, f"Lichess_games_{file_id}_{id}")
                id += 1

                total_positions += len(positions)

                with open(output_dir / f"pretrain_{file_id:03d}.jsonl", "a") as out_f:
                    for pos in positions:
                        json.dump(asdict(pos), out_f)
                        out_f.write("\n") 

                if N_counter >= N:
                    file_id += 1
                    N_counter = 0 
                    id = 0
            
            if total_seen%10000 == 0:
                print(f"Games Seen : {total_seen} - Games Parsed : {num_games}\n")
            if total_positions >= pretrain_target_positions:
                break


        print(f"Wrote to {file_id + 1} shards.")

        month -= 1

        if month < 1:
            month = 12
            yr -= 1

    print(f"Done parsing games. Parsed {num_games} games. Saw {total_seen} games.\n")
    print(f"Seen time controls as follows: \n {tc_counter}\n")

    merge_shards("pretrain_", output_dir, "pretrain.jsonl")

    print(f"Wrote {total_positions} positions to the JSONL file.")


if __name__ == "__main__":
    needed_positions = 1000000 # Modify this as required
    parse_lichess_games(needed_positions)



                    