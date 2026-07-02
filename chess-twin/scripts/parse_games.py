"""

Parses the raw PGN files present in data/raw into JSONL files in data/processed,
using the parsing functions defined in src/chesstwin/data/parsing.py. 
The output JSONL files will be used by the model training pipeline.

"""

import chess.pgn
from chesstwin.data.parsing import positions_from_game, passes_filters, classify_time_control
from chesstwin.data.records import PositionRecord
from chesstwin.config import Config, load_config
import json
from dataclasses import asdict

from pathlib import Path 
from collections import Counter


def parse_games(n_workers=5):
    """
    
    Parses the raw PGN files present in data/raw into JSONL files in data/processed,
    using the parsing functions defined in src/chesstwin/data/parsing.py.
    The output JSONL files will be used by the model training pipeline.

    """
    CONFIG_DIR = Path(__file__).parent.parent / "configs" / "default.yaml"
    config = load_config(CONFIG_DIR)

    files_dir = Path(config.data.raw_dir)
    output_dir = Path(config.data.processed_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = 0

    num_games = 0
    total_seen = 0
    tc_counter = Counter()

    records = []

    for pgn_file in files_dir.rglob("*.pgn"): # recursively find all PGN files in the raw data directory
        files += 1
        with open(pgn_file) as pgn:
            seen = 0
            counter = 0
            game = chess.pgn.read_game(pgn) # read the first game in the PGN file

            while game: # for each game in the PGN file
                seen += 1
                tc_counter[classify_time_control(game.headers.get("TimeControl"))] += 1
                if passes_filters(game, config): # check it passes the filters
                    counter += 1
                    game_id = f"{pgn_file.stem}_{counter}" # assign a name

                    positions = positions_from_game(game, config.player.username, game_id) # get the records from that game

                    records += positions

                game = chess.pgn.read_game(pgn) # load the next game

            num_games += counter
            total_seen += seen

        print(f"Parsed {files} files.")

    print(f"Done parsing games. Parsed {num_games} games. Saw {total_seen} games.\n")
    print(f"Seen time controls as follows: \n {tc_counter}\n")

    written = 0
    with open(output_dir / "games.jsonl", "w") as out_f: # write the JSONL file       
        for pos in records:
            json.dump(asdict(pos), out_f)
            out_f.write("\n")
            written += 1

    print(f"Wrote {written} positions to the JSONL file.")


if __name__ == "__main__":

    parse_games()



                    