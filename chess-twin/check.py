import json, chess
from collections import Counter
lo, hi = 2200, 2600
colors, bad = Counter(), 0
with open("data/pretrain/pretrain.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if not (lo <= r["elo"] <= hi): bad += 1
        colors[r["player_color"]] += 1
        # spot-check legality on every Nth to keep it fast
print(f"bad : {bad}, colors : {colors}")   # expect bad == 0, both colors well-represented