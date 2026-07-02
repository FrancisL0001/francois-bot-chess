from chesstwin.eval.blunder_rate import compute_blunder_rates
import json

if __name__ == "__main__":
    print("=" * 30 + "\n   BLUNDER-RATE MATCH\n" + "=" * 30)
    # Start with a capped run to confirm it works before the full test set.
    result = compute_blunder_rates(max_positions=300)
    print(f"\nPlayer blunder rate: {result['player_blunder_rate']:.3f}")
    print(f"Twin   blunder rate: {result['twin_blunder_rate']:.3f}")
    print(f"Twin   blunder rate Top 1 Move: {result['twin_blunder_rate_top1']:.3f}")
    print(f"Gap: {abs(result['player_blunder_rate'] - result['twin_blunder_rate']):.3f}  "
          f"Gap with Top Move: {abs(result['player_blunder_rate'] - result['twin_blunder_rate_top1']):.3f}"
          f"({result['positions']} positions, depth {result['depth']})")
    
    with open("data/blunder_rate_eval.json", "w") as f:
        json.dump(result, f)

    print("END")