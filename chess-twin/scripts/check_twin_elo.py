from chesstwin.eval.twin_elo import load_twin, evaluate_gameplay

def find_bracket(results):
    """Find the two adjacent Elo levels the twin's 0.5 score falls between."""
    s = sorted(results, key=lambda r: r["elo"])
    for lo, hi in zip(s, s[1:]):
        if lo["score"] >= 0.5 >= hi["score"]:   # score falls as Elo rises
            return lo["elo"], hi["elo"]
    return None

if __name__ == "__main__":
    print("=" * 30 + "\n   TWIN ELO EVALUATION\n" + "=" * 30)
    model, device, config = load_twin()

    # --- Coarse sweep: bracket the crossover cheaply ---
    print("\n[Coarse sweep]")
    coarse_levels = [1320, 1330, 1360, 1380]
    coarse = [evaluate_gameplay(model, device, config, elo, num_games=10)
              for elo in coarse_levels]
    for r in coarse:
        print(f"  Elo {r['elo']}: score {r['score']:.3f}")

    bracket = find_bracket(coarse)
    if bracket is None:
        print("\nNo clean crossover in coarse range — twin may be below 1400 "
              "or above 2300. Inspect coarse scores and adjust levels.")
        raise SystemExit

    lo, hi = bracket
    print(f"\nCrossover bracketed between {lo} and {hi}.")

    # --- Fine sweep: more games at finer steps inside the bracket ---
    print("\n[Fine sweep]")
    step = 100
    fine_levels = list(range(lo, hi + 1, step))
    fine = [evaluate_gameplay(model, device, config, elo, num_games=40)
            for elo in fine_levels]

    print("\n=== RESULTS ===")
    for r in sorted(fine, key=lambda r: r["elo"]):
        print(f"  Elo {r['elo']}: score {r['score']:.3f}  "
              f"(W{r['wins']} D{r['draws']} L{r['losses']})")

    # Crossover estimate: the fine level whose score is closest to 0.5
    best = min(fine, key=lambda r: abs(r["score"] - 0.5))
    print(f"\nEstimated twin Elo ≈ {best['elo']} (score {best['score']:.3f} vs player's ~2400)")