from chesstwin.eval.opening_overlap import determine_overlap

if __name__ == "__main__":
    print("=" * 30 + "\n" + " " * 6 + "OPENING OVERLAP" + "\n" + "=" * 30)

    result = determine_overlap()

    n = result["opening_full_moves"]
    print(f"\nOpening window: first {n} full moves ({2*n} plies)\n")
    print(f"White overlap: {result['white_overlap']:.4f}  "
          f"({result['white_positions']} positions)")
    print(f"Black overlap: {result['black_overlap']:.4f}  "
          f"({result['black_positions']} positions)")