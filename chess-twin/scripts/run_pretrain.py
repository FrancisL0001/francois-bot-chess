import chess
from pathlib import Path

from chesstwin.training.train import train, load_records
from chesstwin.config import load_config
from chesstwin.device import get_device
from chesstwin.training.dataset import random_split_by_game, flatten_games
from chesstwin.model import build_model

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

def run_pretrain():
    config = load_config(CONFIG_PATH)
    device = get_device()
    artifacts_dir = Path(__file__).parent.parent / "artifacts" ; artifacts_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(Path(config.data.processed_dir).parent / "pretrain" / "pretrain.jsonl")
    tr, val = random_split_by_game(records, config.data.val_fraction, seed=config.project.seed)
    tr, val = flatten_games(tr), flatten_games(val)
    print(f"Pretrain: {len(tr)} train / {len(val)} val positions")

    model = build_model(config).to(device)        # FRESH model — random init
    history = train(
        model, tr, val,
        lr=config.training.lr,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        patience=config.training.patience,
        early_stop_metric=config.training.early_stop_metric,
        checkpoint_path=artifacts_dir / "pretrained_trunk.pt",   # ← distinct artifact
        device=device,
        tensorboard_subdir="pretrain",                            # ← distinct logs
    )

    return history


if __name__ == "__main__":

    print("="*30 + "\n" + " "*6 + "BEGIN PRETRAINING" + " "*6 + "\n" + "="*30 + "\n") 
    history = run_pretrain()
    print("="*30 + "\n" + " "*5 + "PRETRAINING COMPLETE" + " "*6 + "\n" + "="*30 + "\n") 

    print(f"Final Store Model's Accuracy: {history["val_acc"][history["best_idx"]]:.4f}\n")

