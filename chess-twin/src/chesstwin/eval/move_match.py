import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path 
import json

from chesstwin.config import load_config
from chesstwin.device import get_device
from chesstwin.training.dataset import flatten_games, split_records, ChessPositionDataset
from chesstwin.training.metrics import topk_accuracy
from chesstwin.training.train import load_records
from chesstwin.model import build_model

def evaluate_model():
    root = Path(__file__).parent.parent.parent.parent
    CONFIG_PATH =  root / "configs" / "default.yaml"
    artifacts_dir = root / "artifacts"
    config = load_config(CONFIG_PATH)

    device = get_device()
    
    records = load_records(Path(config.data.processed_dir) / "games.jsonl")
    _train, _val, test = split_records(records, config.data.val_fraction, config.data.test_fraction)
    test = flatten_games(test)
    print(f"Test: {len(test)} positions — NEVER seen in train or val")

    testing_loader = DataLoader(
        ChessPositionDataset(test),
        batch_size = config.training.batch_size,
        num_workers = 0
    )

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(artifacts_dir / "best_model_parameters.pt", map_location=device))
    model.eval() 

    print("Loaded pretrained trunk.")

    history = {
        "top1_acc" : [],
        "top3_acc" : [],
        "top5_acc" : []
    } 

    num_batches = 0
    # -----------------------------------------------------------------------
    #                               TESTING LOOP
    #------------------------------------------------------------------------

    for batch in testing_loader:
        X, y, mask = batch

        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)

        logits = model(X)

        cur_top_1 = topk_accuracy(logits=logits, targets=y, masks=mask, k=1)
        cur_top_3 = topk_accuracy(logits=logits, targets=y, masks=mask, k=3)
        cur_top_5 = topk_accuracy(logits=logits, targets=y, masks=mask, k=5)

        print(f"Batch {num_batches} - Top 1: {cur_top_1:.4f} - Top 3: {cur_top_3:.4f} - Top 5: {cur_top_5:.4f}\n")

        history["top1_acc"].append(cur_top_1)
        history["top3_acc"].append(cur_top_3)
        history["top5_acc"].append(cur_top_5)

        num_batches += 1 

    return history