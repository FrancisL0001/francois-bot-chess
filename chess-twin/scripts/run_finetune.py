import chess
from pathlib import Path
import torch

from chesstwin.training.train import train, load_records
from chesstwin.config import load_config
from chesstwin.device import get_device
from chesstwin.training.dataset import split_records, flatten_games
from chesstwin.model import build_model
from chesstwin.encoding.moves import move_to_index
from chesstwin.encoding.mask import legal_move_mask

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

def run_finetune():
    config = load_config(CONFIG_PATH)
    device = get_device()
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    # player data — recency split matters again (predicting their FUTURE moves)
    records = load_records(Path(config.data.processed_dir) / "games.jsonl")

    # quick scan over player records
    for rec in records:
        board = chess.Board(rec["fen"])
        idx = move_to_index(chess.Move.from_uci(rec["move_uci"]), board)
        mask = legal_move_mask(board)
        assert mask[idx], f"played move masked out: {rec['move_uci']} @ {rec['fen']}"

    tr, val, _test = split_records(records, config.data.val_fraction, config.data.test_fraction)
    tr, val = flatten_games(tr), flatten_games(val)
    print(f"Finetune: {len(tr)} train / {len(val)} val positions")

    # build fresh, then LOAD the pretrained trunk weights
    model = build_model(config).to(device)
    state = torch.load(artifacts_dir / "pretrained_trunk.pt", map_location=device)
    model.load_state_dict(state)          # ← the transfer: start from general chess, not random
    print("Loaded pretrained trunk.")

    # # --- Freeze early layers: adapt only the late trunk + head ---
    # def freeze_early_layers(model, num_trainable_blocks: int):
    #     # freeze everything first
    #     for p in model.parameters():
    #         p.requires_grad = False
    #     # unfreeze the LAST `num_trainable_blocks` residual blocks
    #     trainable_blocks = model.blocks[-num_trainable_blocks:] if num_trainable_blocks > 0 else []
    #     for blk in trainable_blocks:
    #         for p in blk.parameters():
    #             p.requires_grad = True
    #     # always train the policy head (conv + fc)
    #     for p in model.policy_conv.parameters():
    #         p.requires_grad = True
    #     for p in model.policy_fc.parameters():
    #         p.requires_grad = True

    # freeze_early_layers(model, num_trainable_blocks=2)   # try 2 first

    # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total = sum(p.numel() for p in model.parameters())
    # print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


    finetune_lr = config.training.lr * config.transfer.finetune_lr_factor   # 0.001 * 0.1 = 1e-4
    history = train(
        model, tr, val,
        lr=finetune_lr,                                          # ← dropped 10×
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        patience=config.training.patience,
        early_stop_metric=config.training.early_stop_metric,
        checkpoint_path=artifacts_dir / "best_model_parameters.pt",   # ← the final twin
        device=device,
        tensorboard_subdir="finetune",
    )
    return history


if __name__ == "__main__":

    print("="*30 + "\n" + " "*6 + "BEGIN FINETUNING" + " "*6 + "\n" + "="*30 + "\n") 
    history = run_finetune()
    print("="*30 + "\n" + " "*5 + "FINETUNING COMPLETE" + " "*6 + "\n" + "="*30 + "\n") 

    print(f"Final Store Model's Accuracy: {history["val_acc"][history["best_idx"]]:.4f}\n")