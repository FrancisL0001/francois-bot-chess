import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path 
import json
 
from chesstwin.training.metrics import masked_cross_entropy, top1_accuracy  
from chesstwin.training.dataset import ChessPositionDataset
from chesstwin.config import load_config
from chesstwin.device import get_device
from chesstwin.training.dataset import split_records, flatten_games
from chesstwin.model import build_model



def load_records(path: Path) -> list[dict]:
    """ Loads records data from the data file. """
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
    

def train(
    model: torch.nn.Module,
    train_records: list[dict], # Already Flattened
    val_records: list[dict], # Already flattened
    *,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    early_stop_metric: str,
    checkpoint_path: Path,
    device: torch.device,
    tensorboard_subdir: str
)->dict:
    
    """ All Training Happens Here. """
    
    # Optimizer
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    # history to be exposed to caller
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "best_idx" : 0
    } 

    # Create training and validation dataloaders
    training_loader = DataLoader(
        ChessPositionDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        ChessPositionDataset(val_records),
        batch_size=batch_size, 
        num_workers=0
    ) 

    # writer 
    writer = SummaryWriter(Path(__file__).parent.parent / "eval" / "tensorboard"/ f"{tensorboard_subdir}")

    # Create trackers for early stopping
    epochs_without_improvement = 0 
    best_metric = float("-inf")

    # -----------------------------------------------------------------------
    #                           TRAINING LOOP
    #------------------------------------------------------------------------

    for epoch in range(epochs):

        # training pass
        model.train() # set model to training mode
        running_loss = 0.0
        num_batches = 0
        avg_train_accuracy = 0.0

        for batch in training_loader:
            # inputs and labels
            X, y, mask = batch

            # move them to the device
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)
        

            # zero grad
            optimizer.zero_grad()

            # forward
            logits = model(X)

            # masked_cross_entropy
            loss = masked_cross_entropy(logits, y, mask)
            acc = top1_accuracy(logits=logits, targets=y, masks=mask)

            # backpropagate
            loss.backward()

            # step
            optimizer.step()

            running_loss += loss.item()
            avg_train_accuracy += acc
            num_batches += 1
        
        avg_train_loss = running_loss / (num_batches if num_batches > 0 else 1)
        avg_train_accuracy /= num_batches if num_batches > 0 else 1

        # validation pass
        model.eval() # set model to eval mode
        val_loss = 0.0
        val_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in validation_loader:
                X, y, mask = batch

                X = X.to(device)
                y = y.to(device)
                mask = mask.to(device)

                logits = model(X)

                loss = masked_cross_entropy(logits=logits, targets=y, masks=mask)

                acc = top1_accuracy(logits=logits, targets=y, masks=mask)

                val_loss += loss.item()
                val_acc += acc

                num_batches += 1

            val_acc /= num_batches if num_batches > 0 else 1
            val_loss /= num_batches if num_batches > 0 else 1

        writer.add_scalar(
            "Loss/Train",
            avg_train_loss,
            epoch
        )

        writer.add_scalar(
            "Loss/Validation",
            val_loss,
            epoch
        )

        writer.add_scalar(
            "Accuracy/Top1",
            val_acc,
            epoch
        )

        metric = val_acc if early_stop_metric == "val_top1_move_match" else -1*val_loss

        # log per epoch train loss, validation loss and validation accuracy          
        if metric > best_metric:
            best_metric = metric
            history["best_idx"] = epoch
            epochs_without_improvement = 0
            torch.save(
                model.state_dict(),
                checkpoint_path
            )
        else:
            epochs_without_improvement += 1


        print(f"Epoch [{epoch+1}/{epochs}] - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f} - Training Loss: {avg_train_loss:.4f} - Training Accuracy: {avg_train_accuracy:.4f}")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # early stop check
        if epochs_without_improvement > patience:
            break

    writer.close()
    return history 


def main():
    """ Runnable function for finetunning phase. """

    CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"
    config = load_config(CONFIG_PATH)
    device = get_device()
    artifacts_dir = Path(__file__).parent.parent.parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(Path(config.data.processed_dir) / "games.jsonl")
    tr, val, _test = split_records(records, config.data.val_fraction, config.data.test_fraction)
    tr, val = flatten_games(tr), flatten_games(val)

    model = build_model(config).to(device)
    history = train(
        model, tr, val,
        lr=config.training.lr,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        patience=config.training.patience,
        early_stop_metric=config.training.early_stop_metric,
        checkpoint_path=artifacts_dir / "best_model_parameters.pt",
        device=device,
        tensorboard_subdir="finetune",
    )

    print(f"Final Store Model's Accuracy: {history["val_acc"][history["best_idx"]]:.4f}\n")