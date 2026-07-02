from pathlib import Path
from chesstwin.training.train import train, load_records
from chesstwin.config import load_config
from chesstwin.device import get_device
from chesstwin.training.dataset import split_records, flatten_games
from chesstwin.model import build_model

data_file = Path(__file__).parent / "data" / "games2.jsonl"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

def test_overfits_tiny_batch_memorizes():
    config = load_config(CONFIG_PATH)
    device = get_device()
    artifacts_dir = Path(__file__).parent.parent / "artifacts" ; artifacts_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(data_file) 

    model = build_model(config=config).to(device=device)
    
    print("Begin Training\n")

    metrics_history = train(
        model, records, records, 
        lr = config.training.lr,
        epochs = 100,
        batch_size = 64,
        patience = 10_000,
        early_stop_metric = config.training.early_stop_metric,
        checkpoint_path = artifacts_dir / "test_model_parameters.pt",
        device = device,
        tensorboard_subdir = "test"
    )

    print("Training ended\n")

    assert metrics_history["train_loss"][-1] < 0.1