import torch
from chesstwin.config import load_config
from chesstwin.model import build_model
from chesstwin.encoding.moves import VOCAB_LIST

from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

def test_forward_pass_shapes():
    config = load_config(CONFIG_PATH)
    model = build_model(config)
    model.eval()
    batch = torch.randn(4, config.encoding.num_board_planes, 8, 8)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (4, len(VOCAB_LIST))      # (B, 4184)
    assert torch.isfinite(out).all()              # no NaNs/infs at init

def test_runs_on_mps():
    from chesstwin.device import get_device
    device = get_device("auto")
    config = load_config(CONFIG_PATH)
    model = build_model(config).to(device)
    batch = torch.randn(4, config.encoding.num_board_planes, 8, 8).to(device)
    out = model(batch)
    assert out.shape == (4, len(VOCAB_LIST))