import torch
from pathlib import Path
from chesstwin.config import load_config
from chesstwin.device import get_device
from chesstwin.model import build_model
from chesstwin.encoding.moves import VOCAB_LIST

root = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = root / "configs" / "default.yaml"
artifacts_dir = root / "artifacts"

def export_to_onnx():
    config = load_config(CONFIG_PATH)
    # Export on CPU — avoids MPS-specific op quirks in the ONNX graph, and
    # serving will be CPU anyway. Load weights onto CPU explicitly.
    model = build_model(config)
    state = torch.load(artifacts_dir / "best_model_parameters.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()                                   # ← BatchNorm running stats, critical

    dummy = torch.randn(1, config.encoding.num_board_planes, 8, 8)  # (1,17,8,8)
    onnx_path = artifacts_dir / f"twin_{config.serving.model_version}.onnx"

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["board"],
        output_names=["logits"],
        dynamic_axes={"board": {0: "batch"}, "logits": {0: "batch"}},  # variable batch size
        opset_version=17,
    )
    print(f"Exported to {onnx_path}  (vocab={len(VOCAB_LIST)})")
    return onnx_path