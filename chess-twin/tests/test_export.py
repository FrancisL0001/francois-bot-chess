import numpy as np, torch, onnxruntime as ort
from chesstwin.config import load_config
from chesstwin.model import build_model
from chesstwin.serving.export import export_to_onnx
from pathlib import Path

artifacts_dir = Path(__file__).parent.parent / "artifacts"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

def test_onnx_matches_pytorch():
    config = load_config(CONFIG_PATH)
    model = build_model(config)
    model.load_state_dict(torch.load(artifacts_dir / "best_model_parameters.pt", map_location="cpu"))
    model.eval()

    onnx_path = export_to_onnx()
    sess = ort.InferenceSession(str(onnx_path))

    x = torch.randn(4, config.encoding.num_board_planes, 8, 8)  # batch of 4
    with torch.no_grad():
        torch_out = model(x).numpy()
    onnx_out = sess.run(["logits"], {"board": x.numpy()})[0]

    # same logits within floating-point tolerance
    np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-4)