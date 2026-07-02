from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import chess, time
from pathlib import Path

from chesstwin.serving.inference import TwinInference
from chesstwin.serving.artifacts import fetch_model_from_r2
from chesstwin.config import load_config

root = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = root / "configs" / "default.yaml"
config = load_config(CONFIG_PATH)

version = config.serving.model_version           # e.g. "v1.0"
local = root / "artifacts" / f"twin_{version}.onnx"
ONNX_PATH = str(fetch_model_from_r2(version, local))
twin = TwinInference(ONNX_PATH)
app = FastAPI(title="Chess Twin", version="1.0")

# Constructed ONCE at import/startup — the ONNX session is expensive to build,
# cheap to call. Every request reuses this. This is why inference is stateless.
twin = TwinInference(ONNX_PATH)


class PredictRequest(BaseModel):
    fen: str = Field(..., description="Board position in FEN notation")
    temperature: float = Field(0.0, ge=0.0, le=2.0)   # 0 = argmax (safe default)


class PredictResponse(BaseModel):
    move: str | None
    confidence: float | None
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model": "twin.onnx"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1. Validate the FEN BEFORE touching the model
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid FEN")
    if not board.is_valid():
        raise HTTPException(status_code=422, detail="Illegal position")

    # 2. Inference, timed
    t0 = time.perf_counter()
    result = twin.predict(req.fen, temperature=req.temperature)
    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        move=result["move"],
        confidence=result.get("confidence"),
        latency_ms=latency_ms,
    )