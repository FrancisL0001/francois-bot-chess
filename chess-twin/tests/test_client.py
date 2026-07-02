from fastapi.testclient import TestClient
import chess
from pathlib import Path

from chesstwin.serving.app import app
from chesstwin.config import load_config


client = TestClient(app)
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"
config = load_config(CONFIG_PATH)

def test_predict_starting_position():
    r = client.post("/predict", json={"fen": chess.STARTING_FEN})
    assert r.status_code == 200
    move = r.json()["move"]
    assert chess.Move.from_uci(move) in chess.Board().legal_moves   # legal move returned
    # print(r.json())

def test_predict_within_latency_budget():
    r = client.post("/predict", json={"fen": "rnbqkbnr/pp2pppp/3p4/8/3pP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4"})
    assert r.status_code == 200
    latency = r.json()["latency_ms"]
    assert latency <= config.serving.latency_budget_ms, f"Latency {latency} exceeds budget {config.serving.latency_budget_ms}"
    # print(f"latency: {latency:.2f}ms\n")


def test_rejects_garbage_fen():
    r = client.post("/predict", json={"fen": "not a fen"})
    assert r.status_code == 422        # clean rejection, not a 500