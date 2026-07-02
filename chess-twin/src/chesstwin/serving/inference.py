import chess
import numpy as np
import onnxruntime as ort
from chesstwin.encoding.board import board_to_tensor
from chesstwin.encoding.mask import legal_move_mask
from chesstwin.encoding.moves import index_to_move

class TwinInference:
    """Stateless twin inference: FEN → move. Loads the ONNX model once."""
    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path)

    def predict(self, fen: str, temperature: float = 0.0) -> dict:
        board = chess.Board(fen)
        if board.is_game_over():
            return {"move": None, "reason": "game_over"}

        x = board_to_tensor(board)[None, ...].astype(np.float32)   # (1,17,8,8)
        logits = self.sess.run(["logits"], {"board": x})[0][0]      # (4184,)

        mask = legal_move_mask(board)                # bool (4184,)
        logits = np.where(mask, logits, -1e9)        # ← MANDATORY legal mask

        if board.fullmove_number > 4 and temperature <= 0.0: # introduce some randomness in the more player-accurate opening sequence
            idx = int(logits.argmax())               # ← argmax: the serving default
        else:
            scaled = logits / (temperature + 1e-8)
            probs = np.exp(scaled - scaled.max())
            probs /= probs.sum()
            idx = int(np.random.choice(len(probs), p=probs))

        move = index_to_move(idx, board)
        # full legal distribution, useful for the API to return top-k or analysis
        legal_probs = _softmax(logits)
        return {
            "move": move.uci(),
            "confidence": float(legal_probs[idx]),
        }

def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()