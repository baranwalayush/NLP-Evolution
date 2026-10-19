import sys
import os
import torch

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from tokenizer import CharTokenizer
from model import FeedforwardLM

CKPT_PATH = os.path.join(os.path.dirname(__file__), "ann_lm.pt")
TOK_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.json")
CONTEXT_SIZE = 8


def generate(seed: str = "ROMEO:", length: int = 400) -> str:
    tok = CharTokenizer.load(TOK_PATH)
    model = FeedforwardLM(tok.vocab_size, context_size=CONTEXT_SIZE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    return model.generate(tok, seed=seed, length=length)


if __name__ == "__main__":
    seed = sys.argv[1] if len(sys.argv) > 1 else "ROMEO:"
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    print(generate(seed=seed, length=length))
