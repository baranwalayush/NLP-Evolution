import sys
import os

from model import NgramLM

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ngram.pkl")

def generate(seed: str = "\n", length: int = 300) -> str:
    lm = NgramLM.load(MODEL_PATH)
    return lm.generate(seed=seed, length=length)

if __name__ == "__main__":
    seed = sys.argv[1] if len(sys.argv) > 1 else "ROMEO:"
    print(generate(seed=seed, length=400))
