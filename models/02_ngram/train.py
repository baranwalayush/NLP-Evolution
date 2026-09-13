import sys
import os
import math

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from data_utils import load_and_split
from model import NgramLM 

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tinyshakespeare.txt")
OUT_PATH = os.path.join(os.path.dirname(__file__), "ngram.pkl")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    train_text, val_text = load_and_split(DATA_PATH)

    print(f"Training {n}-gram model on {len(train_text):,} chars...")
    lm = NgramLM(n=n, k=0.02)
    lm.train(train_text)

    val_nll = lm.neg_log_likelihood(val_text[:20000])
    print(f"Validation perplexity: {math.exp(val_nll):.2f}")

    lm.save(OUT_PATH)
    print(f"Saved model to {OUT_PATH}")
