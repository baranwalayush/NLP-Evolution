"""
Utility functions for loading and splitting text data.
"""

def load_and_split(path: str, train_frac: float = 0.9):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    n = int(len(text) * train_frac)
    return text[:n], text[n:]