"""
Utility functions for loading and splitting text data.
"""

def load_and_split(path: str, train_frac: float = 0.9):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    n = int(len(text) * train_frac)
    return text[:n], text[n:]

def fixed_context_batches(data_ids, context_size: int, batch_size: int, device="cpu"):
    """Sample (fixed-window context, next-char target) pairs -- for the Bengio-style feedforward LM."""
    import torch
    ix = torch.randint(len(data_ids) - context_size - 1, (batch_size,))
    x = torch.stack([data_ids[i:i + context_size] for i in ix])
    y = torch.stack([data_ids[i + context_size] for i in ix])
    return x.to(device), y.to(device)