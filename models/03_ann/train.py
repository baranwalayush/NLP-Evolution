import sys
import os
import math
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from tokenizer import CharTokenizer
from data_utils import load_and_split, fixed_context_batches
from model import FeedforwardLM

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tinyshakespeare.txt")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "ann_lm.pt")
TOK_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.json")

CONTEXT_SIZE = 8
BATCH_SIZE = 128
STEPS = 3000
LR = 3e-3
EVAL_EVERY = 500

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_text, val_text = load_and_split(DATA_PATH)

    tok = CharTokenizer(train_text)
    tok.save(TOK_PATH)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    model = FeedforwardLM(tok.vocab_size, context_size=CONTEXT_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training feedforward LM ({sum(p.numel() for p in model.parameters()):,} params) on {device}...")
    for step in range(1, STEPS + 1):
        x, y = fixed_context_batches(train_ids, CONTEXT_SIZE, BATCH_SIZE, device)
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS:
            model.eval()
            with torch.no_grad():
                vx, vy = fixed_context_batches(val_ids, CONTEXT_SIZE, 1024, device)
                val_loss = loss_fn(model(vx), vy).item()
            print(f"step {step:5d} | train loss {loss.item():.3f} | val ppl {math.exp(val_loss):.2f}")
            model.train()

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"Saved model to {CKPT_PATH}")
