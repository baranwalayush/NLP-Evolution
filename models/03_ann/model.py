"""
The first neural network language model (Bengio et al., 2003): a plain feedforward
net over a FIXED context window.

The key idea here is the embedding table, instead of treating each character as an independent symbol
(as the n-gram model does), the net learns a dense vector per character, so
it can generalize across similar contexts it never saw during training.

The default fixed context size for this architecture has been set to 8, means the model can only look 
8 characters behind for generating the embeddings of the current character.
"""

import torch
import torch.nn as nn

class FeedforwardLM(nn.Module):

    def __init__(self, vocab_size: int, context_size: int = 8, emb_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.context_size = context_size
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(context_size * emb_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch, context_size) of token ids
        emb = self.embed(x).view(x.size(0), -1)   # concatenate the context embeddings
        h = self.act(self.fc1(emb))
        return self.fc2(h)                         # (batch, vocab_size) logits

    @torch.no_grad()
    def generate(self, tokenizer, seed: str, length: int = 300, device="cpu"):
        self.eval()
        pad_id = tokenizer.stoi.get("\n", 0)
        context = [pad_id] * self.context_size
        context += tokenizer.encode(seed)
        context = context[-self.context_size:]
        out_ids = list(tokenizer.encode(seed))
        for _ in range(length):
            x = torch.tensor([context], device=device)
            logits = self(x)
            probs = torch.softmax(logits[0], dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            out_ids.append(next_id)
            context = context[1:] + [next_id]
        return tokenizer.decode(out_ids)
