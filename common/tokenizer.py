"""
Shared character-level tokenizer.

Every model in this project (except Eliza, which doesn't need one) will be trained
and compared at the character level on the SAME vocabulary. 

This would make the perplexity scores across eras comparable.
"""
import json


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.stoi, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            stoi = json.load(f)
        obj = cls.__new__(cls)
        obj.stoi = stoi
        obj.itos = {int(i): ch for ch, i in stoi.items()}
        obj.vocab_size = len(stoi)
        return obj
