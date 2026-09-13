"""
Classical statistical NLP

P(next_char | previous n-1 chars) is estimated by counting occurrences in the training corpus, 
with add-k smoothing so unseen contexts don't get zero probability. 
This family of idea (n-gram + smoothing) powered NLP for decades before neural nets took over 
-- no learned representations, no gradient descent, just counting.
"""

import pickle
import random
from collections import defaultdict, Counter


class NgramLM:
    def __init__(self, n: int = 4, k: float = 0.02):
        self.n = n
        self.k = k
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = []

    def train(self, text: str):
        self.vocab = sorted(set(text))
        pad = "\n" * (self.n - 1)
        padded = pad + text
        for i in range(len(padded) - self.n + 1):
            context = padded[i:i + self.n - 1]
            char = padded[i + self.n - 1]
            self.ngram_counts[context][char] += 1
            self.context_counts[context] += 1

    def prob(self, context: str, char: str) -> float:
        v = len(self.vocab)
        numer = self.ngram_counts[context][char] + self.k
        denom = self.context_counts[context] + self.k * v
        return numer / denom

    def neg_log_likelihood(self, text: str) -> float:
        """Average -log P(char | context) over `text`, using the trained model."""
        import math
        pad = "\n" * (self.n - 1)
        padded = pad + text
        total = 0.0
        count = 0
        for i in range(len(padded) - self.n + 1):
            context = padded[i:i + self.n - 1]
            char = padded[i + self.n - 1]
            total += -math.log(self.prob(context, char))
            count += 1
        return total / count

    def generate(self, seed: str = "\n", length: int = 300) -> str:
        context = (("\n" * (self.n - 1)) + seed)[-(self.n - 1):]
        out = seed
        for _ in range(length):
            weights = [self.prob(context, c) for c in self.vocab]
            next_char = random.choices(self.vocab, weights=weights, k=1)[0]
            out += next_char
            context = (context + next_char)[-(self.n - 1):]
        return out

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "NgramLM":
        with open(path, "rb") as f:
            return pickle.load(f)