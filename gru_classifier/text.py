import re
from collections import Counter

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")  # keeps don't / isn't, ! and ?


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, texts, min_freq=2):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        words = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
        return cls([PAD, UNK] + words)

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len):
        ids = [self.stoi.get(tok, UNK_ID) for tok in tokenize(text)][:max_len]
        return ids or [UNK_ID]
