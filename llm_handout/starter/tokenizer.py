"""Character-level tokenizer, vocab built from train_corpus.txt.
Falls back to raw bytes for any character not seen in that corpus, so
decode(encode(text)) == text always holds, even on unseen hidden text.
"""
import json
import os

_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "char_vocab.json")


class CharTokenizer:
    def __init__(self, stoi=None, itos=None):
        # ids 0..255 are RESERVED for raw-byte fallback (so anything
        # outside the trained vocab still round-trips exactly).
        self.stoi = stoi or {}
        self.itos = itos or {}
        self.vocab_size = 256 + len(self.stoi)

    def encode(self, text):
        ids = []
        for ch in text:
            if ch in self.stoi:
                ids.append(256 + self.stoi[ch])
            else:
                ids.extend(list(ch.encode("utf-8")))  # byte fallback
        return ids

    def decode(self, ids):
        out = []
        buf = bytearray()
        for i in ids:
            if i < 256:
                buf.append(i)
            else:
                if buf:
                    out.append(bytes(buf).decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(self.itos.get(i - 256, ""))
        if buf:
            out.append(bytes(buf).decode("utf-8", errors="replace"))
        return "".join(out)

    def save(self, path=_VOCAB_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False)


def train_from_corpus(corpus_path, max_chars=4000):
    text = open(corpus_path, encoding="utf-8").read()
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    chars = sorted(counts, key=counts.get, reverse=True)[:max_chars]
    stoi = {ch: i for i, ch in enumerate(chars)}
    tok = CharTokenizer(stoi=stoi)
    tok.save()
    return tok


def load(path=None):
    p = path or _VOCAB_PATH
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        stoi = data["stoi"]
        itos = {v: k for k, v in stoi.items()}
        return CharTokenizer(stoi=stoi, itos=itos)
    # no trained vocab yet — pure byte fallback (identical to baseline)
    return CharTokenizer()