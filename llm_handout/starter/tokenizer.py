"""BPE tokenizer trained on train_corpus.txt only.

Byte-level BPE: ids 0..255 are raw UTF-8 bytes (fallback, so ANY text
encodes), ids >=256 are learned merges. decode(encode(text)) == text
exactly for arbitrary UTF-8 input.
"""
import json
import os
import re
from collections import Counter

_MERGES_PATH = os.path.join(os.path.dirname(__file__), "bpe_merges.json")

# chunk the text before merging so merges never span whitespace/word
# boundaries. the trailing "." (with DOTALL) guarantees every character is
# covered -> concatenation of chunks == original text, always.
_SPLIT_RE = re.compile(r"\s?\w+|\s?[^\s\w]+|\s+|.", re.UNICODE | re.DOTALL)


class BPETokenizer:
    def __init__(self, merges=None):
        # merges: list of [a, b] in learned order; token id = 256 + rank
        self.merges = merges or []
        self.ranks = {(a, b): i for i, (a, b) in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)
        # id -> bytes, for decoding
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache = {}

    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            new_id = 256 + best_rank
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        ids = []
        for m in _SPLIT_RE.finditer(text):
            ids.extend(self._encode_chunk(m.group()))
        return ids

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            buf.extend(self.vocab.get(i, b""))
        return bytes(buf).decode("utf-8", errors="replace")

    def save(self, path=_MERGES_PATH):
        with open(path, "w") as f:
            json.dump({"merges": self.merges}, f)


def _get_stats(words, counts):
    stats = Counter()
    for w, c in zip(words, counts):
        for pair in zip(w, w[1:]):
            stats[pair] += c
    return stats


def _merge_word(w, pair, new_id):
    out, i = [], 0
    while i < len(w):
        if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(w[i])
            i += 1
    return tuple(out)


def train_from_corpus(corpus_path, vocab_size=2048, sample_bytes=1_000_000):
    """Learn merges on (a prefix of) the corpus. sample_bytes keeps training
    fast; merge quality saturates well before the full 7MB."""
    text = open(corpus_path, encoding="utf-8").read()
    if sample_bytes and len(text) > sample_bytes:
        text = text[:sample_bytes]
    freqs = Counter(m.group() for m in _SPLIT_RE.finditer(text))
    words = [tuple(w.encode("utf-8")) for w in freqs]
    counts = list(freqs.values())

    n_merges = max(0, vocab_size - 256)
    merges = []
    for k in range(n_merges):
        stats = _get_stats(words, counts)
        if not stats:
            break
        pair, freq = stats.most_common(1)[0]
        if freq < 2:
            break
        new_id = 256 + k
        merges.append([pair[0], pair[1]])
        words = [_merge_word(w, pair, new_id) if pair[0] in w else w
                 for w in words]
    tok = BPETokenizer(merges=merges)
    tok.save()
    return tok


def load(path=None):
    p = path or _MERGES_PATH
    if os.path.exists(p):
        with open(p) as f:
            data = json.load(f)
        return BPETokenizer(merges=data["merges"])
    return BPETokenizer(merges=[])  # pure byte fallback