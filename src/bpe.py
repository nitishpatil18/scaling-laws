import regex as re
import pickle
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def count_pairs(byte_word_counts):
    pair_counts = defaultdict(int)
    for word, count in byte_word_counts.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
    return pair_counts

def merge_word(word, pair, new_token):
    new_word = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(new_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)

def encode_word(word: tuple[bytes, ...], merge_ranks: dict[tuple[bytes, bytes], int]) -> tuple[bytes, ...]:
    word = list(word)
    while len(word) > 1:
        pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
        ranked = [(merge_ranks[p], i) for i, p in enumerate(pairs) if p in merge_ranks]
        if not ranked:
            break
        _, best_i = min(ranked)
        pair = pairs[best_i]
        new_token = pair[0] + pair[1]
        word = word[:best_i] + [new_token] + word[best_i + 2:]
    return tuple(word)

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        split_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(split_pattern, text)
    else:
        chunks = [text]

    byte_word_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            word = match.group()
            byte_word = tuple(bytes([b]) for b in word.encode("utf-8"))
            byte_word_counts[byte_word] += 1

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []

    pair_counts = count_pairs(byte_word_counts)
    pair_to_words = {}
    for word in byte_word_counts:
        for pair in zip(word, word[1:]):
            pair_to_words.setdefault(pair, set()).add(word)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        new_token = best_pair[0] + best_pair[1]

        affected_words = pair_to_words.get(best_pair, set()).copy()
        for word in affected_words:
            count = byte_word_counts[word]

            for pair in zip(word, word[1:]):
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(word)

            new_word = merge_word(word, best_pair, new_token)

            del byte_word_counts[word]
            byte_word_counts[new_word] = byte_word_counts.get(new_word, 0) + count

            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + count
                pair_to_words.setdefault(pair, set()).add(new_word)

        merges.append(best_pair)
        vocab[len(vocab)] = new_token

    return vocab, merges

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens if special_tokens else []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def decode(self, ids: list[int]) -> str:
        byte_pieces = [self.vocab[i] for i in ids]
        combined = b"".join(byte_pieces)
        return combined.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(tok) for tok in sorted_specials) + ")"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.byte_to_id[chunk.encode("utf-8")])
            else:
                for match in re.finditer(PAT, chunk):
                    word = match.group()
                    byte_word = tuple(bytes([b]) for b in word.encode("utf-8"))
                    encoded = encode_word(byte_word, self.merge_ranks)
                    for piece in encoded:
                        ids.append(self.byte_to_id[piece])
        return ids

    def encode_iterable(self, iterable):
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

if __name__ == "__main__":
    vocab, merges = train_bpe("/tmp/bpe_test/toy.txt", 262, ["<|endoftext|>"])
    print(merges)
