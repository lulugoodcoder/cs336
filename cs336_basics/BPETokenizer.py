from mpmath import j0
from cs336_basics.PreTokenizer import PreTokenizer
from typing import Iterable, Iterator, List, Tuple, Dict

class BPETokenizer:
    # special_tokens: list[str] A list of strings to add to the vocabulary
    # vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes）
    # merges: list[tuple[bytes, bytes]] b'th', b'e'
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab  # token_id -> bytes
        self.reverse_vocab = {v: k for k, v in vocab.items()} 
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens_bytes = [token.encode("utf-8") for token in self.special_tokens]
        self.special_token_set = set(self.special_tokens_bytes)

    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        # Read vocab file
        vocab = {}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    token_id, token_bytes = line.split(maxsplit=1)
                    vocab[int(token_id)] = eval(token_bytes)

        # Read merges file
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    token1, token2 = line.split()
                    merges.append((eval(token1), eval(token2)))

        # Handle special tokens
        if special_tokens is None:
            special_tokens = []

        return self(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pre_tokenizer = PreTokenizer(self.special_tokens_bytes)
        textByte = text.encode("utf-8")
        chunks = pre_tokenizer.pretokenize(textByte)
        res = []
        # replace chunk inside merge
        for List in chunks:
            for chunk in List:
                if chunk in self.special_token_set:
                    res.append(self.reverse_vocab[chunk])
                else:
                    tokens = [bytes([t]) for t in chunk]
                    while len(tokens) > 1:
                        rank = float('inf')
                        best_pos = -1
                        for i in range (len(tokens) - 1):
                            merge = (tokens[i], tokens[i + 1])
                            if merge in self.merges:
                                idx = self.merges.index(merge)
                                if idx < rank:
                                    rank = idx
                                    best_pos = i

                        if best_pos == -1 :
                            break
                        if best_pos != -1:
                            tokens = tokens[:best_pos] + [tokens[best_pos] + tokens[best_pos + 1]] + tokens[best_pos + 2:]

                    for token in tokens:
                        res.append(self.reverse_vocab[token])
        return res            


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id

        
    def decode(self, token_ids: list[int]) -> str:
        token_bytes = []
        for num in token_ids:
            token_bytes.append(self.vocab[num])

        return b''.join(token_bytes).decode("utf-8", errors='replace')
