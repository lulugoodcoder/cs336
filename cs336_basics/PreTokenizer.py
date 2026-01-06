from collections import defaultdict
import regex
from typing import List, Tuple, Dict


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class PreTokenizer:
    def __init__(self, special_tokens: List[bytes]):
        self.special_tokens = list(special_tokens)
        self.special_token_set = set(special_tokens)

    def gpt2_pretokenize(self, text: bytes) -> List[bytes]:
        s = text.decode("utf-8", errors="ignore")
        return [m.group().encode("utf-8") for m in regex.finditer(GPT2_SPLIT_PATTERN, s)]
    
    def pretokenize(self, text: bytes) -> List[List[bytes]]:
        pieces = self.split_on_special_tokens(text)
        
        chunks: List[List[bytes]] = []
        for piece in pieces:
            if piece in self.special_token_set:
                chunks.append([piece])
            else:
                tokens = self.gpt2_pretokenize(piece)
                if tokens:
                    chunks.append(tokens)
        return chunks

    def split_on_special_tokens(self, text: bytes) -> List[bytes]:
        if not self.special_tokens:
            return [text]
    
        import re
        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
        escaped = [re.escape(t) for t in sorted_tokens]
        pattern = b"(" + b"|".join(escaped) + b")"
        result = [x for x in re.split(pattern, text) if x]
        return result
    
    def bytes_to_ids(self, chunks: List[List[bytes]]):
        token_byte_map = {}
        byte_to_id = {}

        for i in range(256):
            b = bytes([i])
            token_byte_map[i] = b
            byte_to_id[b] = i

        next_id = 256

        for t in self.special_tokens:
            if t not in byte_to_id:
                byte_to_id[t] = next_id
                token_byte_map[next_id] = t
                next_id += 1

        id_chunks = []

        for chunk in chunks:
            for tok in chunk:
                if tok in self.special_token_set:
                    id_chunks.append([byte_to_id[tok]])
                else:
                    chunk_ids = [byte_to_id[bytes([b])] for b in tok]
                    if chunk_ids:
                        id_chunks.append(chunk_ids)

        return id_chunks, token_byte_map, byte_to_id, next_id

    def build_pair_index(self, chunks):
        pair_counts = defaultdict(int)
        pair_positions = defaultdict(set)

        for cid, chunk in enumerate(chunks):
            for i in range(len(chunk) - 1):
                pair = (chunk[i], chunk[i + 1])
                pair_counts[pair] += 1
                pair_positions[pair].add((cid, i))

        return pair_counts, pair_positions

    def apply_merge(self, A, B, C, chunks, pc, pp):
        """Apply merge by rebuilding pair index for affected chunks."""
        affected_chunks = {cid for cid, _ in pp.get((A, B), set())}

        if not affected_chunks:
            return

        # Step 1: Remove ALL pairs from affected chunks
        for cid in affected_chunks:
            chunk = chunks[cid]
            for i in range(len(chunk) - 1):
                pair = (chunk[i], chunk[i + 1])
                if pair in pc:
                    pc[pair] -= 1
                    if pair in pp:
                        pp[pair].discard((cid, i))
                    if pc[pair] == 0:
                        del pc[pair]
                        if pair in pp:
                            del pp[pair]

        # Step 2: Perform merges in each affected chunk
        for cid in affected_chunks:
            chunk = chunks[cid]
            new_chunk = []
            i = 0
            while i < len(chunk):
                if i < len(chunk) - 1 and chunk[i] == A and chunk[i + 1] == B:
                    new_chunk.append(C)
                    i += 2
                else:
                    new_chunk.append(chunk[i])
                    i += 1
            chunks[cid] = new_chunk

        # Step 3: Rebuild pairs for affected chunks
        for cid in affected_chunks:
            chunk = chunks[cid]
            for i in range(len(chunk) - 1):
                pair = (chunk[i], chunk[i + 1])
                pc[pair] = pc.get(pair, 0) + 1
                if pair not in pp:
                    pp[pair] = set()
                pp[pair].add((cid, i))

    def train_bpe(
        self,
        input_path: str,
        vocab_size: int,
    ) -> Tuple[Dict[bytes, int], List[Tuple[bytes, bytes]]]:

        with open(input_path, "rb") as f:
            text = f.read()

        byte_chunks = self.pretokenize(text)
        chunks, token_byte_map, byte_to_id, next_id = self.bytes_to_ids(byte_chunks)

        pair_counts, pair_positions = self.build_pair_index(chunks)

        merges: List[Tuple[bytes, bytes]] = []
        
        while len(token_byte_map) < vocab_size and pair_counts:

            best_pair = max(
                pair_counts.items(),
                key=lambda x: (
                    x[1],
                    (token_byte_map[x[0][0]], token_byte_map[x[0][1]])
                ),
            )[0]

            A, B = best_pair
            C = next_id
            next_id += 1

            merges.append((token_byte_map[A], token_byte_map[B]))
            token_byte_map[C] = token_byte_map[A] + token_byte_map[B]

            self.apply_merge(A, B, C, chunks, pair_counts, pair_positions)

        vocab = {i: b for i, b in token_byte_map.items()}

        return vocab, merges