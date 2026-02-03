# tokenize_fast.py
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

def tokenize_to_file(input_path, output_path):
    with open(output_path, "wb") as out_f:
        with open(input_path, "r") as in_f:
            for i, line in enumerate(in_f):
                ids = enc.encode(line,allowed_special={"<|endoftext|>"})
                np.array(ids, dtype=np.uint16).tofile(out_f)
                if i % 100000 == 0:
                    print(f"Line {i}...")
    print(f"Done: {output_path}")

print("Tokenizing train...")
tokenize_to_file("../data/TinyStoriesV2-GPT4-train.txt", "train.bin")

print("Tokenizing val...")
tokenize_to_file("../data/TinyStoriesV2-GPT4-valid.txt", "val.bin")

print("All done!")