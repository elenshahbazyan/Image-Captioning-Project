import re
from collections import Counter

def basic_english(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

class Vocab:
    def __init__(self, counter, specials):
        self.itos = specials + sorted([t for t in counter if t not in specials])
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.unk_index = self.stoi["<unk>"]

    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_index)

    def __len__(self):
        return len(self.itos)

    def get_itos(self):
        return self.itos

def build_vocab(dataset):
    counter = Counter()
    for item in dataset:
        for i in range(5):
            counter.update(basic_english(item[f"caption_{i}"]))

    specials = ["<unk>", "<pad>", "<bos>", "<eos>"]
    return Vocab(counter, specials)
