import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CaptionDataset(Dataset):
    def __init__(self, dataset, vocab, tokenizer):
        self.samples = []
        self.vocab = vocab
        self.tokenizer = tokenizer

        for item in dataset:
            for i in range(5):
                self.samples.append((item["image"], item[f"caption_{i}"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, caption = self.samples[idx]
        tokens = self.tokenizer(caption)
        ids = [self.vocab["<bos>"]] + [self.vocab[t] for t in tokens] + [self.vocab["<eos>"]]
        return torch.tensor(ids), img

def collate_fn(batch, pad_idx):
    caps, imgs = zip(*batch)
    caps = pad_sequence(caps, batch_first=True, padding_value=pad_idx)
    return caps, imgs
