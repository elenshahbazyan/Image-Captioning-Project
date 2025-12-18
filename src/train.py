from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import os

from config import *
from vocab import build_vocab, basic_english
from dataset import CaptionDataset, collate_fn
from models.encoder import EncoderCNN
from models.decoder import DecoderWithAttention

ds = load_dataset("jxie/flickr8k")

vocab = build_vocab(ds["train"])
pad_idx = vocab["<pad>"]

train_ds = CaptionDataset(ds["train"], vocab, basic_english)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, pad_idx)
)

encoder = EncoderCNN(ENCODER_DIM).to(DEVICE)
decoder = DecoderWithAttention(
    ATTENTION_DIM, EMBED_DIM, DECODER_DIM, len(vocab), ENCODER_DIM
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=LR
)


