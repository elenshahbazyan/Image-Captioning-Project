import torch
import torch.nn as nn
from .attention import Attention

class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim):
        super().__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

    def init_hidden(self, enc_out):
        mean = enc_out.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    def forward(self, enc_out, captions):
        h, c = self.init_hidden(enc_out)
        embeddings = self.embedding(captions)
        outputs = []

        for t in range(captions.size(1) - 1):
            context, _ = self.attention(enc_out, h)
            gate = self.sigmoid(self.f_beta(h))
            lstm_input = torch.cat([embeddings[:, t], gate * context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            outputs.append(self.fc(h))

        return torch.stack(outputs, dim=1)
