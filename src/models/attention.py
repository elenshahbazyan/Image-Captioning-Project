import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.enc = nn.Linear(encoder_dim, attention_dim)
        self.dec = nn.Linear(decoder_dim, attention_dim)
        self.full = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, hidden):
        att = self.full(self.relu(self.enc(encoder_out) + self.dec(hidden).unsqueeze(1)))
        alpha = self.softmax(att.squeeze(2))
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha
