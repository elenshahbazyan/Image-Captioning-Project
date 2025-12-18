import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from collections import Counter
import re
import os

st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("Image Captioning")
st.write("Upload an image and generate a caption.")


st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
        background-color: #6C63FF;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def basic_english(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

class Vocab:
    def __init__(self, counter, specials):
        self.itos = specials + sorted([tok for tok in counter if tok not in specials])
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}
        self.unk_index = self.stoi.get("<unk>", 0)

    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_index)

    def __len__(self):
        return len(self.itos)

    def get_itos(self):
        return self.itos

    def set_default_index(self, index):
        self.unk_index = index

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

from torchvision.models import resnet18, ResNet18_Weights

class EncoderCNN(nn.Module):
    def __init__(self, encoder_dim=256, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.encoded_image_size = encoded_image_size

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.conv = nn.Conv2d(512, encoder_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(encoder_dim)

        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, images):
        x = self.backbone(images)
        x = self.adaptive_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = x.permute(0, 2, 3, 1)
        batch_size, H, W, encoder_dim = x.size()
        x = x.view(batch_size, -1, encoder_dim)
        return x

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=256,
        dropout=0.5
    ):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # IMPORTANT: name must be decode_step (as in checkpoint)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder)
        c = self.init_c(mean_encoder)
        return h, c

def generate_caption_beam(image, encoder, decoder, vocab, beam_size=5, max_len=25):
    encoder.eval()
    decoder.eval()

    img = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_out = encoder(img)
        h, c = decoder.init_hidden_state(encoder_out)

    sequences = [[[vocab["<bos>"]], h, c, 0.0]]
    completed = []

    for _ in range(max_len):
        all_candidates = []

        for seq, h_prev, c_prev, score in sequences:
            last_token = seq[-1]

            if vocab.get_itos()[last_token] == "<eos>":
                completed.append((seq, score))
                continue

            last_token_tensor = torch.tensor([last_token], device=device)
            emb = decoder.embedding(last_token_tensor)

            context, _ = decoder.attention(encoder_out, h_prev)
            gate = decoder.sigmoid(decoder.f_beta(h_prev))
            context = gate * context

            lstm_input = torch.cat([emb.squeeze(0), context.squeeze(0)], dim=0).unsqueeze(0)
            h_new, c_new = decoder.decode_step(lstm_input, (h_prev, c_prev))

            scores = decoder.fc(h_new).log_softmax(dim=1)
            topk = torch.topk(scores, beam_size)

            for idx, logprob in zip(topk.indices[0], topk.values[0]):
                new_seq = seq + [idx.item()]
                new_score = score + logprob.item()
                all_candidates.append((new_seq, h_new, c_new, new_score))

        if not all_candidates:
            break

        sequences = sorted(all_candidates, key=lambda x: x[3], reverse=True)[:beam_size]

    if completed:
        best_seq, _ = sorted(completed, key=lambda x: x[1], reverse=True)[0]
    else:
        best_seq, _, _, _ = sequences[0]

    itos = vocab.get_itos()
    words = []
    for idx in best_seq:
        tok = itos[idx]
        if tok in ("<bos>", "<pad>"):
            continue
        if tok == "<eos>":
            break
        words.append(tok)

    return " ".join(words)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint_epoch_14.pth")

@st.cache_resource
def load_everything():
    # Build vocab exactly like notebook (from Flickr8k train captions)
    ds = load_dataset("jxie/flickr8k")
    counter = Counter()
    for sample in ds["train"]:
        for i in range(5):
            counter.update(basic_english(sample[f"caption_{i}"]))

    specials = ["<unk>", "<pad>", "<bos>", "<eos>"]
    vocab = Vocab(counter, specials)
    vocab.set_default_index(vocab["<unk>"])

    encoder = EncoderCNN(encoder_dim=256).to(device)
    decoder = DecoderWithAttention(
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=len(vocab),
        encoder_dim=256,
        dropout=0.5
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab

with st.spinner("Loading model (first run may take a few minutes)..."):
    encoder, decoder, vocab = load_everything()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption_beam(img, encoder, decoder, vocab, beam_size=5)

        st.subheader("Caption")
        st.write(caption)
