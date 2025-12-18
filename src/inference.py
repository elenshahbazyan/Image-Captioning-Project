import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from config import DEVICE

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def generate_caption_beam(
    image,
    encoder,
    decoder,
    vocab,
    beam_size=5,
    max_len=25
):
    encoder.eval()
    decoder.eval()

    img = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        encoder_out = encoder(img)
        h, c = decoder.init_hidden(encoder_out)

    sequences = [[[vocab["<bos>"]], h, c, 0.0]]
    completed = []

    for _ in range(max_len):
        all_candidates = []

        for seq, h_prev, c_prev, score in sequences:
            last_token = seq[-1]

            if vocab.get_itos()[last_token] == "<eos>":
                completed.append((seq, score))
                continue

            emb = decoder.embedding(
                torch.tensor([last_token], device=DEVICE)
            )

            context, _ = decoder.attention(encoder_out, h_prev)
            gate = decoder.sigmoid(decoder.f_beta(h_prev))
            lstm_input = torch.cat(
                [emb.squeeze(0), gate * context.squeeze(0)],
                dim=0
            ).unsqueeze(0)

            h_new, c_new = decoder.lstm(lstm_input, (h_prev, c_prev))
            scores = decoder.fc(h_new).log_softmax(dim=1)

            topk = torch.topk(scores, beam_size)
            for idx, logp in zip(topk.indices[0], topk.values[0]):
                all_candidates.append(
                    (seq + [idx.item()], h_new, c_new, score + logp.item())
                )

        if not all_candidates:
            break

        sequences = sorted(all_candidates, key=lambda x: x[3], reverse=True)[:beam_size]

    best_seq = (
        sorted(completed, key=lambda x: x[1], reverse=True)[0][0]
        if completed else sequences[0][0]
    )

    words = []
    for idx in best_seq:
        tok = vocab.get_itos()[idx]
        if tok in ("<bos>", "<pad>"):
            continue
        if tok == "<eos>":
            break
        words.append(tok)

    return " ".join(words)


def caption_external_image(
    image_path,
    encoder,
    decoder,
    vocab,
    beam_size=5
):
    img = Image.open(image_path).convert("RGB")

    caption = generate_caption_beam(
        img, encoder, decoder, vocab, beam_size=beam_size
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(caption)
    plt.show()

    return caption
