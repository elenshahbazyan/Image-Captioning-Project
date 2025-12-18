import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm

from config import DEVICE
from inference import generate_caption_beam

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

smooth = SmoothingFunction().method1


def evaluate_all_metrics(
    encoder,
    decoder,
    dataset,
    vocab,
    num_samples=300,
    beam_size=5
):
    encoder.eval()
    decoder.eval()

    bleu1, bleu2, bleu3, bleu4 = [], [], [], []
    meteor, rougeL = [], []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for i in tqdm(range(num_samples), desc="Evaluating"):
        base = i * 5


        refs = []
        for k in range(5):
            cap, _ = dataset[base + k]
            tokens = [
                vocab.get_itos()[t]
                for t in cap.tolist()
                if vocab.get_itos()[t] not in ("<bos>", "<eos>", "<pad>")
            ]
            refs.append(tokens)

        _, img = dataset[base]
        pred = generate_caption_beam(
            img, encoder, decoder, vocab, beam_size=beam_size
        )
        pred_tokens = pred.split()

        bleu1.append(sentence_bleu(refs, pred_tokens, (1, 0, 0, 0), smooth))
        bleu2.append(sentence_bleu(refs, pred_tokens, (0.5, 0.5, 0, 0), smooth))
        bleu3.append(sentence_bleu(refs, pred_tokens, (1/3, 1/3, 1/3, 0), smooth))
        bleu4.append(sentence_bleu(refs, pred_tokens, (0.25,)*4, smooth))


        meteor.append(meteor_score(refs, pred_tokens))


        rouge = scorer.score(
            " ".join(pred_tokens),
            " ".join(refs[0])
        )["rougeL"].fmeasure
        rougeL.append(rouge)

    return {
        "BLEU-1": np.array(bleu1),
        "BLEU-2": np.array(bleu2),
        "BLEU-3": np.array(bleu3),
        "BLEU-4": np.array(bleu4),
        "METEOR": np.array(meteor),
        "ROUGE-L": np.array(rougeL)
    }


def plot_metric_distributions(metrics):
    sns.set(style="whitegrid", font_scale=0.8)

    plt.figure(figsize=(10, 5))
    for i, (name, scores) in enumerate(metrics.items()):
        plt.subplot(2, 3, i + 1)
        sns.histplot(scores, bins=25, kde=True)
        plt.title(name)
        plt.xlabel(name)
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
