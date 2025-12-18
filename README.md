# Image Captioning with CNN–RNN 

## Overview

Implementation of an automatic image captioning system using deep learning that combines convolutional and recurrent neural networks with an attention mechanism to generate natural language descriptions for images. The system is trained and evaluated on the Flickr8k dataset and is based on the **Show, Attend and Tell** framework.

## Model Architecture

- **Encoder**: Based on ResNet-18 pretrained on ImageNet
  - Removal of final pooling and classification layers to preserve spatial information
  - Adaptive average pooling to obtain fixed-size feature maps
  - Linear projection to a lower-dimensional embedding space

- **Attention Mechanism**: Soft additive (Bahdanau) attention applied over spatial features

- **Decoder**: Implemented using an LSTMCell architecture
  - Word embeddings used for textual input representation
  - Gated attention context to regulate visual information flow
  - Word-by-word caption generation

## Dataset

**Flickr8k** image captioning dataset:
- Approximately 8,000 images
- Five human-annotated captions per image
- All captions per image are used during training
- Dataset loaded using the HuggingFace `datasets` library

## Training Setup

- End-to-end training of encoder and decoder
- **Optimizer**: Adam
- **Learning rate**: 3e-4
- **Batch size**: 4
- **Epochs**: 14
- Gradient clipping applied to stabilize training
- Padding tokens ignored during loss computation
- Model checkpoints saved after each epoch

## Inference

- Caption generation performed using **beam search**
- Configurable beam size (default value of 5)
- Supports captioning of both dataset images and external images
- Uses the same preprocessing pipeline as training

## Evaluation

Evaluation performed using standard image captioning metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR score
- ROUGE-L score

Metric distributions visualized using histograms and density plots.


## Results

- The model produces grammatically coherent captions
- Captions capture key objects and actions present in images
- Attention mechanism improves interpretability and visual grounding
- Competitive performance achieved on the Flickr8k dataset

## Academic Relevance

- Demonstrates multimodal deep learning techniques
- Integrates CNNs, RNNs, and attention mechanisms
- Implements a full training, evaluation, and inference pipeline
- Suitable for a bachelor thesis or final academic project

## References

1. Vinyals, O., Toshev, A., Bengio, S., and Erhan, D. *Show and Tell: A Neural Image Caption Generator.* CVPR, 2015.
2. Xu, K. et al. *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.* ICML, 2015.
3. Lin, C.-Y. *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL, 2004.



