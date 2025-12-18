import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
BATCH_SIZE = 4
NUM_EPOCHS = 14
LR = 3e-4
GRAD_CLIP = 5.0

# Model
EMBED_DIM = 256
ENCODER_DIM = 256
DECODER_DIM = 512
ATTENTION_DIM = 256

# Paths
CHECKPOINT_DIR = "checkpoints"
EXAMPLES_DIR = "saved_examples"
