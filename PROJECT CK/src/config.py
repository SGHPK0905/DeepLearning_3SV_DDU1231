import os
import torch

class Config:
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

    # Image Preprocessing
    IMAGE_SIZE = (224, 224)
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]

    # Text & Vocabulary
    MIN_WORD_FREQ = 3
    MAX_CAPTION_LEN = 30
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    # Model Hyperparameters
    EMBED_DIM = 512
    ATTENTION_DIM = 512
    DECODER_DIM = 512
    DROPOUT = 0.3

    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 4e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    GRAD_CLIP = 5.0
    PATIENCE = 5  # Early stopping

    # Inference
    BEAM_SIZE = 3

