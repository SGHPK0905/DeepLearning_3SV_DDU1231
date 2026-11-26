import os
import torch


class Config:
    # Reproducibility & Hardware
    SEED = 42
    SEEDS = [42, 123, 2026]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
    USE_AMP = torch.cuda.is_available()

    # Project Metadata & Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
    EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")

    # Dataset (Flickr8k - Karpathy Split; Flickr30k deferred to Future Work)
    DATASET_NAME = "flickr8k"
    DATASET_URL = "https://www.kaggle.com/datasets/adityajn105/flickr8k"
    KARPATHY_SPLIT = {"train": 6000, "val": 1000, "test": 1000}

    # Image Preprocessing (ImageNet standard)
    RESIZE_SIZE = (256, 256)
    IMAGE_SIZE = (224, 224)
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]

    # Text & Vocabulary (fit ONLY on Train split)
    MIN_WORD_FREQ = 5
    MAX_CAPTION_LEN = 35
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
    BATCH_SIZE = 32
    LEARNING_RATE = 4e-4
    ENCODER_LR = 1e-5
    WEIGHT_DECAY = 1e-4
    EPOCHS = 25
    GRAD_CLIP = 5.0
    PATIENCE = 5  # Early stopping on validation loss / CIDEr

    # Decoding & Translation
    BEAM_SIZES = [1, 3, 5, 7]
    DEFAULT_BEAM_SIZE = 3
    NMT_MODEL_NAME = "VietAI/envit5-translation"
    EVAL_VI_SAMPLE_SIZE = 100
