import torch;

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

PREDICT_SIZE = 68 * 2;
X_COLS_LEN = 68;
IMG_SIZE = 128;
