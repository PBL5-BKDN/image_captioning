import logging
import os

from model.ImageCaptioningModelV2 import ImageCaptionModelV2
from settings import BASE_DIR, DEVICE
from train.helper import load_checkpoint

_model_instance = None


def get_model():
    global _model_instance
    if _model_instance is None:
        print("Loading model...")
        logging.debug("Loading model...")
        checkpoint_path = os.path.join(BASE_DIR, "train", "model", "best_model_transformer_v2.pth")
        _model_instance, _, _ = load_checkpoint(ImageCaptionModelV2, checkpoint_path)
        _model_instance.eval()
        _model_instance.to(DEVICE)
        print("Model loaded successfully.")
        logging.debug("Model loaded successfully.")
    return _model_instance


print("Model loader initialized.")
logging.debug("Model loader initialized.")
