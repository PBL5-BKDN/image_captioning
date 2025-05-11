import os

import torch
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))
stanford_image_paragraph_captioning_dataset_folder = os.path.join(BASE_DIR, 'dataset', 'stanford_Image_Paragraph_Captioning_dataset')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

WORD_COUNT_THRESHOLD = 5
EMBED_DIM = 200
NUM_HEADS = 4
UNITS = 256
BATCH_SIZE = 256

learning_rate = 0.001
epochs = 50
patience = 5
min_delta = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


