import os
import logging
import torch
from dotenv import load_dotenv

load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
FASHION_MODEL_NAME = "rcfg/FashionBLIP-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
