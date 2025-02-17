import os
import sys

# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))  # Adjust to reach root
sys.path.append(PROJECT_ROOT)

from src.utils.logging_utils import app_logger
import torch
from torch.optim.lr_scheduler import StepLR
from transformers import T5Tokenizer, T5ForConditionalGeneration
class Model_Loader:

    def load(self):
        try:
            model_path = f'{PROJECT_ROOT}/models/T5_model'
            tokenizer_path = f'{PROJECT_ROOT}/models/T5_tokenizer'
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
            scheduler = StepLR(optimizer, step_size = 1, gamma = 0.1)
            
            for param in model.shared.parameters():
                param.requires_grad = False
            return model,tokenizer,device,optimizer,scheduler
        except Exception as e:
            app_logger.error(f"ERROR: an Error {e} accured while initializing exporterto train the model")