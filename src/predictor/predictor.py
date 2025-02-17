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
from src.model_loader.model_loader import Model_Loader
class Predictor:
    model_path = f'{PROJECT_ROOT}/models/T5_model'
    tokenizer_path = f'{PROJECT_ROOT}/models/T5_tokenizer'
    model=None
    device=None
    optimizer=None
    scheduler=None
    tokenizer=None

    def __init__(self):
        try:
           loader = Model_Loader()
           self.model,self.tokenizer,self.device,self.optimizer,self.scheduler = loader.load()
        except Exception as e:
            app_logger.error(f"ERROR: an Error {e} accured while initializing exporterto train the model")
    def predict(self,input_text:str):
        try:
            if not isinstance(input_text,str):
                app_logger.error(f"The input text isn't string")
                raise TypeError("The input text isn't string")
                
        except Exception as e:
            app_logger(f"ERROR: an Error accured: {e}")

        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

            # Generate the output
            output_ids = self.model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)

            # Decode the output
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output_text
        except Exception as e:
            app_logger.error(f"ERROR: an error accured: {e}")
        
    
