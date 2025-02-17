import os
import sys

# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))  # Adjust to reach root
sys.path.append(PROJECT_ROOT)

from src.utils.logging_utils import app_logger
import torch
from torch.optim.lr_scheduler import StepLR
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import  DataLoader
from src.model_loader.model_loader import Model_Loader
class Exporter:
    '''
        This class is to train T5 transformer on Grammar correction dataset.
        After training the trained model should exported and saved.
    '''
    model_path = f'{PROJECT_ROOT}/models/T5_model'
    tokenizer_path = f'{PROJECT_ROOT}/models/T5_tokenizer'
    model=None
    device=None
    optimizer=None
    scheduler=None
    tokenizer=None

    def __init__(self):
        '''
            The initializer of the class is to initialize the model and the environmet for training process.
        '''
        try:
            loader = Model_Loader()
            self.model,self.tokenizer,self.device,self.optimizer,self.scheduler = loader.load()
        except Exception as e:
            app_logger.error(f"ERROR: an Error {e} accured while initializing exporterto train the model")


    def train_epoch(self,dataloader):
        self.model = self.model.to(self.device)
        self.model.train()
        total_loss = 0
        for ungrammatical_statement, standard_english in dataloader:
            inputs = self.tokenizer(ungrammatical_statement, return_tensors = "pt", padding = True, truncation=True, max_length = 256).to(self.device)
            labels = self.tokenizer(standard_english, return_tensors = "pt", padding = True, truncation = True, max_length = 256).input_ids.to(self.device)
            outputs = self.model(**inputs, labels = labels)

            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()
        average_loss = total_loss / len(dataloader)
        return average_loss
    
    def eval_model(self, dataloader):
        self.model = self.model.to(self.device)
        self.model.eval()
        total_bleu_score = 0
        smoothing = SmoothingFunction().method1
        with torch.no_grad():
            for ungrammatical_statement, standard_english in dataloader:
                inputs = self.tokenizer(ungrammatical_statement, return_tensors = "pt", padding = True, truncation = True, max_length = 256).to(self.device)
                labels = self.tokenizer(standard_english, return_tensors = "pt", padding = True, truncation = True, max_length = 256).input_ids.to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens = 256)
                corrected_english = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
                reference = [standard_english[0].split()]
                candidate = corrected_english.split()
                bleu_score = sentence_bleu(reference, candidate, smoothing_function = smoothing)
                total_bleu_score += bleu_score
        average_bleu_score = total_bleu_score / len(dataloader)
        return average_bleu_score
    
    def exporte(self,train_dataset,val_dataset,test_dataset):
        try:
            train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
            val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle = True)
            test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = True)
            app_logger.info(f"Starting T5 model training:")
            for epoch in range(3):
                average_loss = self.train_epoch(train_dataloader)
                app_logger.info(f"        Epoch: {epoch+1}, Loss: {average_loss}")
                score = self.eval_model(val_dataloader)
                app_logger.info(f"        Epoch: {epoch+1}, BLUE SCORE: {score}")
            
            app_logger.info("Fiishing training process.")
            app_logger.info("========================================")
            app_logger.info("Evaluating the model:")
            evaluation_score = self.eval_model(test_dataloader)
            app_logger.info(f"        BLUESCORE:{evaluation_score}")
            app_logger.info("========================================")
            app_logger.info("Saving pretrained Model...")
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.tokenizer_path)
            app_logger.info("Saved Successfuly")
        except Exception as e:
            app_logger.error(f"Error: An Error {e} accured while training the model.")

        

    

# Test the Class and main Funcation 
if __name__ == "__main__":
    itransform = Exporter()