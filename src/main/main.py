import os
import sys
import questionary
from fastapi import FastAPI,Request,HTTPException
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)


from src.data_pipeline.Data_Loader.loader import LoadCSVFile
from src.utils.logging_utils import app_logger
from src.data_pipeline.Transformers.transformer import Transformer
from src.data_pipeline.Exporter.exporter import Exporter
from src.predictor.predictor import Predictor

class SentenceInput(BaseModel):
    sentence: str

class Main:
    def train(self):
        """
        GET method to render the HTML form.
        """
        print(PROJECT_ROOT)
        loader = LoadCSVFile()
        df = loader.load_data(f'{PROJECT_ROOT}/Data/dataset.csv')
        transformer = Transformer()
        train_dataset,val_dataset,test_dataset = transformer.transform(df)
        exporter = Exporter()
        exporter.exporte(train_dataset,val_dataset,test_dataset)

    def run_app_to_predict(self):
        app = FastAPI()
        # Set up templates
        templates = Jinja2Templates(directory=f"{PROJECT_ROOT}/templates")

        @app.get("/", response_class=HTMLResponse)
        async def upload_form(request: Request):
            return templates.TemplateResponse("upload.html", {"request": request})

        @app.post("/predict")
        async def predict(sentence_input: SentenceInput):
            print("starting predict")
            try:
                input_text = f"{sentence_input.sentence.strip()}" if sentence_input.sentence.strip().endswith(".") else f"{sentence_input.sentence.strip()}."
                print(f"input text:{input_text}")
                predictor = Predictor()
                corrected_sentence = predictor.predict(input_text)
                print(corrected_sentence)

                return {"corrected_sentence": corrected_sentence}
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        uvicorn.run(app,host="127.0.0.1",port=8000)

if __name__ == "__main__":
    choices = ["Train the model","Run the app to predict"]
    choice = questionary.select(
        "Choose the process:",
        choices=choices,
        use_arrow_keys=True
    ).ask()
    main = Main()
    if choice == choices[0]:
        main.train()
    elif choice ==choices[1]:
        main.run_app_to_predict()
        
