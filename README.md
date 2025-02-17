# Grammar Correction Using T5 Transformer

## Overview

This project implements a **grammar correction model** using the **T5 Transformer** and **T5 Tokenizer**. The goal is to correct grammatical errors such as typos, incorrect tenses, and other common mistakes. The model is trained on a dataset containing **2018 samples** and deployed using **FastAPI** for easy testing.

## Features

- **T5 Transformer-based Grammar Correction**
- **Data Pipeline**
  - Load data
  - Transform and preprocess data to fit T5 input format
  - Train the model on transformed data
  - Export and save the pre-trained model
- **FastAPI Deployment**
  - Deploy the trained model using FastAPI
  - Test the grammar correction via API

## Dataset Description

The dataset consists of **36 different types of grammatical errors** commonly found in English. It is structured as follows:

| Column                      | Description                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| **Serial Number**           | A unique identifier for each entry                                                        |
| **Error Type**              | The grammatical category of the error (e.g., verb tense, punctuation, sentence structure) |
| **Ungrammatical Statement** | The incorrect sentence with grammatical mistakes                                          |
| **Standard English**        | The corrected version of the sentence                                                     |

## Installation & Setup

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/AliAlabed1/Grammar-Correction.git
cd grammar-correction
py -m pip install requirements.txt
```

### **2️⃣ Set Up Google Service Account for DVC**

Since we use **DVC (Data Version Control)** to manage model checkpoints and dataset storage, you need to configure Google Drive as a remote storage.

#### **Creating a Google Service Account**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project or create a new one
3. Navigate to **IAM & Admin > Service Accounts**
4. Click **Create Service Account**
5. Fill in the details and create the account
6. Go to the **Keys** section and click **Add Key > JSON**
7. Download the JSON file (this is your service account key)
8. Rename the file to `dvc.json` and move it to the root of the project directory

### **3️⃣ Pull the model with DVC**

After adding the service account JSON key, run the following command to fetch the dataset and model weights:

```bash
dvc pull
```

### **4️⃣ Run the Project**

Navigate to the main source directory:

```bash
cd src/main
py main.py
```

you will have 2 choices as following:
- Train the model   
- Run the app to predict  

#### **Train the Model**

If you want to **train the model**, choose the first option in the script:  
This option will load the data pipeline that is described above.  



#### **Run the app to predict**

If you want to **test the model**, choose the second option:  
This option will run FAST API app.  

  

Then, open your browser and visit:

```
http://localhost:8000
```

Here, you can interact with the API and test the grammar correction model.

## API Endpoints

The FastAPI application provides the following endpoints:

| Method   | Endpoint   | Description                                                                |
| -------- | ---------- | -------------------------------------------------------------------------- |
| **POST** | `/predict` | Takes a sentence with grammatical errors and returns the corrected version |
| **GET**  | `/`        | Opens the home page interface                                              |


---

Feel free to contribute by opening an issue or a pull request! 🚀

