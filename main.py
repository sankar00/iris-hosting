import imghdr
from os import name
from fastapi import FastAPI, Response, UploadFile
from pydantic import BaseModel 
import time
import pickle 
import numpy as np 
import uvicorn
from models import classe

app = FastAPI()

pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)

@app.get("/")
def index():
    return {"hello": "FastAPI"}

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'hello, {name}'}

@app.post('/predict')
def predict_species(data: classe):
    Sepal_Length = data.Sepal_Length
    Sepal_Width = data.Sepal_Width
    Petal_Length = data.Petal_Length
    Petal_Width = data.Petal_Width

    prediction = classifier.predict([[Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]])

    if prediction[0] == 0:
        species = "setosa"
    elif prediction[0] == 1:
        species = "virginica"
    elif prediction[0] == 2:
        species = "versicolor"
    else:
        species = "unknown"

    return {'prediction': species}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
