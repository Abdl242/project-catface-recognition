from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from catface.utils.rescale import scale_model
from catface.ml.model_creation import model_20
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers


)

app.state.model = model_20()
app.state.model = app.state.model.load_weights('model/')

@app.get("/predict")
def predict(frame):      # 1
    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """
    model = app.state.model

    frame_transformed = np.expand_dims(scale_model(frame),0)



    return {'result':model.predict(frame_transformed)}








@app.get("/")
def root():
    return {'get what your deserve': 'F U'}
