from typing import Union

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import librosa
import tensorflow as tf
import numpy as np

model=tf.keras.models.load_model('diagnosis_GRU_CNN_1.h5')
classes = ["COPD" ,"Healthy", "Pneumoina", "URTI", "Healthy"]

def predict_class(audio_file_path, gru_model=model, classes=classes, features=52):
    val = []
    data_x, sampling_rate = librosa.load(audio_file_path , res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=features).T, axis=0)
    val.append(mfccs)
    val = np.expand_dims(val, axis=1)
    prediction = classes[np.argmax(gru_model.predict(val))]
    print(prediction)
    return prediction
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "test5"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/upload/")
async def upload_sound_file(file: UploadFile):
    if file.content_type.startswith("audio/"):
        # Process the uploaded audio file with librosa
        # For example, get the duration of the audio
        prediction=predict_class(file.file)
        return {'prediction': prediction}
    return {'prediction': "Invalid file format"} 
