from typing import Union,List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import librosa
import tensorflow as tf
import numpy as np
from scipy.signal import resample


model=tf.keras.models.load_model('diagnosis_GRU_CNN_1.h5')
classes = ["COPD" ,"Healthy", "Pneumoina", "URTI", "Healthy"]

def resample_audio(audio_data, original_sample_rate, target_sample_rate):
    # Calculate the resampling factor
    resampling_factor = target_sample_rate / original_sample_rate

    # Calculate the new length of the resampled signal
    new_length = int(len(audio_data) * resampling_factor)

    # Resample the audio data
    resampled_data = resample(audio_data, new_length)

    return resampled_data

def predict_class(data ,features=52):
    audio_data = np.array(data, dtype=np.int8)
    audio_float = librosa.util.buf_to_float(audio_data, dtype=np.float32)

    val = []

    mfccs = np.mean(librosa.feature.mfcc(y=audio_float, sr=1000, n_mfcc=features).T, axis=0)
    val.append(mfccs)
    val = np.expand_dims(val, axis=1)
    prediction = classes[np.argmax(model.predict(val))]
    print(prediction)
    return prediction
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "test5"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/process_data")
async def process_data(data: List[int]):
    # Your processing logic here
    mapped_values = list(map(lambda x: int(x * 255 // 4096), data))

    resampled_data_1000hz = resample_audio(mapped_values, 100, 1000)
    prediction=predict_class(resampled_data_1000hz)
    result = {"received_data": prediction}
    return result
