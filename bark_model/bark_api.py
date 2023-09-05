#import pandas as pd

#from taxifare.ml_logic.registry import load_model
#from taxifare.ml_logic.preprocessor import preprocess_features
import librosa
import soundfile as sf
from tempfile import NamedTemporaryFile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import AutoProcessor, AutoModel
from contextlib import asynccontextmanager
import scipy
from fastapi.responses import FileResponse, StreamingResponse
import io
import os
from app.params import PATH_BARK_WAV, PATH_BARK_DUMMY_WAV

model = dict()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # loading model
    model['processor'] = AutoProcessor.from_pretrained("suno/bark-small")
    model['model'] = AutoModel.from_pretrained("suno/bark-small")
    print('Pretrained Model loaded')
    yield
    model.clear()

app = FastAPI(lifespan=lifespan)

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# 💡 Preload the model to accelerate the predictions
#app.state.model = load_model()



@app.get("/predict")
def predict(
    text_to_transform: str,  # hello this is speech to text
    ):      #wav file
    """
    Makes a single prediction
    Takes in text and returns a wav file
    """

    inputs = model['processor'](
        text=[text_to_transform],
        return_tensors="pt", )

    speech_values = model['model'].generate(**inputs, do_sample=True)
    wav_path = os.path.join(PATH_BARK_WAV, 'bark_out.wav')
    scipy.io.wavfile.write(wav_path, rate=22050, data=speech_values.cpu().numpy().squeeze())
    return FileResponse(wav_path, media_type='audio/wav')







@app.get("/serve_wav")
def serve(
    wav_name='LJ001-0010.wav',
    ):
    dummy_wav_path = os.path.join(PATH_BARK_DUMMY_WAV, wav_name)

    dummy_wav, _ = librosa.load(dummy_wav_path, sr=22050)
    temp_file = NamedTemporaryFile(suffix=".wav",delete=False)
    sf.write(temp_file.name, dummy_wav, samplerate=22050)

    return FileResponse(temp_file.name)


@app.get("/")
def root():
    return dict(status="Running")
