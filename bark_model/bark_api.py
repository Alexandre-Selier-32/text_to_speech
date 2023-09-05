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

app = FastAPI()

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
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small")

    inputs = processor(
        text=[text_to_transform],
        return_tensors="pt",
    )


    speech_values = model.generate(**inputs, do_sample=True)


    temp_file = NamedTemporaryFile(suffix=".wav",delete=False)
    sf.write(temp_file.name, speech_values, samplerate=22050)

    return FileResponse(temp_file.name)


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(status="Running")
    # $CHA_END



@app.get("/test")
async def main():
    return FileResponse(some_file_path)
