#import pandas as pd

#from taxifare.ml_logic.registry import load_model
#from taxifare.ml_logic.preprocessor import preprocess_features
import librosa
import soundfile as sf
from tempfile import NamedTemporaryFile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


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
some_file_path = "/Users/alexandreselier/code/Alexandre-Selier-32/text_to_speech/raw_data/LJSpeech/wavs/LJ001-0001.wav"



@app.get("/predict")
def predict(
    text_to_transform: str,  # hello this is speech to text
    ):      #wav file
    """
    Makes a single prediction
    Takes in text and returns a wav file
    """
# 1 # get model
#    model = app.state.model
#    assert model is not None

# 2 # process text to tokens/embedding
#     X_processed = preprocess_features(text_to_transform)

# 3 # predict mel spectrogram
#     y_pred = model.predict(X_processed)

# 4 # transform mel spectrogram to audio
#     predicted_wav=hifigan.mel_to_audio(y_pred)

    #
    predicted_wav, _ = librosa.load(some_file_path, sr=22050)
    temp_file = NamedTemporaryFile(suffix=".wav",delete=False)
    sf.write(temp_file.name, predicted_wav, samplerate=22050)

    return FileResponse(temp_file.name)


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(status="Running")
    # $CHA_END




@app.get("/test")
async def main():
    return FileResponse(some_file_path)
