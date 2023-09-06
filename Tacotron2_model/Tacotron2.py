from app.params import *
from app.utils import *
from app.model import *
from app.registry import *
from IPython.display import Audio
import numpy as np
import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")


text = "We cross our bridges when we come to them and burn them behind us, with nothing to show for our progress except a memory of the smell of smoke, and a presumption that once our eyes watered."
# Running the TTS
mel_output, mel_length, alignment = tacotron2.encode_text(text)

np.save("mel_pour_antoine.npy", mel_output)

# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output)

# Save the waverform
# torchaudio.save('example_TTS.wav',waveforms.squeeze(1), 22050)
audio_numpy = waveforms.squeeze(1).squeeze(0).cpu().numpy()

Audio(audio_numpy, rate=22050)
