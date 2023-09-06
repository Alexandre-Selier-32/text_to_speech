import streamlit as st
import requests
from app.params import PATH_Tacatron2_DUMMY_WAV
import os
# Personnaliser le thème Streamlit

st.set_page_config(
    page_title='Text-to-Speech Demo',
    page_icon="🔊",
    layout="wide",
)

# Titre de l'application
st.title('Text-to-Speech Demo')

# Diviser la mise en page en deux colonnes
col1, col2 = st.columns(2)

# Zone de texte pour entrer le texte à convertir en discours
with col1:
    text_to_transform = st.text_input('Text to speak', 'sample')
    get_speech_button = st.button('Get Speech', help='Cliquez pour obtenir la conversion en discours')
    get_dummy_speech_button = st.button('Get Dummy Speech', help='Cliquez pour obtenir une conversion déja crée en discours')

# Instructions
st.markdown('**Instructions:** Entrez le texte que vous souhaitez convertir en discours dans la zone de texte ci-dessus et cliquez sur le bouton "Get Speech".')

# Style CSS personnalisé pour le bouton
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #008CBA;
        color: white;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Affichage du résultat audio
if get_speech_button:
    params = dict(text_to_transform=text_to_transform)
    tts_api_url = 'https://docker-tacotron2-2tg6hvtuea-ew.a.run.app/predict?text_to_transform='
    response = requests.get(tts_api_url, params=params)
    wav_file = response.content
    st.audio(wav_file, format="audio/wav")

if get_dummy_speech_button:
    dummy_wav_path = os.path.join(PATH_Tacatron2_DUMMY_WAV, 'LJ001-0010.wav')
    dummy_wav_file = dummy_wav_path
    st.audio(dummy_wav_file, format="audio/wav")
