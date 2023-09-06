import streamlit as st
import requests

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

# Affichage du résultat audio
with col2:
    if get_speech_button:
        params = dict(text_to_transform=text_to_transform)
        tts_api_url = 'http://localhost:8000/predict'
        response = requests.get(tts_api_url, params=params)
        wav_file = response.content
        st.audio(wav_file, format="audio/wav")

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
