import streamlit as st

import requests

'''
Front for text to speech'''

text_to_transform= st.text_input(label='text to speak',value='sample')

params = dict(
    text_to_transform=text_to_transform)
def get_audio():
    tts_api_url = 'http://localhost:8000/predict'

    response = requests.get(tts_api_url, params=params)

    wav_file = response.content
    st.audio(wav_file, format="audio/wav")

st.button('get speech',on_click=get_audio)
