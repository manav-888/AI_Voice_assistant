from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from audio_recorder_streamlit import audio_recorder

import os
from faster_whisper import WhisperModel

import pygame
from gtts import gTTS



# Load environment variables
load_dotenv()

# Define the prompt
prompt = """
    You are an expert; you will answer the question not more than 70 words .
"""

# to fetch AI response using Google Gemini API
def fetch_ai_response(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text


## function to audio to text
model = WhisperModel("base.en", device="cpu") 
def transcribe_audio(audio_file):
    segments, info = model.transcribe(audio_file, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription



def text_to_audio(text, audio_file):
    tts = gTTS(text=text, lang='en', slow= False)
    
    temp_audio_file = audio_file
    tts.save(temp_audio_file)
    
    pygame.mixer.init()
    pygame.mixer.music.load(temp_audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(1)

    pygame.mixer.music.stop()
    pygame.mixer.quit()

    os.remove(temp_audio_file)


def main():
    st.sidebar.title("API Key Configuration")
    api_key = st.sidebar.text_input("Enter your Google Gemini API Key", type="password")

    st.title("AI Voice Assistant")
    st.write("This is a simple voice assistant that uses Google Gemini's API to respond to user inputs.")

    if api_key:
        #genai.configure(api_key=api_key)
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        recorded_audio = audio_recorder()

        if recorded_audio:
            audio_file = "audio.mp3"
            with open(audio_file, "wb") as f:
                f.write(recorded_audio)

            transcript = transcribe_audio(audio_file)
            st.write("Transcript: ", transcript)

            ai_response = fetch_ai_response(transcript, prompt)
            st.write("AI Response: ", ai_response)

            response_audio_file = "audio_response.mp3"
            text_to_audio(ai_response, response_audio_file)


if __name__ == "__main__":
    main()
