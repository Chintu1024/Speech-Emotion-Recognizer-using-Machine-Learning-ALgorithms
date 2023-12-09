import os
import time
import soundfile
import streamlit as st
import speech_recognition as sr
import wave
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import librosa
import keras

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()
st.title("Speech Emotion Recognizer")

# Initialize session state
if "audio_file_name" not in st.session_state:
    st.session_state.audio_file_name = None

# Reading Microphone as source
# listening to the speech and storing it in the audio_text variable
start_recording_button = st.button("Start Recording")


if start_recording_button:
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        st.write("Listening... Speak now.")
        # Generate a unique file name with a timestamp
        timestamp = time.strftime("%Y%m%d%H%M%S")
        audio_file_name = f"recorded_audio11_{timestamp}.wav"
        st.session_state.audio_file_name = audio_file_name

        audio_text = r.listen(source, timeout=5)

        st.write("Time over...Thanks")

        try:
            # using google speech recognition
            recognized_text = r.recognize_google(audio_text)
            st.write("Text: " + recognized_text)

            # Save the audio data in wave format with the unique file name
            with wave.open(audio_file_name, "wb") as audio_file:
                audio_file.setnchannels(1)  # mono audio
                audio_file.setsampwidth(2)  # 16-bit audio
                audio_file.setframerate(44100)  # sample rate (you can adjust this)
                audio_file.writeframes(audio_text.frame_data)

            st.write(f"Audio saved as '{audio_file_name}'")

        except sr.UnknownValueError:
            st.write("Sorry, I did not get that")
        except sr.RequestError as e:
            st.write("Could not request results; {0}".format(e))

# Load the pre-trained emotion recognition model
loaded_model = keras.models.load_model("venv/cnnnnnnn.h5")
#with open("venv/best_model_CNN.h5", "rb") as model_file:
 #   loaded_model = pickle.load(model_file)

display_wave = st.button("Display Wave")
if display_wave and st.session_state.audio_file_name is not None and os.path.exists(st.session_state.audio_file_name):
    # Plot the recorded audio as a waveform
    audio = AudioSegment.from_wav(st.session_state.audio_file_name)
    samples = np.array(audio.get_array_of_samples())
    plt.figure(figsize=(10, 4))
    plt.plot(samples)
    plt.title("Recorded Audio Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

def extract_feature(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
    return np.hstack((mfccs, chroma, mel))


predict_emotion = st.button("Get Emotion")
if predict_emotion and st.session_state.audio_file_name is not None and os.path.exists(st.session_state.audio_file_name):
    # Load the audio data and preprocess it
    filename = st.session_state.audio_file_name  # Use st.session_state.audio_file_name
    features = extract_feature(filename)
    features = StandardScaler().fit_transform(features.reshape(-1, 1).T)

    # Make predictions using the loaded model
    predicted_emotion = loaded_model.predict(features)
    result = np.argmax(predicted_emotion)
    # Display the predicted emotion
    st.write("Predicted Emotion:")
    if(result == 0):
        st.write("Angry")
        st.image("venv/emotions/angry.jpg")
    elif(result == 1):
        st.write("Calm")
        st.image("venv/emotions/calm.jpg")
    elif (result == 2):
        st.write("Disgust")
        st.image("venv/emotions/disgust.jpg")
    elif (result == 3):
        st.write("fearful")
        st.image("venv/emotions/fearful.jpg")
    elif (result == 4):
        st.write("Happy")
        st.image("venv/emotions/happy.jpg")
    elif (result == 5):
        st.write("Neutral")
        st.image("venv/emotions/neutral.jpg")
    elif (result == 6):
        st.write("Sad")
        st.image("venv/emotions/sad.jpg")
    else:
        st.write("Surprised")
        st.image("venv/emotions/surprised.jpg")

    # Delete the audio file after using it for emotion prediction
    if st.session_state.audio_file_name is not None and os.path.exists(st.session_state.audio_file_name):
        os.remove(st.session_state.audio_file_name)
        st.session_state.audio_file_name = None  # Clear the audio file name from session state
        predicted_emotion = None

