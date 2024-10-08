import os
import numpy as np
import librosa
import keras
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog
import threading

# Load the trained model
model = keras.models.load_model("sound_classification_model.keras")

# Define parameters
n_mfcc = 40  # Number of MFCC features to extract
max_pad_len = 174  # Maximum padding length for MFCC feature array
fs = 44100  # Sample rate for microphone input
duration = 2  # Duration to listen to microphone (in seconds)

# Function to extract MFCC features from an audio file or audio data
def extract_features(audio, sample_rate):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfccs
    except Exception as e:
        print(f"Error encountered while extracting features: {e}")
        return None

# Function to classify sound from audio data
def classify_sound_from_data(audio_data):
    features = extract_features(audio_data, fs)
    if features is not None:
        features = np.expand_dims(features, axis=-1)  # Add a channel dimension
        features = np.expand_dims(features, axis=0)   # Add a batch dimension (1 sample)
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)
        return predicted_class[0]
    return None

# Function to classify sound from a file
def classify_sound(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        return classify_sound_from_data(audio)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}. Error: {e}")
        return None

# Dictionary to map class IDs to class names
class_names = {
    0: "Air Conditioner",
    1: "Car Horn",
    2: "Children Playing",
    3: "Dog Bark",
    4: "Drilling",
    5: "Engine Idling",
    6: "Gun Shot",
    7: "Jackhammer",
    8: "Siren",
    9: "Street Music"
}

# Function to capture microphone input and classify
def listen_and_classify_mic():
    print("Listening for audio...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    audio_data = np.squeeze(audio_data)  # Flatten the audio data array
    predicted_class = classify_sound_from_data(audio_data)
    if predicted_class is not None:
        class_name = class_names.get(predicted_class, "Unknown")
        result_var.set(f"Predicted sound: {class_name}")
    else:
        result_var.set("Could not classify the sound from the microphone.")

# Function to select and classify a file
def classify_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class = classify_sound(file_path)
        if predicted_class is not None:
            class_name = class_names.get(predicted_class, "Unknown")
            result_var.set(f"Predicted sound: {class_name}")
        else:
            result_var.set("Could not classify the sound from the file.")

# Tkinter GUI
root = tk.Tk()
root.title("Sound Classifier")
root.geometry("400x300")

result_var = tk.StringVar()
result_var.set("Press 'Classify from File' or 'Listen to Mic'.")

label = tk.Label(root, textvariable=result_var, font=("Helvetica", 14))
label.pack(pady=20)

# Button to classify from a file
file_button = tk.Button(root, text="Classify from File", command=classify_file, width=20, height=2)
file_button.pack(pady=10)

# Button to classify from the microphone
mic_button = tk.Button(root, text="Listen to Mic", command=lambda: threading.Thread(target=listen_and_classify_mic).start(), width=20, height=2)
mic_button.pack(pady=10)

# Start the GUI loop
root.mainloop()
