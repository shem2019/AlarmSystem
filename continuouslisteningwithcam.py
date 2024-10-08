import os
import numpy as np
import librosa
import keras
import sounddevice as sd
import soundfile as sf
import threading
import time
import cv2  # For camera control
from datetime import datetime

# Load the trained model
model = keras.models.load_model("sound_classification_model.keras")

# Define parameters
n_mfcc = 40  # Number of MFCC features to extract
max_pad_len = 174  # Maximum padding length for MFCC feature array
fs = 44100  # Sample rate for microphone input
chunk_duration = 6  # Duration for each audio chunk (in seconds)
recording_dir = "./recordings"  # Directory to store audio clips

# Create directory if it doesn't exist
if not os.path.exists(recording_dir):
    os.makedirs(recording_dir)

# Signal flag for whether recording is in progress
recording_in_progress = False

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


# Function to extract MFCC features from audio
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)  # Load with the native sample rate
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # Pad or truncate MFCCs to ensure uniform shape
        if mfccs.shape[1] < max_pad_len:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, max_pad_len - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        # Expand dimensions to match the input shape expected by the model
        return np.expand_dims(np.expand_dims(mfccs, axis=-1), axis=0)
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


# Function to open the camera for 5 seconds
def open_camera_for_5_seconds():
    print("Opening camera for 5 seconds...")
    cap = cv2.VideoCapture(0)  # Open the default camera (0 is usually the default camera)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()  # Capture frame-by-frame
        if ret:
            cv2.imshow('Camera', frame)  # Display the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
                break
        else:
            print("Error: Failed to capture image.")
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print("Closed camera after 5 seconds.")


# Function to classify sound from an audio file
def classify_sound(file_path):
    print(f"Classifying sound from {file_path}...")
    features = extract_features(file_path)
    if features is not None:
        try:
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][predicted_class] * 100
            print(f"Predicted sound: {class_names[predicted_class]} ({confidence:.2f}%)")

            # Check if the predicted sound is a car horn or a siren
            if class_names[predicted_class] in ["Car Horn", "Siren"]:
                # Open the camera for 5 seconds
                open_camera_for_5_seconds()

        except Exception as e:
            print(f"Error during prediction: {e}")


# Function to record 6-second audio clips
def record_audio():
    global recording_in_progress

    while True:
        recording_in_progress = True
        print("Recording audio...")
        audio_data = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait for the recording to complete
        recording_in_progress = False

        # Save the audio clip as a .wav file
        clip_filename = os.path.join(recording_dir, f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        sf.write(clip_filename, audio_data, fs)
        print(f"Saved audio clip {clip_filename}")

        # Pass the file name to the classification function
        classify_sound(clip_filename)

        time.sleep(1)  # Small pause between recordings


# Function to control the recording and prediction
def start_recording_and_prediction():
    record_thread = threading.Thread(target=record_audio, daemon=True)
    record_thread.start()


# Start the recording and classification process
if __name__ == "__main__":
    start_recording_and_prediction()

    # Keep the main thread alive
    while True:
        time.sleep(10)
