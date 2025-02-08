import cv2
from deepface import DeepFace
import librosa
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pyaudio

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize PyAudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Facial Emotion Detection
    analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
    dominant_emotion = analysis[0]["dominant_emotion"]

    # Voice Stress Detection
    data = stream.read(CHUNK)
    np_data = np.frombuffer(data, dtype=np.int16)
    pitch = librosa.yin(np_data.astype(float), fmin=50, fmax=300, sr=RATE)
    energy = np.sum(np_data**2) / len(np_data)

    # Display results
    cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(f"Voice Tone: Pitch={np.mean(pitch):.2f} Hz, Energy={energy:.2f}")

    cv2.imshow("Live AI Negotiation Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
audio.terminate()
