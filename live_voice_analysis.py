import pyaudio
import numpy as np
import librosa

# Initialize PyAudio for live audio streaming
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("🎙️ Listening for live audio...")

while True:
    data = stream.read(CHUNK)
    np_data = np.frombuffer(data, dtype=np.int16)

    # Extract pitch (higher pitch may indicate stress)
    pitch = librosa.yin(np_data.astype(float), fmin=50, fmax=300, sr=RATE)

    # Extract energy (higher energy may indicate confidence)
    energy = np.sum(np_data**2) / len(np_data)

    print(f"Pitch: {np.mean(pitch):.2f} Hz, Energy: {energy:.2f}")

    # Stop on user command
    if input("Press Enter to stop or type 'q' to quit: ") == 'q':
        break

stream.stop_stream()
stream.close()
audio.terminate()
