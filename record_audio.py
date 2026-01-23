import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import os
from datetime import datetime

def record_audio(filename="recording.wav", duration=10, fs=16000, channels=1, chunk_size=1024):
    """Record audio with specified parameters.
    
    Args:
        filename (str): Output filename
        duration (int): Recording duration in seconds
        fs (int): Sample rate (default: 16000 for Whisper)
        channels (int): Number of channels (1 for mono)
        chunk_size (int): Audio chunk size
    """
    print("Recording started...")
    try:
        # Create recordings directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Record audio
        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=channels,
            dtype='float32'
        )
        sd.wait()
        
        # Save the recording
        wav.write(filename, fs, recording)
        print("Recording saved to", filename)
        return True
    except Exception as e:
        print(f"Error during recording: {str(e)}")
        return False

class AudioRecorder:
    def __init__(self, sample_rate=44100, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_data = []
        
    def callback(self, indata, frames, time, status):
        """This is called for each audio block."""
        if status:
            print(f"Status: {status}")
        if self.recording:
            self.audio_data.append(indata.copy())
    
    def start_recording(self):
        """Start recording audio."""
        self.recording = True
        self.audio_data = []
        print("Recording started...")
        
        # Start the input stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.callback
        )
        self.stream.start()
    
    def stop_recording(self):
        """Stop recording audio and save to file."""
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            
            # Combine all recorded blocks
            if self.audio_data:
                audio_data = np.concatenate(self.audio_data, axis=0)
                
                # Create recordings directory if it doesn't exist
                if not os.path.exists('recordings'):
                    os.makedirs('recordings')
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recordings/recording_{timestamp}.wav"
                
                # Save the recording
                wav.write(filename, self.sample_rate, audio_data)
                print(f"Recording saved to {filename}")
                return filename
            else:
                print("No audio data recorded")
                return None

def main():
    # Example of using the simple recording function
    print("Using simple recording function:")
    record_audio("simple_recording.wav", duration=5)
    
    # Example of using the class-based recorder
    print("\nUsing class-based recorder:")
    recorder = AudioRecorder()
    
    try:
        # Start recording
        recorder.start_recording()
        
        # Record for 5 seconds
        print("Recording for 5 seconds...")
        time.sleep(5)
        
        # Stop recording and save
        filename = recorder.stop_recording()
        if filename:
            print(f"Recording completed and saved to {filename}")
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
        recorder.stop_recording()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        recorder.stop_recording()

if __name__ == "__main__":
    main() 