import whisper
import os
from datetime import datetime

def transcribe_audio(audio_path):
    """
    Simple function to transcribe audio using Whisper.
    Returns just the transcription text.
    """
    # Convert to absolute path and verify file exists
    abs_path = os.path.abspath(audio_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Audio file not found at: {abs_path}")
    
    print(f"Loading Whisper model...")
    model = whisper.load_model("base")  # You can also use "small", "medium", "large"
    print(f"Transcribing file: {abs_path}")
    result = model.transcribe(abs_path)
    return result["text"]

class AudioTranscriber:
    def __init__(self, model_size="base"):
        """
        Initialize the transcriber with a Whisper model.
        model_size options: "tiny", "base", "small", "medium", "large"
        """
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully!")
        
    def transcribe_audio(self, audio_file):
        """
        Transcribe an audio file using Whisper.
        Returns the transcription text and saves it to a file.
        """
        # Convert to absolute path and verify file exists
        abs_path = os.path.abspath(audio_file)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Audio file not found at: {abs_path}")
            
        print(f"Transcribing {abs_path}...")
        
        # Perform transcription
        result = self.model.transcribe(abs_path)
        
        # Get the transcription text
        transcription = result["text"]
        
        # Create transcriptions directory if it doesn't exist
        if not os.path.exists('transcriptions'):
            os.makedirs('transcriptions')
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join('transcriptions', f"transcript_{timestamp}.txt")
        
        # Save transcription to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcription)
            
        print(f"Transcription saved to {output_file}")
        return transcription, output_file

def main():
    # Example of using the simple transcription function
    print("Using simple transcription function:")
    try:
        # Check if there are any recordings to transcribe
        if not os.path.exists('recordings'):
            print("No recordings directory found. Please record some audio first.")
            return
            
        # Get the most recent recording
        recordings = os.listdir('recordings')
        if not recordings:
            print("No recordings found in the recordings directory.")
            return
            
        # Sort recordings by modification time (newest first)
        latest_recording = max(
            [os.path.join('recordings', f) for f in recordings],
            key=os.path.getmtime
        )
        
        print(f"Found latest recording: {latest_recording}")
        
        # Transcribe using simple function
        print(f"\nTranscribing {latest_recording}...")
        transcription = transcribe_audio(latest_recording)
        print("\nSimple Transcription:")
        print("-" * 50)
        print(transcription)
        print("-" * 50)
        
        # Example of using the class-based transcriber
        print("\nUsing class-based transcriber:")
        transcriber = AudioTranscriber(model_size="base")
        transcription, output_file = transcriber.transcribe_audio(latest_recording)
        
        print("\nClass-based Transcription:")
        print("-" * 50)
        print(transcription)
        print("-" * 50)
        print(f"\nFull transcription saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir('.')}")

if __name__ == "__main__":
    main() 