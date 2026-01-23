import whisper
import torch
import numpy as np
import soundfile as sf
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model (you can change the model size based on your needs)
# Options: tiny, base, small, medium, large
MODEL_SIZE = "base"
model = None

def load_model():
    """Load the Whisper model."""
    global model
    if model is None:
        logger.info(f"Loading Whisper {MODEL_SIZE} model...")
        model = whisper.load_model(MODEL_SIZE)
        logger.info("Model loaded successfully")
    return model

def transcribe_audio(audio_file):
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_file (str): Path to the audio file
        
    Returns:
        str: Transcribed text
    """
    try:
        logger.info(f"Starting transcription of {audio_file}")
        
        # Check if file exists
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Check file size
        file_size = os.path.getsize(audio_file)
        logger.info(f"Audio file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("Audio file is empty")
        
        # Load the model if not already loaded
        model = load_model()
        
        # Load audio file
        logger.info("Loading audio file...")
        audio, sample_rate = sf.read(audio_file)
        logger.info(f"Audio loaded: shape={audio.shape}, sample_rate={sample_rate}")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            logger.info("Converting stereo to mono")
            audio = audio.mean(axis=1)
        
        # Convert to float32
        audio = audio.astype(np.float32)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Transcribe
        logger.info("Starting transcription...")
        result = model.transcribe(audio)
        logger.info("Transcription completed")
        
        text = result["text"].strip()
        logger.info(f"Transcribed text: {text}")
        
        return text
        
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return "" 