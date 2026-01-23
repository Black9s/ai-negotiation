import streamlit as st
import cv2
import deepface
from deepface import DeepFace
import numpy as np
from collections import Counter
import time
import pandas as pd
from record_audio import record_audio
from whisper_transcribe import transcribe_audio, load_model
import os
import threading
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import json
import base64
import queue
import re
from textblob import TextBlob
import logging
import sounddevice as sd
import soundfile as sf
import sys
import traceback

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Streamlit app
st.set_page_config(page_title="AI Negotiation Emotion Tracker", layout="wide")
st.title("🧠 AI-Powered Negotiation Emotion & Deception Analyzer")

# Debug mode
DEBUG = True

def log_debug(message, error=None):
    """Enhanced debug logging function."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    debug_message = f"[{timestamp}] {message}"
    
    # Log to console
    logger.debug(debug_message)
    
    # If there's an error, log the full traceback
    if error:
        error_traceback = traceback.format_exc()
        logger.error(f"Error details:\n{error_traceback}")
        debug_message += f"\nError: {str(error)}\nTraceback:\n{error_traceback}"
    
    # Log to Streamlit debug panel
    if DEBUG:
        with st.sidebar.expander("Debug Log", expanded=True):
            st.text(debug_message)

# Function to check available audio devices
def check_audio_devices():
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if not input_devices:
            log_debug("No audio input devices found", "No input devices available")
            return False
            
        log_debug(f"Found {len(input_devices)} input devices:")
        for i, device in enumerate(input_devices):
            device_info = f"Device {i}: {device['name']} (Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']})"
            log_debug(device_info)
        
        return True
    except Exception as e:
        log_debug("Error checking audio devices", e)
        return False

# Function to analyze emotion from frame
def analyze_emotion(frame):
    try:
        log_debug("Starting emotion analysis...")
        
        # Convert frame to RGB for DeepFace
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        log_debug("Frame converted to RGB")
        
        # Try to analyze emotion using DeepFace
        try:
            result = DeepFace.analyze(frame_rgb, actions=["emotion"], enforce_detection=False)
            log_debug(f"DeepFace analysis result: {result}")
            
            # Handle different result formats
            if isinstance(result, list):
                result = result[0]
            elif isinstance(result, str):
                log_debug("Unexpected string result from DeepFace", result)
                return None
                
            # Extract emotion data
            if isinstance(result, dict) and 'emotion' in result:
                emotions = result['emotion']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                
                log_debug(f"Emotion detected: {dominant_emotion[0]} with confidence {dominant_emotion[1]:.2f}")
                
                return {
                    'emotion': dominant_emotion[0],
                    'confidence': float(dominant_emotion[1]),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
            else:
                log_debug("Unexpected result format from DeepFace", result)
                return None
                
        except Exception as e:
            log_debug("Error in DeepFace analysis", e)
            return None
            
    except Exception as e:
        log_debug("Error in emotion analysis", e)
        return None

# Function to process audio chunk
def process_audio_chunk(audio_file):
    try:
        log_debug(f"Processing audio file: {audio_file}")
        
        # Check if file exists and has content
        if not os.path.exists(audio_file):
            log_debug(f"Audio file not found: {audio_file}", "File does not exist")
            return
            
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            log_debug("Audio file is empty", "File size is 0 bytes")
            return
            
        log_debug(f"Audio file size: {file_size} bytes")
        
        # Load Whisper model
        try:
            model = load_model()
            log_debug("Whisper model loaded successfully")
        except Exception as e:
            log_debug("Error loading Whisper model", e)
            return
        
        # Transcribe audio
        log_debug("Starting transcription...")
        try:
            transcription = transcribe_audio(audio_file)
            log_debug(f"Transcription result: {transcription}")
        except Exception as e:
            log_debug("Error during transcription", e)
            return
        
        if transcription:
            st.session_state.transcription_text += transcription + " "
            
            # Analyze sentiment
            try:
                blob = TextBlob(transcription)
                sentiment = blob.sentiment.polarity
                log_debug(f"Sentiment analysis: {sentiment}")
            except Exception as e:
                log_debug("Error in sentiment analysis", e)
                sentiment = 0
            
            # Add to deception analysis
            st.session_state.deception_analysis.append({
                'text': transcription,
                'sentiment': sentiment,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        else:
            log_debug("No speech detected in audio")
            
    except Exception as e:
        log_debug("Error processing audio", e)

# Function to check available audio devices
def check_microphone_permission():
    try:
        # Try to access the microphone
        sd.check_input_settings()
        return True
    except Exception as e:
        st.error("⚠️ Microphone access is required but not granted. Please allow microphone access in your browser settings.")
        st.info("To enable microphone access:")
        st.markdown("""
        1. Click the lock/info icon in your browser's address bar
        2. Find 'Microphone' in the permissions list
        3. Change it from 'Block' to 'Allow'
        4. Refresh the page
        """)
        return False

# Create necessary directories
for directory in ['recordings', 'transcriptions', 'sessions', 'exports']:
    if not os.path.exists(directory):
        os.makedirs(directory)
        log_debug(f"Created directory: {directory}")

# Session management
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_debug(f"Created new session: {st.session_state.session_id}")

if 'emotion_log' not in st.session_state:
    st.session_state.emotion_log = []
    log_debug("Initialized emotion log")

if 'recording_active' not in st.session_state:
    st.session_state.recording_active = False
    log_debug("Recording state initialized")

if 'transcription_text' not in st.session_state:
    st.session_state.transcription_text = ""
    log_debug("Transcription text initialized")

if 'deception_analysis' not in st.session_state:
    st.session_state.deception_analysis = []
    log_debug("Deception analysis initialized")

if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
    log_debug("Audio queue initialized")

if 'mic_permission' not in st.session_state:
    st.session_state.mic_permission = check_microphone_permission()

# Deception detection patterns
DECEPTION_PATTERNS = {
    'hedging': r'\b(um|uh|like|sort of|kind of|basically|actually|literally)\b',
    'qualifiers': r'\b(maybe|perhaps|possibly|probably|supposedly|allegedly)\b',
    'distancing': r'\b(they|them|those|that|it)\b',
    'negative_emotion': r'\b(angry|upset|disappointed|frustrated|worried|anxious)\b',
    'overly_formal': r'\b(indeed|furthermore|moreover|consequently|therefore)\b',
    'repetition': r'\b(\w+)\s+\1\b',
    'inconsistency': r'\b(but|however|although|yet|nevertheless)\b'
}

def analyze_deception(text):
    """Analyze text for potential deception indicators."""
    analysis = {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'indicators': {},
        'score': 0
    }
    
    # Check for deception patterns
    for pattern_name, pattern in DECEPTION_PATTERNS.items():
        matches = len(re.findall(pattern, text.lower()))
        analysis['indicators'][pattern_name] = matches
    
    # Calculate deception score (0-100)
    total_indicators = sum(analysis['indicators'].values())
    analysis['score'] = min(100, total_indicators * 10)  # Scale score
    
    return analysis

# Sidebar for session controls and analysis
with st.sidebar:
    st.header("Session Controls")
    new_session = st.button("Start New Session")
    if new_session:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.emotion_log = []
        st.session_state.transcription_text = ""
        st.session_state.deception_analysis = []
        st.success("New session started!")

    st.header("Session Info")
    st.write(f"Session ID: {st.session_state.session_id}")
    
    # Emotion statistics
    if st.session_state.emotion_log:
        df = pd.DataFrame(st.session_state.emotion_log)
        
        # Emotion distribution
        st.subheader("Emotion Distribution")
        emotion_counts = df['emotion'].value_counts()
        fig_emotion = px.pie(values=emotion_counts.values, 
                           names=emotion_counts.index,
                           title="Emotion Distribution")
        st.plotly_chart(fig_emotion, use_container_width=True)
        
        # Emotion trends
        st.subheader("Emotion Trends")
        # Create a line plot for each emotion
        fig_trends = px.line(df, 
                           x='second',
                           y='confidence',
                           color='emotion',
                           title="Emotion Trends Over Time")
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Deception analysis
        if st.session_state.deception_analysis:
            st.subheader("Deception Analysis")
            deception_df = pd.DataFrame(st.session_state.deception_analysis)
            if 'timestamp' in deception_df.columns:
                deception_df['timestamp'] = pd.to_datetime(deception_df['timestamp'], format='%H:%M:%S')
                fig_deception = px.line(deception_df, 
                                      x='timestamp', 
                                      y='score',
                                      title="Deception Score Over Time")
                st.plotly_chart(fig_deception, use_container_width=True)

# Emotion label colors
colors = {
    "angry": (255, 0, 0),
    "disgust": (0, 255, 0),
    "fear": (0, 0, 255),
    "happy": (255, 255, 0),
    "sad": (0, 255, 255),
    "surprise": (255, 0, 255),
    "neutral": (200, 200, 200)
}

html_colors = {
    "angry": "red",
    "disgust": "green",
    "fear": "purple",
    "happy": "gold",
    "sad": "lightblue",
    "surprise": "orange",
    "neutral": "grey"
}

def record_audio_thread():
    """Record audio with detailed logging."""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        audio_file = os.path.join('recordings', f"negotiation_{st.session_state.session_id}_{timestamp}.wav")
        log_debug(f"Starting audio recording to: {audio_file}")
        
        # Check if microphone is available
        if not check_audio_devices():
            raise Exception("No audio input devices found")
        log_debug(f"Found audio devices")
        
        # Record audio
        record_audio(audio_file, duration=10)
        log_debug(f"Audio recording completed: {audio_file}")
        
        # Verify the audio file was created and has content
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file was not created: {audio_file}")
        
        file_size = os.path.getsize(audio_file)
        log_debug(f"Recorded audio file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("Recorded audio file is empty")
        
        st.session_state.audio_queue.put(audio_file)
        log_debug("Added audio file to processing queue")
        
    except Exception as e:
        error_msg = f"Error during recording: {str(e)}"
        log_debug(error_msg)
        st.error(error_msg)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Real-time Emotion Analysis")
    
    # Add audio device check
    if not check_audio_devices():
        st.error("No audio input devices found. Please connect a microphone and refresh the page.")
        st.stop()
    
    # Video feed and controls
    col_video1, col_video2 = st.columns(2)
    with col_video1:
        run = st.button("Start Analysis")
    with col_video2:
        stop = st.button("Stop Analysis")

    if not st.session_state.mic_permission:
        st.warning("⚠️ Please enable microphone access to use all features of this application.")
        st.stop()

    emotion_caption_placeholder = st.empty()
    frame_window = st.image([])
    
    # Emotion chart
    chart_placeholder = st.empty()
    
    # Process video frames
    if run:
        log_debug("Starting analysis...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open camera")
            log_debug("Camera initialization failed")
            st.stop()
        
        start_time = time.time()
        
        # Start audio recording in a separate thread
        if not st.session_state.recording_active:
            st.session_state.recording_active = True
            log_debug("Starting audio recording thread")
            audio_thread = threading.Thread(target=record_audio_thread)
            audio_thread.start()

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not working.")
                log_debug("Failed to read frame from camera")
                break

            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)

            # Analyze emotion with debugging
            result = analyze_emotion(frame)
            if result:
                dominant_emotion = result['emotion']
                confidence = result['confidence']
                
                # Update emotion caption
                color = html_colors.get(dominant_emotion.lower(), "white")
                emotion_caption_placeholder.markdown(
                    f"<h3 style='text-align: center; color: {color};'>{dominant_emotion.capitalize()}: {confidence:.1f}%</h3>",
                    unsafe_allow_html=True
                )

                # Log time and emotion
                timestamp = round(time.time() - start_time)
                emotion_entry = {
                    "second": timestamp,
                    "emotion": dominant_emotion,
                    "confidence": confidence,
                    "session_id": st.session_state.session_id
                }
                st.session_state.emotion_log.append(emotion_entry)
                log_debug(f"Added emotion entry: {emotion_entry}")
                
            else:
                emotion_caption_placeholder.empty()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)

            # Update emotion chart
            if st.session_state.emotion_log:
                df = pd.DataFrame(st.session_state.emotion_log)
                chart_data = df.groupby(["second", "emotion"]).size().unstack(fill_value=0)
                chart_placeholder.line_chart(chart_data)
            
            # Process any completed audio chunks
            try:
                while not st.session_state.audio_queue.empty():
                    audio_file = st.session_state.audio_queue.get_nowait()
                    process_audio_chunk(audio_file)
            except queue.Empty:
                pass

        cap.release()
        st.session_state.recording_active = False
        log_debug("Analysis stopped")
        st.success("Analysis stopped.")

with col2:
    st.header("Real-time Transcription & Analysis")
    
    # Initialize session state
    if 'transcription' not in st.session_state:
        st.session_state.transcription = ""
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    
    # Voice recording controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 Start Recording"):
            try:
                # Record audio
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                audio_file = os.path.join('recordings', f"voice_{timestamp}.wav")
                
                with st.spinner("Recording... Speak clearly and at a normal pace..."):
                    record_audio(audio_file, duration=10, fs=16000, channels=1)
                
                # Transcribe
                with st.spinner("Transcribing..."):
                    transcription = transcribe_audio(audio_file)
                    if transcription:
                        st.session_state.transcription = transcription
                        
                        # Analyze sentiment
                        blob = TextBlob(transcription)
                        sentiment = blob.sentiment.polarity
                        
                        # Store analysis
                        st.session_state.analysis = {
                            'sentiment': sentiment,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        }
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("🛑 Stop Recording"):
            st.session_state.transcription = ""
            st.session_state.analysis = None
    
    # Display transcription
    if st.session_state.transcription:
        st.markdown("### 📝 Transcription")
        st.text_area("", st.session_state.transcription, height=150)
        
        # Display analysis
        if st.session_state.analysis:
            st.markdown("### 📊 Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment = st.session_state.analysis['sentiment']
                sentiment_color = "green" if sentiment > 0 else "red" if sentiment < 0 else "gray"
                st.metric("Sentiment", f"{sentiment:.2f}")
            
            with col2:
                st.metric("Time", st.session_state.analysis['timestamp'])
            
            # Sentiment gauge
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background-color: {sentiment_color}; opacity: 0.2; border-radius: 10px; margin-top: 20px;'>
                <h4 style='margin-bottom: 10px;'>Sentiment Analysis</h4>
                <p style='font-size: 18px; font-weight: bold;'>Score: {sentiment:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Click 'Start Recording' to begin voice recording")

# Export options
st.header("Export Analysis")
col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    if st.button("Export Session Data"):
        if st.session_state.emotion_log:
            df = pd.DataFrame(st.session_state.emotion_log)
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="emotion_data.csv">Download Emotion Data</a>'
            st.markdown(href, unsafe_allow_html=True)

with col_exp2:
    if st.button("Export Transcription"):
        if st.session_state.transcription_text:
            txt = st.session_state.transcription_text
            b64 = base64.b64encode(txt.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="transcription.txt">Download Transcription</a>'
            st.markdown(href, unsafe_allow_html=True)

with col_exp3:
    if st.button("Export Full Analysis"):
        if st.session_state.emotion_log and st.session_state.transcription_text:
            analysis_data = {
                'session_id': st.session_state.session_id,
                'emotion_data': st.session_state.emotion_log,
                'transcription': st.session_state.transcription_text,
                'deception_analysis': st.session_state.deception_analysis
            }
            json_str = json.dumps(analysis_data, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="full_analysis.json">Download Full Analysis</a>'
            st.markdown(href, unsafe_allow_html=True)

# Save session data
if st.session_state.emotion_log:
    session_file = os.path.join('sessions', f"session_{st.session_state.session_id}.csv")
    pd.DataFrame(st.session_state.emotion_log).to_csv(session_file, index=False)
    log_debug(f"Session data saved to {session_file}")
    st.sidebar.success(f"Session data saved to {session_file}")
