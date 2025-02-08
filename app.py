import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("🧠 AI-Powered Feeling Recognition for Negotiations")

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Text Analysis Section
st.subheader("📢 Text-Based Emotion & Deception Analysis")
text_input = st.text_area("Enter negotiation text:")
if st.button("Analyze Text"):
    result = analyzer.polarity_scores(text_input)
    st.write("📊 AI Analysis:", result)

# Live Video Analysis Section
st.subheader("🎥 Real-Time Facial Emotion Analysis")
if st.button("Start Live Video"):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze emotions
        analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        dominant_emotion = analysis[0]["dominant_emotion"]

        # Display the detected emotion
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
