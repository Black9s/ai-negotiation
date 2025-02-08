import cv2
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze emotions
    analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

    # Extract dominant emotion
    dominant_emotion = analysis[0]["dominant_emotion"]

    # Display result
    cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
