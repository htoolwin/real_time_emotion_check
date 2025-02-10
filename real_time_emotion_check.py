import cv2
import mediapipe as mp
from deepface import DeepFace

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect Faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w_box, h_box = (int(bboxC.xmin * w), int(bboxC.ymin * h), 
                                  int(bboxC.width * w), int(bboxC.height * h))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)

            # Extract face region for emotion analysis
            face_region = frame[y:y+h_box, x:x+w_box]

            # Run emotion detection
            try:
                analysis = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']

                # Display emotion label
                cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except Exception as e:
                print("Emotion Detection Error:", str(e))

    # Show Video
    cv2.imshow("Real-Time Facial Emotion Tracking", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
