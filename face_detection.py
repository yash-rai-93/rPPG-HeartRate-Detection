import cv2
import mediapipe as mp
from google.colab.patches import cv2_imshow

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_face_and_landmarks(image):
    """Detect face and extract landmarks using MediaPipe FaceMesh."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(img_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    return results.multi_face_landmarks[0]

def draw_landmarks(image, face_landmarks):
    """Draw facial landmarks on the image."""
    for landmark in face_landmarks.landmark:
        h, w, _ = image.shape
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

def process_video(video_path):
    """Process a video file frame by frame in Google Colab."""
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        face_landmarks = detect_face_and_landmarks(frame)
        if face_landmarks:
            draw_landmarks(frame, face_landmarks)
        
        cv2_imshow(frame)  # Display frame in Google Colab
    
    cap.release()

if _name_ == "_main_":
    video_path = "path/to/your/video.mp4"  # Replace with your video path
    process_video(video_path)
