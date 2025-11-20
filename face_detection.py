import cv2
import mediapipe as mp

# Path to input image
image_path = r"OIP.jpg"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

mp_face = mp.solutions.face_detection

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
    # Convert image to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face
    result = face_detector.process(rgb)

    # Output
    if result.detections:
        print("Face detected")
    else:
        print("No face detected")
