from deepface import DeepFace

image_path = r"img1.JPEG"

try:
    DeepFace.extract_faces(img_path=image_path)
    print("Real face detected")
except:
    print("No real face detected")
