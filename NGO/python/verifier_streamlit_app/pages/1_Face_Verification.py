import streamlit as st
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Function to detect grayscale images
def is_color_image(image, threshold=10):
    if len(image.shape) < 3 or image.shape[2] != 3:
        return False  # True grayscale image

    R, G, B = cv2.split(image)

    diff_rg = np.mean(np.abs(R - G))
    diff_rb = np.mean(np.abs(R - B))
    diff_gb = np.mean(np.abs(G - B))

    color_score = (diff_rg + diff_rb + diff_gb) / 3

    return color_score > threshold

# Function to detect less saturated images
def has_enough_saturation(image, sat_threshold=25):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    mean_sat = np.mean(saturation)

    return mean_sat > sat_threshold

#Function to detect negative images
def is_negative_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Negative images usually have high brightness but abnormal contrast
    if mean_intensity > 170 and std_intensity < 60:
        return True

    return False

# Function for validating images
def validate_selfie_face(image):
    print(f"Image shape: {image.shape}")
    h, w, _ = image.shape
    img_area = h * w
    
    if not is_color_image(image):
        return False, "Uploaded image is not a color image", None
    
    if not has_enough_saturation(image):
        return False, "Image has very low color information", None
    
    if is_negative_image(image):
        return False, "Uploaded image is not a color image", None

    rgb = image
    # cv2.imwrite("uploaded_img.jpg",rgb)
    results = face_mesh.process(rgb)

    # 1. Exactly ONE face
    if not results.multi_face_landmarks:
        return False, "No face detected", None

    if len(results.multi_face_landmarks) != 1:
        return False, "Multiple faces detected", None

    landmarks = results.multi_face_landmarks[0]

    # Convert normalized landmarks to pixel coordinates
    xs = [int(pt.x * w) for pt in landmarks.landmark]
    ys = [int(pt.y * h) for pt in landmarks.landmark]

    min_x, max_x = max(0, min(xs)), min(w, max(xs))
    min_y, max_y = max(0, min(ys)), min(w, max(ys))

    face_w = max_x - min_x
    face_h = max_y - min_y
    face_area = face_w * face_h

    # 2. Face size check (ONLY FACE condition)
    face_ratio = face_area / img_area
    print(f"(Face / Image) area ratio: {face_ratio}")
    if face_ratio < 0.15:   # VERY important threshold
        return False, "Face too small / background too dominant", None

    # 3. Centered face check
    face_cx = (min_x + max_x) / 2
    face_cy = (min_y + max_y) / 2
    img_cx, img_cy = w / 2, h / 2

    dx = abs(face_cx - img_cx) / w
    dy = abs(face_cy - img_cy) / h

    if dx > 0.15 or dy > 0.15:
        return False, "Face not centered", None

    # 4. Aspect ratio check (reject full body shots)
    aspect_ratio = face_h / face_w
    print(f"Face aspect ratio: {aspect_ratio}")
    if aspect_ratio < 1.05 or aspect_ratio > 1.7:
        return False, "Invalid aspect ratio", None

    # 5. FRONTAL FACE CHECK (Pose Validation)
    LEFT_EYE = 33
    RIGHT_EYE = 263
    NOSE = 1

    lx, ly = xs[LEFT_EYE], ys[LEFT_EYE]
    rx, ry = xs[RIGHT_EYE], ys[RIGHT_EYE]
    nx, ny = xs[NOSE], ys[NOSE]

    # Eye level alignment
    if abs(ly - ry) > face_h * 0.05:
        return False, "Head tilted", None

    # Nose centered between eyes
    if abs(nx - (lx + rx) / 2) > face_w * 0.07:
        return False, "Face not frontal", None

    return True, "Valid face image", (min_x, min_y, max_x, max_y)


#Streamlit part
st.set_page_config(page_title="Face Verification")

st.title("Photo Upload and Verification")

# Aadhaar must be verified before this page is accessible
if "verified_aadhar" not in st.session_state or st.session_state.verified_aadhar == False:
    st.error(" You cannot access this page yet.")
    st.stop()

# Maintain face verification state
if "verified_face" not in st.session_state:
    st.session_state.verified_face = False

st.write("Please upload a clear photo containing your face.")

def reset_session_state():
    st.session_state.verified_face = False
    
# If face already verified previously → show success & skip processing
if st.session_state.verified_face:
    st.success("✔ Face already verified earlier")
    st.write("You may proceed to the next step.")
    st.button("Reupload the image", on_click=reset_session_state)
else:
    uploaded_face = st.file_uploader("Upload your picture", type=["jpg", "jpeg", "png"])

    if uploaded_face:
        img = np.array(Image.open(uploaded_face).convert("RGB"))
        
        try:
            is_valid, message, bbox = validate_selfie_face(img)
            
            col1, col2 = st.columns(2)

            #Column 1 - Original Image
            with col1:
                st.subheader("Uploaded Image")
                st.image(img, width="stretch")

            #Column 2 - Bounding Box Image
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                boxed_img = img.copy()
                cv2.rectangle(boxed_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

                with col2:
                    st.subheader("Detected Face")
                    st.image(boxed_img, width="stretch")
            
            if not is_valid:
                st.error(message)
            else:
                st.session_state.verified_face = True
                st.success("Photo uploaded successfully.")
                
        except Exception as e:
            st.error(" Something went wrong processing the image.")
            st.write(e)
