import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
from supervision import Detections
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def enhance_for_ocr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced

def extract_aadhar_content(aadhar_image):
    extracted_info = dict()
    id2reg = st.session_state.extraction_model.names
    
    detections = Detections.from_ultralytics(st.session_state.extraction_model.predict(aadhar_image)[0])
    xyxy_list = detections.xyxy
    class_list = detections.class_id
    
    custom_config = r'--oem 3 --psm 6'

    for i, xyxy in zip(class_list, xyxy_list):
        preprocessed_region = enhance_for_ocr(extract_region(aadhar_image, xyxy))
        # _, preprocessed_region = cv2.threshold(preprocessed_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        extracted_text = pytesseract.image_to_string(preprocessed_region, config = custom_config).strip()
        print(f'{id2reg[i]} contains {extracted_text}')
        extracted_info[f'{id2reg[i]}'] = extracted_text
        
    return extracted_info

def extract_region(img, xyxy):
    x1, y1, x2, y2 = [int(coordinate) for coordinate in xyxy]
    return img[y1:y2, x1:x2]

def field_validation(name, aadhar_num, dob, gender):
    validated_info = dict()
    
    # Date of Birth
    if len(dob.split('/')) != 3:
        dob = None
    validated_info['Date of Birth'] = dob
    
    # Aadhar Number
    aadhar_num_parts = aadhar_number.split(' ')
    if len(aadhar_num_parts) == 3:
        for part in aadhar_num_parts:
            if len(part) == 4:
                continue
            else:
                aadhar_number = None
                break
    else:
        aadhar_number = None
    validated_info['Aadhar Number'] = aadhar_num
    
    # Name
    for char in name:
        if not (char.isalpha() or char.isspace()):
            name = None
            break
    if name is not None:
        if len(name) <= 3:
            name = None
    validated_info['Name'] = name
    
    #Gender
    if gender.lower() not in ('male', 'female'):
        gender = None
    validated_info['Gender'] = gender
    
def reset_state():
    st.session.verified_aadhar = False

st.set_page_config(page_title="Aadhaar Verification", page_icon="🪪")

# Load model only once for performance
if "detection_model" not in st.session_state:
    st.session_state.detection_model = YOLO("models/aadhar_classifier.pt")
    
if "extraction_model" not in st.session_state:
    st.session_state.extraction_model = YOLO("models/model.pt")

# Persist state
if "verified_aadhar" not in st.session_state:
    st.session_state.verified_aadhar = False

st.title("Aadhar Upload and Verification")

# If Aadhaar was already verified before and user navigates back → DO NOT RESET
if st.session_state.verified_aadhar:
    st.success("Aadhar already verified earlier")
    st.page_link("pages/1_Face_Verification.py", label="Continue", icon="▶")
    st.button("Reupload Aadhar Image", on_click=reset_state)
else:
    st.write("Upload the Aadhar card image here.")
    uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        results = st.session_state.detection_model(image)
        pred = results[0].probs
        cls_id = pred.top1
        conf = pred.top1conf
        class_name = st.session_state.detection_model.names[cls_id]

        if class_name.lower() == "aadhar":
            st.session_state.verified_aadhar = True  # Persist!
            st.success(f"Aadhaar detected with Confidence: ({conf:.2f})")
            
            with st.spinner("Reading Aadhar Content..."):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                extracted_info = extract_aadhar_content(image)
                
            st.success("Details extracted successfully")
            # Get values
            name = extracted_info.get('NAME',None)
            aadhar_number = extracted_info.get('AADHAR_NUMBER',None)
            gender = extracted_info.get('GENDER', None)
            dob = extracted_info.get('DATE_OF_BIRTH', None)
            
            st.write(f"Name: {name}\nAadhar: {aadhar_number}\ngender: {gender}\nDOB: {dob}")
            
            # Continue button becomes ACTIVE only now
            st.page_link("pages/1_Face_Verification.py", label="Continue", icon="▶")

        else:
            st.error("This is not an Aadhar card. Please upload the picture of a valid Aadhar card.")
            st.button("Continue", disabled=True)
