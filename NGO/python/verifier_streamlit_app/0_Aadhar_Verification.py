import streamlit as st
import cv2
import pytesseract
import numpy as np
import json
import os

from huggingface_hub import hf_hub_download
from supervision import Detections
from ultralytics import YOLO
from PIL import Image
from utils.ocr_utils import *
from utils.s3_utils import *

# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
def download_models():
    os.makedirs("models", exist_ok=True)

    model_path = hf_hub_download(
        repo_id="Chaitanya1729/mset-models",
        filename="model.pt",
        local_dir="models",
        token=os.getenv('HF_TOKEN')
    )

    classifier_path = hf_hub_download(
        repo_id="Chaitanya1729/mset-models",
        filename="aadhar_classifier.pt",
        local_dir="models",
        token=os.getenv('HF_TOKEN')
    )

    return model_path, classifier_path


def extract_aadhar_content(aadhar_image):
    extracted_info = dict()
    id2reg = st.session_state.extraction_model.names
    
    detections = Detections.from_ultralytics(st.session_state.extraction_model.predict(aadhar_image)[0])
    xyxy_list = detections.xyxy
    class_list = detections.class_id
    
    custom_config = r'--oem 3 --psm 6'

    for i, xyxy in zip(class_list, xyxy_list):
        preprocessed_region = enhance_for_ocr(extract_region(aadhar_image, xyxy))
        extracted_text = pytesseract.image_to_string(preprocessed_region, config = custom_config).strip()
        print(f'{id2reg[i]} contains {extracted_text}')
        extracted_info[f'{id2reg[i]}'] = extracted_text
        
    return extracted_info

def extract_region(img, xyxy):
    x1, y1, x2, y2 = [int(coordinate) for coordinate in xyxy]
    return img[y1:y2, x1:x2]
    
def create_aadhar_json(name, aadhar_num, dob, gender):
    return {
        "Aadhar" : {
            "Verified": True,
            "Name": name,
            "Gender": gender,
            "Date of Birth": dob,
            "Aadhar Number":aadhar_num
        }
    }

def reset_state():
    st.session_state.status['verified_aadhar'] = False

model_path, classifier_path = download_models()

st.set_page_config(page_title="Aadhaar Verification", page_icon="🪪")

# Load model only once for performance
if "detection_model" not in st.session_state:
    st.session_state.detection_model = YOLO(classifier_path)
    
if "extraction_model" not in st.session_state:
    st.session_state.extraction_model = YOLO(model_path)
    
# Persist state
if 'status' not in st.session_state:
    st.session_state.status = {
        'verified_aadhar': False,
        'verified_face': False,
        'verified_income': False,
        'verified_rec': False,
        'verified_marksheet': False,
        'verified_mset': False
    }

# if "verified_aadhar" not in st.session_state:
#     st.session_state.verified_aadhar = False
if 'clicked_verify' not in st.session_state:
    st.session_state.clicked_verify = False

st.title("Aadhar Upload and Verification")

aadhar = st.text_input(
    "Enter 12-digit Aadhaar number (no spaces)",
    max_chars=12
)

if st.button("Verify Aadhaar"):
    st.session_state.clicked_verify = True
    if not aadhar.isdigit() or len(aadhar) != 12:
        st.error("Please enter a valid 12-digit Aadhaar number.")
    else:
        if folder_exists(aadhar):
            st.session_state.folder_name = aadhar
            details_json = download_file_bytes(f'{st.session_state.folder_name}/details.json')
            st.session_state.info_json = details_json
            if 'Aadhar' in details_json and details_json['Aadhar']['Verified']:
                st.session_state.status['verified_aadhar'] = True
            if 'Photograph' in details_json and details_json['Photograph']['Verified']:
                st.session_state.status['verified_face'] = True
            if 'Income Certificate' in details_json and details_json['Income Certificate']['Verified']:
                st.session_state.status['verified_income'] = True
            if 'Recommendation Letter' in details_json and details_json['Recommendation Letter']['Verified']:
                st.session_state.status['verified_rec'] = True
            if 'Marksheet' in details_json and details_json['Marksheet']['Verified']:
                st.session_state.status['verified_marksheet'] = True
            if 'MSET Application' in details_json and details_json['MSET Application']['Verified']:
                st.session_state.status['verified_mset'] = True
        else:
            st.session_state.folder_name = aadhar
            create_s3_folder(str(st.session_state.folder_name))
            info_json = {
                'Aadhar':
                    {
                        'Verified': False
                    }
            }
            upload_json(f'{st.session_state.folder_name}/details.json',info_json)

# If Aadhaar was already verified before and user navigates back → DO NOT RESET
if st.session_state.clicked_verify and st.session_state.status['verified_aadhar']:
    st.success("Aadhar already verified earlier")
    st.page_link("pages/1_Face_Verification.py", label="Continue", icon="▶")
    st.button("Reupload Aadhar Image", on_click=reset_state)
elif st.session_state.clicked_verify and not st.session_state.status['verified_aadhar']:
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
            st.success(f"Aadhaar detected with Confidence: ({conf:.2f})")
            
            with st.spinner("Reading Aadhar Content..."):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                extracted_info = extract_aadhar_content(image)
                
            st.success("Details extracted")
            
            name = extracted_info.get('NAME',None)
            aadhar_number = extracted_info.get('AADHAR_NUMBER',None)
            gender = extracted_info.get('GENDER', None)
            dob = extracted_info.get('DATE_OF_BIRTH', None)
            
            aadhar_json = create_aadhar_json(name, aadhar_number, dob, gender)
            
            st.write(aadhar_json)
            
            img = np.array(Image.open(uploaded))
            image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            url = upload_cv2_image_to_s3(image_bgr, st.session_state.folder_name,'uploaded_aadhar.jpeg')
            aadhar_json['Aadhar']['Image URL'] = url
            st.session_state.info_json = aadhar_json
            upload_json(f'{st.session_state.folder_name}/details.json',st.session_state.info_json)
            
            st.session_state.status['verified_aadhar'] = True
            
            st.page_link("pages/1_Face_Verification.py", label="Continue", icon="▶")

        else:
            st.error("This is not an Aadhar card. Please upload the picture of a valid Aadhar card.")
            st.button("Continue", disabled=True)
