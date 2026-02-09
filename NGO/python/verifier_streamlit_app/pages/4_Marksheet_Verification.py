import pytesseract
import numpy as np
import cv2
import re
import streamlit as st

from PIL import Image
from rapidfuzz import fuzz
from utils.ocr_utils import *
from utils.s3_utils import *


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

CORPUS = {
    # high confidence
    "marksheet": 5,
    "marks": 4,
    "grade": 4,
    "examination": 4,
    "subject": 3,
    "gpa": 4,
    "degree": 4,

    # institutions
    "school": 3,
    "college": 3,
    "university": 3,
    "institute": 3,
    "government": 3,

    # subjects and related words
    "mathematics": 2,
    "physics": 2,
    "chemistry": 2,
    "english": 2,
    "science": 2,
    "botany": 2,
    "zoology": 2,
    "tamil": 2,
    "candidate": 2,
    "result": 3,
    "theory": 3,
    "laboratory": 2,

    # generic academic
    "roll": 2,
    "register": 2,
    "registration": 2,
    "enrollment": 2,
    "semester": 2,
    "result": 2,
    "pass": 2,
    "fail": 2,
    "signature": 1,
    "headmaster": 1,
    
}
def length_compatible(a, b):
    return min(len(a), len(b)) / max(len(a), len(b)) >= 0.8


def extract_words(img):
    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
        config="--oem 1 --psm 6"
    )
    words = [
        w.lower()
        for w in data["text"]
        if w.strip() != "" and len(w) > 2
    ]
    return words

def normalize(word):
    word = word.lower()
    word = re.sub(r'[^a-z]', '', word)
    return word

def score_words(words, corpus, threshold=90):
    score = 0
    matched = set()
    matched_dict = dict()

    for w in words:
        for key, weight in corpus.items():
            if length_compatible(w, key) and fuzz.partial_ratio(w, key) >= threshold:
                score += weight
                matched.add(key)
                matched_dict[w] = key
    return score, matched, matched_dict

def is_marksheet(img):
    words = extract_words(enhance_for_ocr(img))
    words = [normalize(w) for w in words]
    
    # print(words)
    normalized_words = [normalize(word) for word in words if normalize(word) != '']
    unique_words = list(set(normalized_words))

    score, matched, matched_dict = score_words(unique_words, CORPUS)
    print(matched_dict)

    return score >= 20

#Streamlit part

st.set_page_config(page_title="Marksheet Verification")

st.title("Marksheet Verification")

is_accessible = True
for key, value in st.session_state.status.items():
    if key != 'verified_marksheet':
        is_accessible = is_accessible and value
    else:
        break
        
if not is_accessible:
    st.error("You cannot access this page yet.")
    st.stop()
    
def reset_state():
    st.session_state.status['verified_marksheet'] = False
    
if st.session_state.status['verified_marksheet']:
    st.success("✔ Marksheet is already uploaded.")
    st.button(label="Reupload Image",on_click=reset_state)
else:
    st.write("Upload or capture the picture of the marksheet.")

    upload_type = st.radio("Choose input method:", ["Upload Image", "Camera Capture"])

    uploaded = None
    if upload_type == "Upload Image":
        uploaded = st.file_uploader("Upload document", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.camera_input("Capture document")

    if uploaded:
        img = np.array(Image.open(uploaded))
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Checking if the marksheet is valid"):
            result = is_marksheet(image_bgr)
        
        if result:
            st.session_state.status['verified_marksheet'] = True
            
            url = upload_cv2_image_to_s3(image_bgr, st.session_state.folder_name,'uploaded_marksheet.jpeg')
            marksheet_json = {
                "Marksheet": {
                    "Verified": True,
                    "Image URL": url
                }
            }
            
            st.session_state.info_json = {**st.session_state.info_json, **marksheet_json}
            upload_json(f'{st.session_state.folder_name}/details.json',st.session_state.info_json)
            
            st.success("Marksheet image uploaded! Proceed to the next step.")
            st.page_link('pages/5_MSET_Application_Verification.py',label="Continue")
        else:
            st.error("The uploaded image is not a marksheet. Please check and upload again")
            st.image(img, caption="Original Image")     