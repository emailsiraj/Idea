import streamlit as st
import cv2
import numpy as np
from PIL import Image

from utils.img_utils import *
from utils.ocr_utils import *
from utils.s3_utils import *

st.set_page_config(page_title="Recommendation Letter Upload")

st.title("Recommendation Letter Upload")

is_accessible = True
for key, value in st.session_state.status.items():
    if key != 'verified_rec':
        is_accessible = is_accessible and value
    else:
        break
        
if not is_accessible:
    st.error("You cannot access this page yet.")
    st.stop()
    
def reset_state():
    st.session_state.status['verified_rec'] = False
    
if st.session_state.status['verified_rec']:
    st.success("✔ Recommendation letter is uploaded successfully")
    st.button(label="Reupload Image",on_click=reset_state)
else:
    st.write("Upload or capture the picture of the recommendation letter.")

    upload_type = st.radio("Choose input method:", ["Upload Image", "Camera Capture"])

    uploaded = None
    if upload_type == "Upload Image":
        uploaded = st.file_uploader("Upload document", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.camera_input("Capture document")

    if uploaded:
        st.session_state.status['verified_rec'] = True
        
        img = np.array(Image.open(uploaded))
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        url = upload_cv2_image_to_s3(image_bgr, st.session_state.folder_name,'uploaded_recommendation.jpeg')
        rec_json = {
            "Recommendation Letter": {
                "Verified": True,
                "Image URL": url
            }
        }
        
        st.session_state.info_json = {**st.session_state.info_json, **rec_json}
        upload_json(f'{st.session_state.folder_name}/details.json',st.session_state.info_json)
        
        st.success("Image uploaded! Proceed to the next step.")
        st.page_link('pages/4_Marksheet_Verification.py',label="Continue")     