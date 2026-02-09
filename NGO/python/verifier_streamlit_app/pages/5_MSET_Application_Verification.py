import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract

from utils.img_utils import *
from utils.ocr_utils import *
from utils.s3_utils import *

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def is_field_filled(field_img):
    gray = cv2.cvtColor(field_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )
    
    # st.image(binary, width="content")

    ink_pixels = cv2.countNonZero(binary)
    ink_ratio = ink_pixels / binary.size
    
    # st.write(ink_ratio)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    valid_components = sum(
        1 for s in stats[1:]
        if s[cv2.CC_STAT_AREA] > 18
    )
    
    # st.write(valid_components)

    return ink_ratio > 0.005 and valid_components >= 2


def check_content(warped):
    resized = cv2.resize(warped, (640,800))
    bounding_boxes = {
"name":(244, 146, 572, 162),
"dob":(139, 174, 236, 193),
"age":(300, 178, 340, 192),
"father_name":(150, 233, 432, 253),
"mother_name":(158, 262, 472, 282),
"address":(236, 297, 556, 349),
"course":(161, 370, 324, 395),
"college":(225, 397, 619, 425),
"father_occupation":(236, 428, 591, 451),
"family_member_count":(235, 455, 402, 482),
"family_monthly_income":(205, 487, 319, 509),
"student_signature":(423, 649, 583, 664),
"guardian_signature":(406, 670, 555, 693)
}

# Run OCR
    detected_text_dict = {}
    # custom_config = r'--oem 1 --psm 7'
    for key, bbox in bounding_boxes.items():
        x1, y1, x2, y2 = bbox
        region = resized[y1:y2, x1:x2]
        # st.image(region, width="content")
        # cv2.imshow(f"Image: {key}", region)
        # region = enhance_for_ocr(region)
        # detected_text = pytesseract.image_to_string(region).strip()
        # if detected_text == "":
        if not is_field_filled(region):
            detected_text_dict[key] = None
        else:
            detected_text_dict[key] = "True"#detected_text
    
    return detected_text_dict
# -----------------------
# Streamlit UI starts here
# -----------------------
st.set_page_config(page_title="MSET Application Verification")

st.title("MSET Application Verification")

is_accessible = True
for key, value in st.session_state.status.items():
    if key != 'verified_mset':
        is_accessible = is_accessible and value
    else:
        break
        
if not is_accessible:
    st.error("You cannot access this page yet.")
    st.stop()
    
def reset_state():
    st.session_state.status['verified_mset'] = False
    
if st.session_state.status['verified_mset']:
    st.success("✔ MSET Application is verified successfully")
    st.button(label="Reupload Image",on_click=reset_state)
else:
    st.write("Upload or capture the picture of the first page of MSET application form.")

    upload_type = st.radio("Choose input method:", ["Upload Image", "Camera Capture"])

    uploaded = None
    if upload_type == "Upload Image":
        uploaded = st.file_uploader("Upload document", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.camera_input("Capture document")

    if uploaded:
        img = np.array(Image.open(uploaded))
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Detect document outline
        pts = detect_document(image_bgr)
        
        if pts is not None: 
            # Draw contour overlay
            overlay = image_bgr.copy()
            cv2.polylines(overlay, [pts.astype(int)], True, (0, 255, 0), 3)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # Warp document
            warped = four_point_transform(image_bgr, pts)
            warped = enhance_for_ocr(warped)
            
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

            # Display result
            col1, col2 = st.columns(2)

            col1.header("Detected Document")
            col1.image(overlay, caption="Detected Outline")

            col2.header("Main document content")
            col2.image(warped, caption="Perspective Corrected Output")

            st.success("Document detected!")
            
            with st.spinner("Checking whether the required fields are populated..."):
                missing_value_list = []
                detected_text_dict = check_content(warped)
                for key, value in detected_text_dict.items():
                    if value is None:
                        missing_value_list.append(key)
                
            if len(missing_value_list) != 0:
                st.error(f'Values are missing for the following fields: {missing_value_list}')
            else:
                st.session_state.status['verified_mset'] = True
                
                url = upload_cv2_image_to_s3(cv2.cvtColor(warped, cv2.COLOR_RGB2BGR), st.session_state.folder_name,'uploaded_mset_application.jpeg')
                mset_json = {
                    "MSET Application": {
                        "Verified": True,
                        "Image URL": url
                    }
                }
                
                st.session_state.info_json = {**st.session_state.info_json, **mset_json}
                upload_json(f'{st.session_state.folder_name}/details.json',st.session_state.info_json)
                
                st.success(f'Application validated successfully!')
        else:
            st.error("❌ Could not detect a clean document outline. Please retake the photo.")
            st.image(img, caption="Original Image")
            # st.stop()                    