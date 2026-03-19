import streamlit as st
import cv2
import numpy as np
import pytesseract
import os

from PIL import Image
from utils.img_utils import *
from utils.ocr_utils import *
from utils.s3_utils import *

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

REGION_CONFIG = {
    "title": {
        "bbox": (47, 135, 143, 635),   # ref image coords
        "task": "presence"
    },
    "top": {
        "bbox": (4, 3, 635, 149),
        "task": "extract",
        "subregions": {
            "certificate_no": {
                "bbox": (127, 121, 378, 143),
                "task": "text"
            }
        }
    }
}

def extract_sift(image, n_features=0):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des

def compute_homography(ref_patch, doc_gray, ratio=0.75, ransac_thresh=5.0):
    kp1, desc1 = extract_sift(ref_patch)
    kp2, desc2 = extract_sift(doc_gray)

    if desc1 is None or desc2 is None:
        return None, None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]

    if len(good) < 20:
        return None, None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    return H, len(good)

def project_bbox(bbox, H):
    x1, y1, x2, y2 = bbox
    pts = np.float32([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ]).reshape(-1, 1, 2)

    return cv2.perspectiveTransform(pts, H).reshape(4, 2)


def extract_roi(image, quad):
    return crop_quad(image, quad)[1]

def process_region(name, config, ref_gray, doc_gray, doc_color):
    x1, y1, x2, y2 = config["bbox"]
    ref_patch = ref_gray[y1:y2, x1:x2]

    H, match_count = compute_homography(ref_patch, doc_gray)
    if H is None:
        return {"present": False}

    result = {"present": True, "matches": match_count}

    if config["task"] == "presence":
        return result

    region_quad = project_bbox(config["bbox"], H)
    region_roi = extract_roi(doc_color, region_quad)

    if "subregions" not in config:
        return result

    result["subregions"] = {}

    for sub_name, sub_cfg in config["subregions"].items():
        sub_quad = project_bbox(sub_cfg["bbox"], H)
        sub_roi = extract_roi(doc_color, sub_quad)

        if sub_cfg["task"] == "text":
            # ocr_img = enhance_for_ocr(sub_roi)
            text = pytesseract.image_to_string(
                sub_roi, config='--oem 1 --psm 8'
            ).strip()
            result["subregions"][sub_name] = text

    return result

def validate_content(warped):
    ref = cv2.imread("pages/resized_document.jpg")
    warped = cv2.resize(warped, (640, 800))

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    doc_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    results = {}

    for name, cfg in REGION_CONFIG.items():
        results[name] = process_region(
            name, cfg, ref_gray, doc_gray, warped
        )

    return results
       
# -----------------------
# Streamlit UI starts here
# -----------------------

st.set_page_config(page_title="Income Certificate Verification")

st.title("Income Certificate Verification")

is_accessible = True
for key, value in st.session_state.status.items():
    if key != 'verified_income':
        is_accessible = is_accessible and value
    else:
        break
        
if not is_accessible:
    st.error("You cannot access this page yet.")
    st.stop()
    
# st.session_state.verified_income = True
def reset_session_state():
    st.session_state.status['verified_income'] = False

if st.session_state.status['verified_income']:
    st.success("✔ Income Certificate is verified successfully")
    st.button("Reupload Image",on_click=reset_session_state)
else:
    st.write("Upload or capture the picture of the Income Certificate.")

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
            
            with st.spinner("Checking whether the certificate is valid..."):
                result = validate_content(warped)

                is_valid = True
                for key, value in result.items():
                    is_valid = is_valid and value['present']
                    if not is_valid:
                        break                        
                
            if not is_valid:
                st.error(f'Uploaded certificate is invalid. Please upload a valid certificate.')
            else:
                st.session_state.status['verified_income'] = True
                
                url = upload_cv2_image_to_s3(image_bgr, st.session_state.folder_name,'uploaded_income_certificate.jpeg')
                
                income_json = {
                    "Income Certificate": {
                        "Verified": True,
                        "Certificate Number": result['top']['subregions']['certificate_no'].split(':')[1].strip(),
                        "Image URL": url
                    }
                }
                
                st.session_state.info_json = {**st.session_state.info_json, **income_json}
                upload_json(f'{st.session_state.folder_name}/details.json',st.session_state.info_json)
                
                st.success(f'Certificate validated successfully! Proceed to the next stage.')
                st.page_link('pages/3_Recommendation_Upload.py',label="Continue")
        else:
            st.error("❌ Could not detect a clean document outline. Please retake the photo.")
            st.image(img, caption="Original Image")
            # st.stop()
