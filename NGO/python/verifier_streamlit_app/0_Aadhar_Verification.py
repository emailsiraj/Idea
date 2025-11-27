import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Aadhaar Verification", page_icon="🪪")

# Load model only once for performance
if "model" not in st.session_state:
    st.session_state.model = YOLO("../models/aadhar_classifier.pt")

# Persist state
if "verified_aadhar" not in st.session_state:
    st.session_state.verified_aadhar = False

st.title("AADHAR Upload and Verification")

# If Aadhaar was already verified before and user navigates back → DO NOT RESET
if st.session_state.verified_aadhar:
    st.success("✔ Aadhaar already verified earlier")
    st.page_link("pages/1_Face_Verification.py", label="➡ Continue", icon="▶")
else:
    st.write("Upload an image to validate if it is an Aadhaar card.")
    uploaded = st.file_uploader("Upload Aadhaar Card Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        results = st.session_state.model(image)
        pred = results[0].probs
        cls_id = pred.top1
        conf = pred.top1conf
        class_name = st.session_state.model.names[cls_id]

        if class_name.lower() == "aadhar":
            st.session_state.verified_aadhar = True  # Persist!
            st.success(f"✔ Aadhaar detected ({conf:.2f})")
            
            # Continue button becomes ACTIVE only now
            st.page_link("pages/1_Face_Verification.py", label="➡ Continue", icon="▶")

        else:
            st.error("❌ This is NOT an Aadhaar card.")
            st.button("Continue", disabled=True)
