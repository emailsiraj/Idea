import streamlit as st
import face_recognition
from PIL import Image
import numpy as np

st.set_page_config(page_title="Face Verification")

st.title("Photo Upload and Verification")

# Aadhaar must be verified before this page is accessible
if "verified_aadhar" not in st.session_state or st.session_state.verified_aadhar == False:
    st.error("❌ You cannot access this page yet.")
    st.stop()

# Maintain face verification state
if "verified_face" not in st.session_state:
    st.session_state.verified_face = False

st.write("Please upload a clear photo containing your face.")

# If face already verified previously → show success & skip processing
if st.session_state.verified_face:
    st.success("✔ Face already verified earlier")
    st.write("You may proceed to the next step.")
else:
    uploaded_face = st.file_uploader("Upload your picture", type=["jpg", "jpeg", "png"])

    if uploaded_face:
        img = np.array(Image.open(uploaded_face))
        
        try:
            faces = face_recognition.face_locations(img)

            if len(faces) > 0:
                st.session_state.verified_face = True  # Store state permanently
                st.success("✔ Face detected successfully!")

            else:
                st.error("❌ No face detected. Please upload a proper selfie.")

        except Exception as e:
            st.error("❌ Something went wrong processing the image.")
            st.write(e)
