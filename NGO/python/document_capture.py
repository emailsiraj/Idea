import streamlit as st
import cv2
import numpy as np
from PIL import Image

import cv2
import numpy as np

def order_points(pts):
    """
    Robust ordering of 4 points: returns [tl, tr, br, bl] as floats.
    Accepts pts as shape (4,2) or list of points.
    """
    pts = np.array(pts, dtype="float32")
    if pts.shape != (4, 2):
        raise ValueError("order_points expects 4 points with shape (4,2)")

    # Sum and diff method
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).reshape(-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    rect = np.vstack([tl, tr, br, bl]).astype("float32")
    return rect

def four_point_transform(image, pts, dest_size=None, interp=cv2.INTER_LINEAR):
    """
    pts: 4 points (any order). Output is perspective-warped top-down image.
    dest_size: optional (width, height). If None, computed from source points.
    """
    rect = order_points(pts)  # ensures tl,tr,br,bl
    (tl, tr, br, bl) = rect

    # compute width of new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))

    # compute height of new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))

    # If user forced dest size, use it (useful for fixed-form templates)
    if dest_size is not None:
        maxWidth, maxHeight = int(dest_size[0]), int(dest_size[1])

    # Safety floor
    maxWidth = max(2, maxWidth)
    maxHeight = max(2, maxHeight)

    # Destination coordinates: top-left, top-right, bottom-right, bottom-left
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute perspective transform matrix and warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=interp, borderMode=cv2.BORDER_REPLICATE)

    return warped

def detect_document_contour(image, debug=False, min_area_frac=0.2):
    """
    Detect the largest 4-point contour likely corresponding to a document.
    Returns points as shape (4,2) float32 or None.
    debug -> returns (pts, debug_image)
    """
    orig = image.copy()
    # Work with a resized copy for speed & stability (but keep ratio)
    h, w = image.shape[:2]
    target_width = 1000
    scale = 1.0
    if w > target_width:
        scale = target_width / float(w)
        image_small = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        image_small = image.copy()

    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Dilate to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    cnts_info = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Compatible extraction for different OpenCV versions
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
    if not cnts:
        if debug:
            return None, image_small
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    img_area = image_small.shape[0] * image_small.shape[1]
    biggest = None

    for c in cnts:
        area = cv2.contourArea(c)
        if area < (min_area_frac * img_area):
            # skip small contours
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            # Found candidate
            pts = approx.reshape(4, 2)
            # Scale back to original image coordinates
            pts = pts.astype("float32") / scale
            biggest = pts
            break

    if debug:
        dbg = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
        if biggest is not None:
            # draw scaled back contour on resized image for visualization
            scaled_back = (biggest * scale).astype(int)
            cv2.polylines(dbg, [scaled_back.reshape(-1,2)], True, (0,255,0), 2)
        return biggest, dbg

    return biggest

# -----------------------
# Streamlit UI starts here
# -----------------------
st.set_page_config(page_title="Document Scanner Demo")

st.title("📄 Document Capture & Correction Demo")
st.write("Upload or capture a document; the app will detect the outline and straighten it.")

upload_type = st.radio("Choose input method:", ["Upload Image", "Camera Capture"])

uploaded = None
if upload_type == "Upload Image":
    uploaded = st.file_uploader("Upload a document photo", type=["jpg", "jpeg", "png"])
else:
    uploaded = st.camera_input("Capture document")

if uploaded:
    img = np.array(Image.open(uploaded))
    image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Detect document outline
    pts = detect_document_contour(image_bgr)

    if pts is None:
        st.error("❌ Could not detect a clean document outline. Please retake the photo.")
        st.image(img, caption="Original Image")
        st.stop()

    # Draw contour overlay
    overlay = image_bgr.copy()
    cv2.polylines(overlay, [pts.astype(int)], True, (0, 255, 0), 3)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Warp document
    warped = four_point_transform(image_bgr, pts)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    # Display result
    col1, col2 = st.columns(2)

    col1.header("📸 Detected Document")
    col1.image(overlay, caption="Detected Outline")

    col2.header("✨ Corrected (Top-Down View)")
    col2.image(warped, caption="Perspective Corrected Output")

    st.success("Document detected and corrected successfully!")
