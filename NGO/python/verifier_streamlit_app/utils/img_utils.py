import numpy as np
import cv2

def _is_valid_document_quad(pts, img_shape):
    """
    Basic geometric sanity checks for document quadrilateral.
    """
    area = cv2.contourArea(pts)
    img_area = img_shape[0] * img_shape[1]

    if area < 0.1 * img_area:
        return False

    rect = cv2.minAreaRect(pts)
    w, h = rect[1]

    if w == 0 or h == 0:
        return False

    aspect = max(w, h) / min(w, h)
    if aspect > 5.0:
        return False

    return True

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

def detect_by_border(image, scale=1.0, debug=False):
    """
    Detect the largest 4-point contour likely corresponding to a document.
    Returns points as shape (4,2) float32 or None.
    debug -> returns (pts, debug_image)
    """
    min_area_frac = 0.2
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

def detect_by_region(
    image,
    debug=False,
    min_area_frac=0.1
):
    """
    Detect the main document region and return 4 corner points.

    Returns:
        pts: np.ndarray of shape (4, 2) float32 in original image coords
        If debug=True -> (pts, debug_image)
    """

    orig = image.copy()
    h, w = image.shape[:2]

    # Resize for stability
    target_width = 1000
    scale = 1.0
    if w > target_width:
        scale = target_width / float(w)
        image_small = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        image_small = image.copy()

    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)

    # Illumination normalization
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Adaptive threshold (document as region, not edge)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    # Morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # External contours only
    cnts, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not cnts:
        return (None, image_small) if debug else None

    img_area = image_small.shape[0] * image_small.shape[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    best_quad = None

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area_frac * img_area:
            continue

        # Remove concavities caused by text
        hull = cv2.convexHull(c)

        # Always produces a 4-point rectangle
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)  # shape (4,2)
        box = np.array(box, dtype=np.float32)

        if not _is_valid_document_quad(box, image_small.shape):
            continue

        # Scale back to original resolution
        best_quad = box / scale
        break

    if debug:
        dbg = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
        if best_quad is not None:
            scaled = (best_quad * scale).astype(int)
            cv2.polylines(dbg, [scaled], True, (0, 255, 0), 3)
        return best_quad, dbg

    return best_quad

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

def crop_quad(image, quad):
    x_coords = quad[:, 0]
    y_coords = quad[:, 1]

    x1, x2 = int(x_coords.min()), int(x_coords.max())
    y1, y2 = int(y_coords.min()), int(y_coords.max())
    
    x1, x2 = max(x1,0), min(x2,image.shape[1])
    y1, y2 = max(y1,0), min(y2, image.shape[0])

    return (x1,y1,x2,y2), image[y1:y2, x1:x2]

def detect_document(image):
    pts = detect_by_border(image)
    if pts is not None:
        print("Detected using Border")
        return pts
    print("Detected using Region")
    return detect_by_region(image)