from ultralytics import YOLO
import cv2
from huggingface_hub import hf_hub_download
from supervision import Detections
import pytesseract

# Path to tesseract executable 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#Download model
repo_config = dict(
    repo_id = "arnabdhar/YOLOv8-nano-aadhar-card",
    filename = "model.pt",
    local_dir = "./models"
)

# Load model
model = YOLO("models/model.pt")
id2reg = model.names

# Path to the Aadhar Card image
image_path = "aadhar_test_img_3.jpg"
img = cv2.imread(image_path)

# Detect regions of interest from the image using trained model
detections = Detections.from_ultralytics(model.predict(img)[0])
xyxy_list = detections.xyxy
class_list = detections.class_id

# Helper function for extracting image regions
def extract_region(img, xyxy):
    x1, y1, x2, y2 = [int(coordinate) for coordinate in xyxy]
    return img[y1:y2, x1:x2]
    
# Read and process image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert to grayscale
img = cv2.medianBlur(img, 3)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # Adaptive Thresholding
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu Thresholding

custom_config = r'--oem 3 --psm 6'
# Show image regions
for i, xyxy in zip(class_list, xyxy_list):
    print(f'{id2reg[i]}: \n {xyxy}')
    preprocessed_region = extract_region(img, xyxy)
    # cv2.imshow(f'{id2reg[i]}', preprocessed_region)
    print(f'{id2reg[i]} contains {pytesseract.image_to_string(preprocessed_region, config = custom_config).strip()}')

# cv2.waitKey(0)
# cv2.destroyAllWindows()
