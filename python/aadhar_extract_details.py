from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections

# repo details
repo_config = dict(
    repo_id = "arnabdhar/YOLOv8-nano-aadhar-card",
    filename = "model.pt",
    local_dir = "./models"
)

# load model
model = YOLO(hf_hub_download(**repo_config))

# get id to label mapping
id2label = model.names
print(id2label)

# Perform Inference
image_url = "https://i.pinimg.com/originals/08/6d/82/086d820550f34066764f4047ddc263ca.jpg"

detections = Detections.from_ultralytics(model.predict(image_url)[0])

print(detections)
