from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch

from ultralytics import YOLO

models_to_benchmark =[
    "hustvl/yolos-tiny",
    "PekingU/rtdetr_r18vd_coco_o365"
]

ultralytics_model = [
    "yolo26n.pt",
    "yolo11n.pt",
    "yolov10n.pt",
    "yolov8n.pt",
]

# processor = AutoImageProcessor.from_pretrained(models_to_benchmark[1])
# model = AutoModelForObjectDetection.from_pretrained(models_to_benchmark[1])

# url = "object\\couch.png"
# image = Image.open(url).convert("RGB")

# inputs = processor(images=image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# target_sizes = torch.tensor([image.size[::-1]])
# results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     class_name = model.config.id2label[label.item()]
#     print(f"Detected {class_name} with {round(score.item(), 3)} confidence at location {box}")

model = YOLO(ultralytics_model[0])
source = "object\\couch.png"
model.predict(source, save=True)