from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch

models_to_benchmark =[
    "hustvl/yolos-tiny",
    "PekingU/rtdetr_r101vd_coco_o365"
]

processor = AutoImageProcessor.from_pretrained(models_to_benchmark[1])
model = AutoModelForObjectDetection.from_pretrained(models_to_benchmark[1])

url = "object\\couch.png"
image = Image.open(url).convert("RGB")

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    class_name = model.config.id2label[label.item()]
    print(f"Detected {class_name} with {round(score.item(), 3)} confidence at location {box}")