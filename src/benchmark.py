import torchvision
import onnxruntime
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import cv2
import psutil

images_folder = './data/val2017'
annotations_file = './data/annotations/instances_val2017.json'

coco_data = torchvision.datasets.CocoDetection(root=images_folder, annFile=annotations_file)

model_paths = {
    "yolov10n": './onnx/yolov10n.onnx',
    "yolov8n": './onnx/yolov8n.onnx',
    "yolo26n": './onnx/yolo26n.onnx',
    "yolo11n": './onnx/yolo11n.onnx',
    "ssd-mobilenet": './onnx/ssd-mobilenet.onnx',
    "rtdetr-l": './onnx/rtdetr-l.onnx',
    "efficientdet-d1": './onnx/efficientdet-d1.onnx',
    "centernet": './onnx/centernet.onnx',
    "yolos-tiny": './onnx/yolos-tiny-onnx/model.onnx',
    "r18vd": "./onnx/r18vd/model.onnx"
}

session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 4


def gather_images(num_images=10, seed=42):
    random.seed(seed)
    images = []
    for i in range(num_images):
        random_index = random.randint(0, len(coco_data) - 1)
        img, ground_truth = coco_data[random_index]
        images.append((img, ground_truth))
    
    
    return images

batch_images_1 = gather_images(num_images=100, seed=42)
batch_images_2 = gather_images(num_images=100, seed=43)
batch_images_3 = gather_images(num_images=100, seed=44)

def preprocess_image(image, model_session):
    model_input = model_session.get_inputs()[0]
    target_type = model_input.type
    target_shape = model_input.shape

    img_array = np.array(image)

    if len(target_shape) >= 4 and target_shape[1] == 3:
        height_index, width_index = 2, 3
    else:
        height_index, width_index = 1, 2

    target_height = target_shape[height_index] if isinstance(target_shape[height_index], int) else 640
    target_width = target_shape[width_index] if isinstance(target_shape[width_index], int) else 640

    resized_img = cv2.resize(img_array, (target_width, target_height))

    if target_type == "tensor(uint8)":
        return np.expand_dims(resized_img, axis=0).astype(np.uint8)
    elif target_type == "tensor(float)":
        img_float = resized_img.astype(np.float32) / 255.0

        img_chw = np.transpose(img_float, (2, 0, 1))

        return np.expand_dims(img_chw, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported input type: {target_type}")

def benchmark_model(images):
    process = psutil.Process(os.getpid())
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}. Skipping.")
            continue

        session = onnxruntime.InferenceSession(model_path, sess_options=session_options)
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        print(f"Loaded {model_name} with input name: {input_name}")
        print(f"Output names: {output_names}")

        runs = 0
        total_time = 0.0

        psutil.cpu_percent(interval=None)  # Reset CPU percent measurement

        for img, ground_truth in images:
            input_data = preprocess_image(img, session)
            try:
                start_time = time.perf_counter()
                outputs = session.run(output_names, {input_name: input_data})
                end_time = time.perf_counter()
                if runs >= 10:
                    total_time += end_time - start_time
                runs += 1
            except Exception as e:
                print(f"Error running model {model_name}: {e}")
        avg_cpu = psutil.cpu_percent(interval=None)

        ram_mb = process.memory_info().rss / (1024 * 1024)

        avg_time = total_time / runs if runs > 0 else 0
        print(f"Model: {model_name}, Average Time: {avg_time}")

        print(f"Model: {model_name}, Average CPU Usage: {avg_cpu}%, RAM Usage: {ram_mb:.2f} MB")
benchmark_model(batch_images_1)
# benchmark_model(batch_images_2)
# benchmark_model(batch_images_3)

