import torchvision
import onnxruntime
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import cv2
import psutil
import time
import sys
import seaborn as sns
from ina219 import INA219

# 
SHUNT_OHMS = 0.1

# Checking if the INA219 sensor is connected to the I2C bus.
try:
    # Initialize the sensor
    ina = INA219(SHUNT_OHMS, busnum=1, address=0x42)
    ina.configure() # This automatically handles all the complex calibration math!
    print("Power Sensor configured successfully!")
except Exception as e:
    print(f"Failed to initialize Power Sensor: {e}")
    ina = None

#COCO validation dataset paths
images_folder = './data/val2017'
annotations_file = './data/annotations/instances_val2017.json'

# Loading the COCO val_set using torchvision
coco_data = torchvision.datasets.CocoDetection(root=images_folder, annFile=annotations_file)

#Model paths for each ONNX model for the benchmark. Some of the models have been quanitsed as this speeds up inference.
model_paths = {
    "yolov10n": './onnx/quantized/yolov10n_int8.onnx',
    "yolov8n": './onnx/quantized/yolov8n_int8.onnx',
    "yolo26n": './onnx/quantized/yolo26n_int8.onnx',
    "yolo11n": './onnx/quantized/yolo11n_int8.onnx',
    "yolov5n": './onnx/quantized/yolov5n_int8.onnx',
    "yolov6nlite": './onnx/yolov6nlite/yolov6lite_s.onnx',
    "nanodet-plus-m_416": "./onnx/nanodet-plus-m_416.onnx",
    "picodet_s": './onnx/picodet_s_320.onnx'
}

#ONNX Runtime session options to optimise for ARM CPU inference.
#Also limits number of cores to 2 as Pi has 4 cores but also need to run SLAM algorithm.
session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 2
session_options.inter_op_num_threads = 1
session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

#Get images from the COCO dataset with the number of images and a random seed to ensure reproducibility.
def gather_images(num_images=10, seed=42):
    """
    Gathers a specified number of random images from the COCO dataset for benchmarking.
    Arguments:
        num_images: Number of random images to gather.
        seed: Random seed for reproducibility.
    Returns:
        A list of tuples containing the image and its corresponding ground truth annotations.
    """
    random.seed(seed)
    images = []
    for i in range(num_images):
        random_index = random.randint(0, len(coco_data) - 1)
        img, ground_truth = coco_data[random_index] #Gather image and the label associated with the image
        images.append((img, ground_truth))
    return images

#Gathering 100 random images from the COCO val2017 dataset for benchmarking. This will be used to run inference on each model and measure latency, CPU usage, RAM usage, temperature, and power consumption.
batch_images_1 = gather_images(num_images=100, seed=42)

#preprocessing the image for inference.
#Resizes the image to the input size of the model dynamically.
#Converts the image to the correct format and data type.
def preprocess_image(image, model_session):
    """
    Preprocesses the input image for the ONNX model.
    Dynamically resizes the image to match the model's expected input size and format.
    Arguments:
        image: PIL Image to preprocess.
        model_session: ONNX Runtime session to extract input details.
    Returns:
        Preprocessed image as a numpy array ready for inference.
    """
    model_input = None
    for inp in model_session.get_inputs():
        if len(inp.shape) == 4 or inp.name in ['image', 'images', 'input_tensor','x', 'serving_default_images:0']:
            model_input = inp
            break
    
    if model_input is None:
        model_input = model_session.get_inputs()[0]

    target_shape = model_input.shape
    target_type = model_input.type

    img_array = np.array(image.convert("RGB"))

    is_chw = False
    if len(target_shape) >= 4 and target_shape[1] == 3:
        is_chw = True
        height_index, width_index = 2, 3
    else:
        height_index, width_index = 1, 2

    target_height = target_shape[height_index] if isinstance(target_shape[height_index], int) else 640
    target_width = target_shape[width_index] if isinstance(target_shape[width_index], int) else 640

    resized_img = cv2.resize(img_array, (target_width, target_height))

    if target_type == "tensor(float)":
        processed_img = resized_img.astype(np.float32) / 255.0
    elif target_type == "tensor(uint8)": #some of the models have been quantized to uint8 as this speeds up inference time, but can cause issues for some architectures.
        processed_img = resized_img.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported input type: {target_type}")

    if is_chw:
        processed_img = processed_img.transpose(2, 0, 1) #change the image shape depending on the model.

    return np.expand_dims(processed_img, axis=0).astype(np.float32)

def wait_for_cooldown(target_temp=63.0, timeout=300):
    """
    Pauses execution until the Pi cools down to the target temperature.
    timeout: Maximum seconds to wait before giving up and continuing.
    Arguments:
        target_temp: Temperature in Celsius to cool down to before starting the next model.
        timeout: Maximum time in seconds to wait for cooldown before forcing start.
    Returns:
        None
    """
    current_temp = get_pi_temp()
    
    if current_temp <= target_temp:
        return
        
    print(f"Heat Soak Detected ({current_temp:.1f}°C). Cooling down to {target_temp}°C...")
    
    start_time = time.time()
    while True:
        current_temp = get_pi_temp()
        
        if current_temp <= target_temp:
            print(f"Cooldown complete! Starting next model at {current_temp:.1f}°C.")
            break
            
        if (time.time() - start_time) > timeout:
            print(f"Cooldown timeout reached. Forcing start at {current_temp:.1f}°C.")
            break
            
        time.sleep(5) # Check the temperature every 5 seconds

def get_pi_temp():
    """
    Reads the CPU temp of the Pi from the system file.
    Arguments:
        None
    Returns:
        Current CPU temperature in Celsius as a float.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_c = float(f.read().strip()) / 1000.0
            return temp_c
    except Exception:
        pass
        
    try:
        import subprocess
        temp_str = subprocess.check_output(['vcgencmd', 'measure_temp']).decode('utf-8') # Read the temperature using vcgencmd as a fallback method
        return float(temp_str.replace('temp=', '').replace('\'C\n', ''))
    except Exception:
        print("Could not read temperature sensor. Defaulting to 40C.")
        return 40.0

def get_safe_power_metrics():
    """
    Reads live Watts directly from the chip from UPS Hat B.
    Arguments:
        None
    Returns:
        Power in Watts as a float.
    """
    if ina is None:
        return 0.0, 0.0
        
    try:
        # The pi-ina219 library makes this incredibly easy
        voltage_V = ina.voltage()
        power_mW = ina.power()
        power_W = power_mW / 1000.0  # Convert to Watts
        return float(power_W)
    except Exception as e:
        print(f"I2C Read Error: {e}")
        return 0.0, 0.0

def benchmark_model(images):
    """
    Runs the main inference loop for the ONNX models and captures latency, CPU usage, RAM usage, temp and power usage.
    Arguments:
        images: List of tuples containing the image and its corresponding ground truth annotations.
    Returns:
        A dictionary containing the benchmark results for each model, including per-image latency, CPU usage, RAM usage, temperature, and power usage.
    """
    # Get the current process to measure RAM usage.
    process = psutil.Process(os.getpid())
    benchmark_results = {}

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"SKIPPED: Cannot find {model_name} at path: {model_path}")
            continue

        wait_for_cooldown(target_temp=63.0)
        session = onnxruntime.InferenceSession(model_path, sess_options=session_options)
        print(f"\nRunning {model_name}...")

        warmup_data = preprocess_image(images[0][0], session)
        warmup_dict = {}
        for inp in session.get_inputs():
            if inp.name in ['image', 'images', 'input_tensor', 'x', 'serving_default_images:0']:
                warmup_dict[inp.name] = warmup_data
            elif inp.name == 'scale_factor':
                warmup_dict[inp.name] = np.array([[1.0, 1.0]], dtype=np.float32)
            elif inp.name == 'im_shape':
                target_h, target_w = warmup_data.shape[2], warmup_data.shape[3]
                warmup_dict[inp.name] = np.array([[target_h, target_w]], dtype=np.float32)
            else:
                warmup_dict[inp.name] = warmup_data

        # Warmup runs to stabilise performance and make sure any model initialisation is out of the way.
        for _ in range(10):
            try: session.run(None, warmup_dict)
            except: pass

        #Initialise lists to store per-image data values.
        per_image_times = []
        per_image_cpu = []
        per_image_ram = []
        per_image_temp = []
        per_image_power = []
        
        # Get the baseline CPU usage before starting inference.
        psutil.cpu_percent(interval=None)

        for img, ground_truth in images:
            # Preprocess image before inference
            input_data = preprocess_image(img, session)
            
            input_dict = {}
            # Build the input dictionary based on the model inputs.
            for inp in session.get_inputs():
                if inp.name in ['image', 'images', 'input_tensor', 'x', 'serving_default_images:0']:
                    input_dict[inp.name] = input_data
                elif inp.name == 'scale_factor':
                    orig_w, orig_h = img.size
                    target_h, target_w = input_data.shape[2], input_data.shape[3]
                    scale_y = target_h / orig_h
                    scale_x = target_w / orig_w
                    input_dict[inp.name] = np.array([[scale_y, scale_x]], dtype=np.float32)
                elif inp.name == 'im_shape':
                    target_h, target_w = input_data.shape[2], input_data.shape[3]
                    input_dict[inp.name] = np.array([[target_h, target_w]], dtype=np.float32)
                else:
                    input_dict[inp.name] = input_data

            try:
                #Main data gathering loop
                start_time = time.perf_counter() # Start the timer right before inference
                outputs = session.run(None, input_dict)
                end_time = time.perf_counter() # End the time right after inference to get the most accurate latency measurement.

                current_temp = get_pi_temp() # Gather the CPU temp right after inference to see the impact of the model on the temperature.
                
                # Get the CPU and RAM usage 
                current_cpu = psutil.cpu_percent(interval=None)
                current_ram = process.memory_info().rss / (1024 * 1024)

                #Measure the current power usage in Watts
                current_power= get_safe_power_metrics()

                per_image_times.append(end_time - start_time)
                per_image_cpu.append(current_cpu)
                per_image_ram.append(current_ram)
                per_image_temp.append(current_temp)
                per_image_power.append(current_power)

            except Exception as e:
                print(f"Error running {model_name} on image: {e}")

        # Get the resolution of the input image for this model during warmup as it can affect results.
        shape = warmup_data.shape
        if len(shape) >= 4 and shape[1] == 3:
            h, w = shape[2], shape[3]
        else:
            h, w = shape[1], shape[2]

        benchmark_results[model_name] = {
            "per_image_times": per_image_times,
            "per_image_cpu": per_image_cpu,
            "per_image_ram": per_image_ram,
            "per_image_temp": per_image_temp,
            "per_image_power": per_image_power,
            "resolution": (h, w)

        }
        
    return benchmark_results

def plot_graphs(results):
    """
    Plots the graphs for latency, CPU usage, RAM usage, and temp change over time from benchamrk script results.
    Arguments:
        results: A dictionary containing the benchmark results for each model, including per-image latency, CPU
    Returns:
        None but saves the plot as an image file.
    """
    #Check if any models ran successfully and have data.
    successful_models = [m for m in results.keys() if len(results[m]['per_image_times']) > 0]
    
    if not successful_models:
        print("All models failed during inference. Nothing to plot!")
        return
        
    print(f"Plotting data for: {successful_models}")

    latencies = []
    #Plot latency graphs from data with all successful models.
    for m in successful_models:
        h, w = results[m]['resolution']
        megapixels = (h * w) / 1_000_000.0
        
        norm_times = [(t * 1000) / megapixels for t in results[m]['per_image_times']]
        latencies.append(norm_times)

    #Plot CPU usage data from all successful models.
    cpus = [results[m]['per_image_cpu'] for m in successful_models]
    
    # Get average RAM usage for each model to plot.
    avg_rams = [sum(results[m]['per_image_ram']) / len(results[m]['per_image_ram']) for m in successful_models]

    # Get change in temperature to plot against time.
    temps = [results[m]['per_image_temp'] for m in successful_models]
    
    # 4 row subplot of CPU, RAM, Temp and Latency.
    fig, axes = plt.subplots(4, 1, figsize=(8, 20), constrained_layout=True)
    fig.suptitle('CPU, RAM, Temp, Latency Measurements', fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(successful_models)))

    # Beeswarm plot for latency to show the distribution of latency values per image.
    axes[0].set_title('Normalized Latency (Every Single Frame)')
    axes[0].set_ylabel('Latency per Megapixel (ms/MP)')
    sns.stripplot(data=latencies, ax=axes[0], palette='tab10', size=3, jitter=True, alpha=0.6)
    axes[0].set_xticks(range(len(successful_models)))
    axes[0].set_xticklabels(successful_models, rotation=45)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    # Box plot for CPU usage to show the average CPU load for each model including spikes in CPU usage as part of the box plot.
    bplot2 = axes[1].boxplot(cpus, tick_labels=successful_models, patch_artist=True, showfliers=False)
    axes[1].set_title('CPU Load (Baseline & Spikes)')
    axes[1].set_ylabel('CPU Usage (%)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # bar plot for RAM usage
    axes[2].bar(successful_models, avg_rams, color=colors, edgecolor='black')
    axes[2].set_title('Average RAM Footprint')
    axes[2].set_ylabel('Megabytes (MB)')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.5)
    axes[2].set_ylim(bottom=400)

    #Temp change line plot to show the overall heating trend of the CPU, the initial change will have different starting temperatures.
    axes[3].set_title('Overall CPU Heating Trend')
    axes[3].set_ylabel('Temp Increase (°C)')
    axes[3].set_xlabel('Images Processed (Time)')
    
    for idx, model in enumerate(successful_models):
        temps = results[model].get('per_image_temp', [])
    
        if len(temps) > 1:
            starting_temp = temps[0]
            temp_change = np.array(temps) - starting_temp
            x_values = np.arange(len(temp_change))

            axes[3].plot(
                x_values,
                temp_change,
                color=colors[idx],
                alpha=0.3,
                linewidth=1
            )
            
            slope, intercept = np.polyfit(x_values, temp_change, 1)
            trendline = (slope * x_values) + intercept

            axes[3].plot(
                x_values,
                trendline,
                label=f"{model} ({slope:.3f} °C/img)",
                color=colors[idx],
                linewidth=2
            )

    axes[3].legend(loc='upper left', fontsize=8)
    axes[3].grid(True, linestyle='--', alpha=0.5)

    colors = plt.cm.tab10(np.linspace(0, 1, len(successful_models)))
    
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    image_name = "improved_hardware_benchmark.png"
    plt.savefig(image_name, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved as '{image_name}'")

def plot_power_and_battery(results):
    """
    Plots the average power consumption for each model from the benchmark script.
    Arguments:
        results: A dictionary containing the benchmark results for each model, including per-image power usage.
    Returns:
        None but saves the plot as an image file.
    """
    successful_models = [m for m in results.keys()]
    
    if not successful_models:
        print("No power data found! Did the Waveshare sensor disconnect?")
        return

    fig, axes = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    fig.suptitle('Power Benchmark', fontsize=18, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(successful_models)))

    avg_powers = []
    
    #Plot average power usage for each model as bar plot to compare the average usage.
    for idx, model in enumerate(successful_models):
        power_watts = results[model]['per_image_power']
        latency_secs = results[model]['per_image_times']

        avg_w = sum(power_watts) / len(power_watts)
        avg_powers.append(avg_w)

    axes.bar(successful_models, avg_powers, color=colors, edgecolor='black', alpha=0.8)
    axes.set_title('Average Power Draw', fontsize=14)
    axes.set_ylabel('Average Power (Watts)')
    axes.tick_params(axis='x', rotation=45)
    axes.grid(True, axis='y', linestyle='--', alpha=0.5)

    axes.set_ylim(bottom=5.0)
    
    image_name = 'power_battery_benchmark.png'
    plt.savefig(image_name, dpi=300, bbox_inches='tight')
    print(f"Power Dashboard saved as '{image_name}'")

# Run the benchmark and plot.
results_data = benchmark_model(batch_images_1)
plot_graphs(results_data)
plot_power_and_battery(results_data)