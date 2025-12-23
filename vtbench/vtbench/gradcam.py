import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
from vtbench.models.chart_models.deepcnn import DeepCNN
from torchvision import transforms as T
import matplotlib.pyplot as plt
import random

# Ensure you have pytorch-gradcam installed: pip install pytorch-gradcam
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Configuration ---
NUM_DATASETS = 5
NUM_CLASSES = 2
NUM_ENCODINGS = 4 # e.g., different feature extraction layers or pre-processing methods

# Define your datasets, models, and image paths
# Replace with your actual data loading and model paths
#DATASET_NAMES = [f"dataset_{i+1}" for i in range(NUM_DATASETS)]
#DATASET_NAMES = ["Yoga", "Strawberry" , "GunPoint","Wafer","FordB"]  # Add more as needed
DATASET_NAMES = ["ArrowHead","Strawberry" , "GunPoint","CricketX","CricketY"]  # Add more as needed
CLASS_NAMES = ["class_A", "class_B"]
#ENCODING_TYPES = ["encoding_1", "encoding_2", "encoding_3", "encoding_4"]
ENCODING_TYPES = ["area","bar","line","scatter"]
chart_config = {}
chart_config['area'] = "_charts_color_with_label"
chart_config['line'] = "_charts_color_with_label"
chart_config['bar'] = "_charts_border_color_without_label"
chart_config['scatter'] = "_charts_plain_color_without_label"
# Base directory for saving overlays
OUTPUT_DIR = "grad_cam_overlays"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---
def load_image(image_path):
    """Loads and preprocesses an image for model input."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE     = 128
    transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    pil_img = Image.open(image_path).convert("RGB")
    rgb_np = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    inp = transform(pil_img).unsqueeze(0).to(device)    
    return inp, np.array(inp) / IMG_SIZE # Return tensor and normalized numpy array

def load_model_and_target_layer(dataset_name, encoding_type):
    """Loads a pre-trained model and identifies the target layer for Grad-CAM."""
    # This is a placeholder. You need to load your specific model based on dataset and encoding.
    # Example:
    # if dataset_name == "dataset_1" and encoding_type == "encoding_1":
    #     model = YourModel1()
    #     model.load_state_dict(torch.load("path/to/model1_weights.pth"))
    #     target_layer = model.features[-1] # Example for a VGG-like model
    # else:
    #     # Load other models
    #     pass
    
    # For demonstration, using a pre-trained ResNet and its last convolutional layer
    MODEL_PATH   = r"saved_models/area.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepCNN(input_channels=3, num_classes=NUM_CLASSES).to(device)
    target_layer = get_last_conv_layer(model) 
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.eval()
    return model, target_layer

def get_last_conv_layer(m: nn.Module):
    convs = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    if not convs:
        raise RuntimeError("No Conv2d layer found in the model.")
    return convs[-1]

def generate_grad_cam_overlay(model, target_layer, input_tensor, image_file , original_image_np, target_class_idx, cam_method=GradCAM):
    """Generates and returns a Grad-CAM overlay."""
    '''
    targets = [ClassifierOutputTarget(target_class_idx)]
    #cam = cam_method(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    cam = cam_method(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
    '''
    IMG_SIZE     = 128
    RES_IMG_SIZE     = 512
    transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    pil_img = Image.open(image_file).convert("RGB")
    rgb_np = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inp = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)                          # [1, num_classes]
        pred_cls = int(torch.argmax(logits, dim=1))  
    targets = [ClassifierOutputTarget(pred_cls)]
    cam = GradCAM(model=model, target_layers=[target_layer])
    cam_map = cam(input_tensor=inp, targets=targets)[0]   # [h', w'] in [0,1]

    # resize 
    cam_resized = cv2.resize(cam_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    overlay_rgb = show_cam_on_image(rgb_np, cam_resized, use_rgb=True)  
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
    overlay_bgr = cv2.resize(overlay_bgr, (300, 275))
    overlay_bgr = cv2.filter2D(overlay_bgr, -1, kernel)

    return overlay_bgr
def get_random_file(directory_path, isdir = False , dir_like = None):
    """
    Returns the full path of a random file from the specified directory.
    Ignores subdirectories and only considers files directly within the given path.
    """
    try:
        # Get a list of all entries (files and directories) in the specified path
        entries = os.listdir(directory_path)
        
        # Filter out directories to only include actual files
        if not isdir:   
            files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
        
            if not files:
                print(f"No files found in directory: {directory_path}")
                return None
        else:
            files = [entry for entry in entries
            if os.path.isdir(os.path.join(directory_path, entry)) and dir_like in entry]
        # Select a random file from the list of files
        random_filename = random.choice(files)
        # Construct the full path to the random file
        random_file_path = os.path.join(directory_path, random_filename)
        print(f"Random file is {random_file_path}")
        return random_file_path
        
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
# --- Main Logic ---
if __name__ == "__main__":
    for dataset_idx, dataset_name in enumerate(DATASET_NAMES):
        for class_idx, class_name in enumerate(CLASS_NAMES):
            for encoding_idx, encoding_type in enumerate(ENCODING_TYPES):
                print(f"Processing: Dataset={dataset_name}, Class={class_name}, Encoding={encoding_type}")

                # --- 1. Load Model and Target Layer ---
                model, target_layer = load_model_and_target_layer(dataset_name, encoding_type)

                # --- 2. Load Image (replace with your actual image path for the current dataset/class) ---
                # This is a placeholder. You need to select an image relevant to the current dataset and class.
                # For example: image_path = f"data/{dataset_name}/{class_name}/sample_image.jpg"
                IMAGE_PATH   = f"chart_images/{dataset_name}_images/{encoding_type}{chart_config[encoding_type]}/test"
                root_dir = f"chart_images/{dataset_name}_images/"
                
                sample_image_path = IMAGE_PATH # Replace with a real image path
                image_file = get_random_file(sample_image_path)

                ### Cheeck a random directory in the data_name
                if not os.path.exists(sample_image_path):
                    IMAGE_PATH = get_random_file(root_dir,True ,encoding_type )
                    print (f"New image path {IMAGE_PATH}")
                    sample_image_path = f"{IMAGE_PATH}/test" # Replace with a real image path
                    image_file = get_random_file(sample_image_path)

                if not os.path.exists(sample_image_path):
                    print(f"Warning: Sample image not found at {image_file}. Skipping this iteration.")
                    continue
                
                input_tensor, original_image_np = load_image(image_file)

                # --- 3. Generate Grad-CAM Overlay ---
                # You can choose different CAM methods here (GradCAM, HiResCAM, etc.)
                cam_overlay = generate_grad_cam_overlay(model, target_layer, input_tensor, image_file ,  original_image_np, class_idx, cam_method=GradCAM)

                # --- 4. Save the Overlay ---
                output_filename = f"{dataset_name}_{class_name}_{encoding_type}_grad_cam.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                cv2.imwrite(output_path, cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))
                print(f"Saved Grad-CAM overlay to: {output_path}")

    print(f"\nCompleted generating 40 Grad-CAM overlays in '{OUTPUT_DIR}' directory.")