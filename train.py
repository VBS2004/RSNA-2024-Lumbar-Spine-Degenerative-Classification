import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import convolve
from threading import Thread, Lock

# Define the paths for the dataset and the output directories
base_path = "E:/rsna/"
train_dirs = [
    'filtered/hsv/train/normal',
    'filtered/hsv/train/Moderate',
    'filtered/hsv/train/Severe',
    'filtered/hsv/test/normal',
    'filtered/hsv/test/Moderate',
    'filtered/hsv/test/Severe',
    'filtered/lab/train/normal',
    'filtered/lab/train/Moderate',
    'filtered/lab/train/Severe',
    'filtered/lab/test/normal',
    'filtered/lab/test/Moderate',
    'filtered/lab/test/Severe',
    'filtered/gray/train/normal',
    'filtered/gray/train/Moderate',
    'filtered/gray/train/Severe',
    'filtered/gray/test/normal',
    'filtered/gray/test/Moderate',
    'filtered/gray/test/Severe',
    
]

# Function to create output directories if they don't exist
def create_dirs():
    for dir_path in train_dirs:
        os.makedirs(os.path.join(base_path, dir_path), exist_ok=True)

# Define the LoG filter function
def LoG_filter(image, sigma, size=None):
    if size is None:
        size = int(6 * sigma + 1) if sigma >= 1 else 7
    if size % 2 == 0:
        size += 1
    x, y = np.meshgrid(np.arange(-size//2 + 1, size//2 + 1), np.arange(-size//2 + 1, size//2 + 1))
    kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(np.abs(kernel))
    
    # Handle multi-channel images
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[2]):
            result[:,:,i] = convolve(image[:,:,i].astype(np.float32), kernel)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = convolve(image.astype(np.float32), kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

# Thread-safe image saving function
lock = Lock()

def save_image(image, path):
    with lock:
        cv2.imwrite(path, image)

# Counter for tracking the number of processed images
counter_lock = Lock()
counter = 0

# Function to process images in a thread
def process_image(img_path, train_mode=True):
    global counter
    img_name = os.path.basename(img_path)
    img = Image.open(img_path)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE) #Applying to every image
    cat=img_path.split('/')[-1]
    cat=cat.split('\\')[0]
    # Convert to OpenCV format for further processing
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Convert to Grayscale and apply LoG filter
    gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cv_img_log = LoG_filter(gray_img, sigma=2.0)

    save_gray_path = os.path.join(base_path, f'filtered/gray/train/{cat}' if train_mode else f'filtered/gray/test/{cat}', img_name)
    save_image(cv_img_log, save_gray_path)

	#HSV
    hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    cv_img_log = LoG_filter(hsv_img, sigma=2.0)

    save_hsv_path = os.path.join(base_path, f'filtered/hsv/train/{cat}' if train_mode else f'filtered/hsv/test/{cat}', img_name)
    save_image(cv_img_log, save_hsv_path)
    
	# LAB
    lab_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    cv_img_log = LoG_filter(lab_img, sigma=2.0)

    save_LAB_path = os.path.join(base_path, f'filtered/lab/train/{cat}' if train_mode else f'filtered/lab/test/{cat}', img_name)
    save_image(cv_img_log, save_LAB_path)
    # Update and print the count every 500 images processed
    print(cat.split('\\')[0])
    with counter_lock:
        counter += 1
        if counter % 500 == 0:
            print(f"Converted {counter} images.")
# Function to process images from a source directory
def process_images(src_dir, train_mode=True):
    threads = []
    
    for img_name in os.listdir(src_dir):
        img_path = os.path.join(src_dir, img_name)
        thread = Thread(target=process_image, args=(img_path, train_mode))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

# Create output directories
create_dirs()

train_src_dirs = [
    "E:/rsna/neural_foraminal_narrowing/Moderate",
    "E:/rsna/neural_foraminal_narrowing/normal",
    "E:/rsna/neural_foraminal_narrowing/Severe"
]

for src_dir in train_src_dirs:
    process_images(src_dir, train_mode=True)

test_dirs = [
    'E:/rsna/neural_foraminal_narrowing/test/Moderate',
    'E:/rsna/neural_foraminal_narrowing/test/normal',
    'E:/rsna/neural_foraminal_narrowing/test/Severe'
]

for test_src_dir in test_dirs:
    process_images(test_src_dir, train_mode=False)
