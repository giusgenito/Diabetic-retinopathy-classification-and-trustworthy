#!/usr/bin/env python
# coding: utf-8

# # Diabetic Retinopathy Preprocessing

# ## Library Imports
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
from functools import reduce

# ## Helper Functions

# These two functions are adapted from the keras-yolo3 GitHub repository:
# https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/utils.py

def compose(*funcs):
    """Compose a sequence of functions."""
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image_pil, target_size_wh, padding_color):
    """
    Resizes a PIL Image to a target size while maintaining aspect ratio by adding padding.
    
    Args:
        image_pil (PIL.Image): The input PIL Image.
        target_size_wh (tuple): The target width and height (w, h).
        padding_color (int): The color to use for padding (for grayscale images).
        
    Returns:
        PIL.Image: The resized and padded image.
    """
    iw, ih = image_pil.size
    w_target, h_target = target_size_wh

    if iw == 0 or ih == 0:  # Handle empty input image
        return Image.new('L', target_size_wh, padding_color)

    scale = min(w_target / iw, h_target / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    # Ensure new dimensions are at least 1 pixel
    nw = max(1, nw)
    nh = max(1, nh)

    resized_image = image_pil.resize((nw, nh), Image.BICUBIC)
    
    new_image = Image.new('L', target_size_wh, padding_color)  # 'L' for grayscale
    new_image.paste(resized_image, ((w_target - nw) // 2, (h_target - nh) // 2))
    return new_image


# ## Preprocessing Pipeline Functions

def convert_to_grayscale(img_bgr):
    """Converts a BGR image to grayscale using specific weights."""
    if len(img_bgr.shape) == 2: return img_bgr
    if img_bgr.shape[2] == 1: return img_bgr.reshape(img_bgr.shape[0], img_bgr.shape[1])
    # Specific weights for BGR to Grayscale conversion
    b, g, r = cv2.split(img_bgr)
    gray_img = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray_img.astype(np.uint8)

def apply_clahe(img_gray):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)

def apply_gaussian_blur(img_gray, kernel_size=(5, 5)):
    """Applies a Gaussian blur to the image."""
    return cv2.GaussianBlur(img_gray, kernel_size, 0)

def apply_median_filter(img_gray, kernel_size=5):
    """Applies a median filter to the image."""
    return cv2.medianBlur(img_gray, kernel_size)

def cv2_to_pil_grayscale(img_cv2):
    """Converts a CV2 grayscale image to a PIL Image."""
    return Image.fromarray(img_cv2, mode='L')

def pil_to_cv2_grayscale(img_pil):
    """Converts a PIL grayscale image to a CV2 NumPy array."""
    return np.array(img_pil)

def segment_fundus_and_create_mask(image_cv2_gray, image_name_for_debug=""):
    """
    Segments the ocular fundus, returning a binary mask and its bounding box.
    
    Returns:
        tuple: (mask, bounding_box) where bounding_box is (x, y, w, h) or None.
    """
    # Segmentation tuning parameters
    blur_kernel_size_seg = (15, 15)
    threshold_value = 30
    morph_kernel_size_open = (15, 15)   # Kernel for MORPH_OPEN
    morph_kernel_size_close = (35, 35)  # Larger kernel for MORPH_CLOSE to merge regions
    
    blurred_for_seg = cv2.GaussianBlur(image_cv2_gray, blur_kernel_size_seg, 0)
    
    # Using a fixed threshold. cv2.THRESH_OTSU could be an alternative if this isn't robust.
    _, thresh_img = cv2.threshold(blurred_for_seg, threshold_value, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size_close)
    
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel_open, iterations=1)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(image_cv2_gray)
    bounding_box = None

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Consider filtering out contours that are too small relative to the image area.
        fundus_contour = contours[0]
        hull = cv2.convexHull(fundus_contour)
        cv2.drawContours(mask, [hull], -1, (255), thickness=cv2.FILLED)
        bounding_box = cv2.boundingRect(hull)  # Returns (x, y, w, h)
    else:
        print(f"Warning: No fundus contour found for {image_name_for_debug}. The resulting image might be black.")
        
    return mask, bounding_box

# --- Global Parameters ---
FINAL_IMAGE_SIZE = (512, 512)
FUNDUS_TARGET_SCALE_FACTOR = 0.9  # The fundus will occupy 90% of the final image's largest dimension.
DEBUG_SAVE_INTERMEDIATE = False   # Set to True to save intermediate debug images.
DEBUG_OUTPUT_DIR = '/home/jupyter-sdm/GENITO/LAVORO_COMPLETO/Dataset_resize/1_IDRiD_DEBUG/'
if DEBUG_SAVE_INTERMEDIATE:
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

# --- Main Processing Loop ---
# This script processes multiple diabetic retinopathy datasets.
# The main loop iterates through a CSV file for each dataset,
# applies a series of preprocessing steps to each image,
# and saves the result into a class-specific folder.


# --- 1. IDRiD Dataset ---

# Processing IDRiD Training Set
print("--- Processing IDRiD Training Set ---")
csv_path = '/home/jupyter-sdm/GENITO/DATASETS/1_IDRiD/a. IDRiD_Disease Grading_Training Labels.csv'
image_dir = '/home/jupyter-sdm/GENITO/DATASETS/1_IDRiD/train'
final_output_dir = '/home/jupyter-sdm/GENITO/LAVORO_COMPLETO/Dataset_resize/1_IDRiD/'
file_extension = ".jpg"
column_class_name = 'Retinopathy grade'
colum_image_name = 'Image name'

# The core processing logic is repeated for each dataset below.
# For brevity, the detailed comments are in this first block.
try:
    df = pd.read_csv(csv_path)
    classes = df[column_class_name].unique()
    for cls in classes:
        os.makedirs(os.path.join(final_output_dir, str(cls)), exist_ok=True)

    error_count = 0
    null_bbox_count = 0

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="IDRiD Train", ncols=100):
        image_name_no_ext = row[colum_image_name]
        image_name = image_name_no_ext + file_extension
        image_class = str(row[column_class_name])
        src_path = os.path.join(image_dir, image_name)

        # Handle different file extensions if needed
        if not os.path.exists(src_path):
            src_path_jpeg = os.path.join(image_dir, image_name_no_ext + ".jpeg")
            if os.path.exists(src_path_jpeg):
                src_path = src_path_jpeg
                image_name = image_name_no_ext + ".jpeg"
            else:
                print(f"Warning: {image_name_no_ext} not found in {image_dir} with .jpg or .jpeg extension.")
                error_count += 1
                continue
        
        img_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Error reading image {src_path}")
            error_count += 1
            continue
        
        try:
            # 1. Convert to Grayscale
            img_cv2_gray = convert_to_grayscale(img_bgr)

            # 2. Segment the fundus to create a mask and get the bounding box
            fundus_mask, fundus_bbox = segment_fundus_and_create_mask(img_cv2_gray.copy(), image_name)

            # 3. Apply enhancement pipeline: CLAHE -> Gaussian Blur -> Median Filter
            img_clahe = apply_clahe(img_cv2_gray)
            img_gaussian_blurred = apply_gaussian_blur(img_clahe)
            img_median_filtered = apply_median_filter(img_gaussian_blurred)
            fully_processed_gray_data = img_median_filtered

            # 4. Apply the mask to black out the background
            masked_processed_fundus_cv2 = cv2.bitwise_and(fully_processed_gray_data, fully_processed_gray_data, mask=fundus_mask)

            # --- Fundus Size Normalization ---
            if fundus_bbox and fundus_bbox[2] > 0 and fundus_bbox[3] > 0:
                x, y, w_bbox, h_bbox = fundus_bbox
                # Crop the fundus using the bounding box
                cropped_fundus_cv2 = masked_processed_fundus_cv2[y:y+h_bbox, x:x+w_bbox]

                # Calculate new dimensions for the cropped fundus
                target_max_dim_px = int(max(FINAL_IMAGE_SIZE) * FUNDUS_TARGET_SCALE_FACTOR)
                current_max_dim_bbox = max(w_bbox, h_bbox)
                scale_ratio = target_max_dim_px / current_max_dim_bbox
                
                new_w = max(1, int(w_bbox * scale_ratio))
                new_h = max(1, int(h_bbox * scale_ratio))
                
                interpolation = cv2.INTER_AREA if scale_ratio < 1 else cv2.INTER_CUBIC
                resized_cropped_fundus_cv2 = cv2.resize(cropped_fundus_cv2, (new_w, new_h), interpolation=interpolation)
                
                image_to_letterbox_pil = cv2_to_pil_grayscale(resized_cropped_fundus_cv2)
            else:
                print(f"Warning: Invalid or null bounding box for {image_name}. The image will be black.")
                image_to_letterbox_pil = Image.new('L', (1, 1), 0) # Placeholder for letterboxing
                null_bbox_count += 1
            
            # 5. Letterbox: Pad the normalized fundus to the final 512x512 size
            letterboxed_img_pil = letterbox_image(image_to_letterbox_pil, FINAL_IMAGE_SIZE, padding_color=0) 
            
            # 6. Convert back to a NumPy array for saving
            final_img_to_save = pil_to_cv2_grayscale(letterboxed_img_pil)
            
            # 7. Save the processed image as PNG
            output_filename = os.path.splitext(image_name)[0] + '.png'
            output_path = os.path.join(final_output_dir, image_class, output_filename)
            cv2.imwrite(output_path, final_img_to_save)
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

    print(f"Images with reading/processing errors: {error_count}")
    print(f"Images with null or invalid fundus bounding box: {null_bbox_count}")
    print("IDRiD training set processing complete!")

except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}")
