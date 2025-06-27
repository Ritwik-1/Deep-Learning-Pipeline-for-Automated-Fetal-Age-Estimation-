import os
import re
import cv2
import math
import numpy as np
import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

cleaned_folder = r'OUTPUT_2\CLEANED_SEG'
pattern = re.compile(r'^(Patient\d+)_.*Fetal (abdomen|brain|femur)_seg_.*\.png$', re.IGNORECASE)
patient_files = defaultdict(dict)

# PIXEL_SPACING 
pixel_spacing = [0.01,0.01]

# get abd, femur and brain image for the patient 
def get_image_size_from_mask_filename(filename, folder_path = r"FETAL_PLANES_ZENODO\Images"):

    pattern = r'_(\d+_of_\d+)'
    match = re.search(pattern, filename)
    if not match:
        raise ValueError("Pattern '_<num>_of_<num>' not found in filename")
    
    part = match.group(1) 
    
    split_index = filename.find(part) + len(part)
    base_filename = filename[:split_index] + ".png" 
    full_path = os.path.join(folder_path, base_filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    print(f"Full path for {filename} : ",full_path)
    
    img_gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    return img_gray, full_path

def preprocess_mask(mask,image):
    # 1) resize using bilinear interpolation to image size 
    # H, W = 224, 224
    # 2) Erosion followed by dilation (i.e., morphological opening)
    # Create a 5x5 cross-shaped structuring element
    # Morphological opening = erosion then dilation
    # 3) median blur filter 

    W, H = image.shape
    mask = cv2.resize(mask, (H,W), interpolation=cv2.INTER_LINEAR)
    thresholded = (mask > 0.6).astype(np.uint8)  
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    cleaned_mask = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    smoothed = cv2.medianBlur(cleaned_mask, ksize=13)

    return smoothed

def get_head_abd(mask, image, filename, show=True):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("No contours found.")
        return None, None, image
    contour = max(contours, key=cv2.contourArea)

    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 5:
        print("Not enough points to fit an ellipse.")
        return None, None, image

    ellipse = cv2.fitEllipse(approx)
    (x, y), (MA, ma), angle = ellipse

    a, b = max(MA, ma) / 2, min(MA, ma) / 2
    h = ((a - b)**2) / ((a + b)**2)
    circumference = math.pi * (3*(a + b) - math.sqrt((3*a + b)*(a + 3*b)))
    bpd = min(MA, ma)

    # Overlay mask 
    if image is not None:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        overlay = image.copy()
        mask_rgb = np.zeros_like(image, dtype=np.uint8)
        mask_rgb[mask > 0] = [0, 0, 255]  
        blended = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)
    else:
        blended = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    cv2.ellipse(blended, ellipse, (0, 255, 0), 2)
    cv2.drawContours(blended, [approx], -1, (255, 0, 0), 1)

    if show:
        plt.imshow(blended[..., ::-1])  
        plt.title(f"Circumference: {circumference:.2f}, BPD: {bpd:.2f}")
        plt.axis('off')
        plt.savefig(f"OVERLAY_SEG/{filename}.png")
        plt.show()

    return circumference, bpd, blended

def get_femur_length(femur_mask,img,filename, show=True):
    contours, _ = cv2.findContours(femur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None, femur_mask
    
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)  
    (cx, cy), (w, h), angle = rect

    femur_length = max(w, h)
    if show:
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        if img is not None:
            if len(img.shape) == 2:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                vis = img.copy()
        else:
            vis = cv2.cvtColor(femur_mask * 255, cv2.COLOR_GRAY2BGR)

        overlay = np.zeros_like(vis)
        overlay[femur_mask > 0] = (0, 255, 0)  
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.3, 0)

        cv2.drawContours(vis, [box], 0, (255, 0, 0), 2)

        plt.figure(figsize=(6, 6))
        plt.imshow(vis[..., ::-1])  
        plt.title(f"Femur Length: {femur_length:.2f} pixels")
        plt.savefig(f"OVERLAY_SEG/{filename}.png")
        plt.axis('off')
        plt.show()
    else:
        vis = None
    return femur_length, vis

def get_head_measurements(brain_mask,brain_name,filename, show=True):
    circumference, bpd, vis = get_head_abd(brain_mask, brain_name,filename, show=show)
    return circumference, bpd, vis

def get_abdomen_measurement(abdomen_mask,abdomen_name,filename, show=True):
    circumference, _, vis = get_head_abd(abdomen_mask,abdomen_name,filename, show=show)
    return circumference, vis

def estimate_ga(BPD, HC, AC, FL):
    GA = (
        10.6
        - 0.168 * BPD
        + 0.045 * HC
        + 0.03 * AC
        + 0.058 * FL
        + 0.002 * (BPD ** 2)
        + 0.002 * (FL ** 2)
        + 0.0005 * (BPD * AC)
        - 0.005 * (BPD * FL)
        - 0.0002 * (HC * AC)
        + 0.0008 * (HC * FL)
        + 0.0005 * (AC * FL)
    )
    return GA

def estimate_fbw(HC, AC, FL):
    log10_BW = (
        1.326
        - 0.00326 * AC * FL
        + 0.0107 * HC
        + 0.0438 * AC
        + 0.158 * FL
    )
    print("log10_BW : ",log10_BW)
    BW = 10 ** log10_BW  
    return BW

#####################################################################################
for filename in os.listdir(cleaned_folder):
    if filename.endswith('.png'):
        match = pattern.match(filename)
        if match:
            patient_id, organ_class = match.groups()
            patient_files[patient_id][organ_class.lower()] = os.path.join(cleaned_folder, filename)

patient_parameters = {}

for patient_id, class_files in patient_files.items():
    print(f"\nProcessing {patient_id}:-")
    print((class_files["abdomen"].split("\\")[-1])[0])

    abdomen_image, abdomen_image_file_name = get_image_size_from_mask_filename(class_files["abdomen"].split("\\")[-1])
    brain_image, brain_image_file_name = get_image_size_from_mask_filename(class_files["brain"].split("\\")[-1])
    femur_image, femur_image_file_name = get_image_size_from_mask_filename(class_files["femur"].split("\\")[-1])

    abdomen_image = np.array(abdomen_image)
    brain_image = np.array(brain_image)
    femur_image = np.array(femur_image)

    abdomen_mask = preprocess_mask(np.array(Image.open(class_files.get('abdomen')).convert('L')),abdomen_image)
    brain_mask = preprocess_mask(np.array(Image.open(class_files.get('brain')).convert('L')),brain_image)
    femur_mask = preprocess_mask(np.array(Image.open(class_files.get('femur')).convert('L')),femur_image)

    HC, BPD, brain_vis = get_head_measurements(brain_mask,brain_image,brain_image_file_name.split(".")[0].split("\\")[-1])
    AC, abdomen_vis = get_abdomen_measurement(abdomen_mask,abdomen_image,abdomen_image_file_name.split(".")[0].split("\\")[-1])
    FL, femur_vis = get_femur_length(femur_mask,femur_image,femur_image_file_name.split(".")[0].split("\\")[-1])

    print(f"Head Circumference (HC): {HC:.2f} pixels")
    print(f"Biparietal Diameter (BPD): {BPD:.2f} pixels")
    print(f"Abdominal Circumference (AC): {AC:.2f} pixels")
    print(f"Femur Length (FL): {FL:.2f} pixels")

    # convert pixels to cm
    # pixel_spacing = ds.PixelSpacing  # e.g., [0.5, 0.5]
    pixel_size_cm = float(pixel_spacing[0]) / 10  
    HC *= pixel_size_cm
    AC *= pixel_size_cm
    BPD *= pixel_size_cm
    FL *= pixel_size_cm

    #### GA and FBW calculation
    GA = estimate_ga(BPD,HC,AC,FL)
    FBW = estimate_fbw(HC, AC, FL)
    patient_parameters[patient_id] = [BPD,HC,AC,FL,GA,FBW]

    print([BPD,HC,AC,FL,GA,FBW])
    print("GA : ",GA)
    print("FBW : ",FBW)
    
df = pd.DataFrame.from_dict(patient_parameters, orient='index', columns=["BPD", "HC", "AC", "FL", "GA", "FBW"])
df.index.name = "PatientID"
df.to_csv("patient_parameters.csv")

