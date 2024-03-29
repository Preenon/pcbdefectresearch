import os
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet

def zhu_preprocessing(input_image_path):
    # Step 1: Read image and convert to grayscale
    img = cv2.imread(input_image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Wavelet denoising
    denoised_img = denoise_wavelet(img_gray)
    
    # Scale the denoised image pixel values back to the range of 0 to 255
    denoised_img = (denoised_img * 255).astype(np.uint8)

    # Step 3: Histogram equalization
    equalized_img = cv2.equalizeHist(denoised_img)
    
    # Combine grayscale preprocessed image with color information from original image
    img_colorized = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)
    img_output = cv2.addWeighted(img_colorized, 0.2, img, 0.8, 0)
    
    return img_colorized, img_output

# Input and output directories
input_dir = "Inputs"
output_dir = "Outputs"
colored_output_dir = "Colored Outputs"

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(colored_output_dir, exist_ok=True)

# Process images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_image_path = os.path.join(input_dir, filename)
        
        # Preprocess the image
        output_image, colored_output_image = zhu_preprocessing(input_image_path)
        
        # Save the preprocessed image (after histogram equalization) to the "Outputs" folder
        #output_image_path = os.path.join(output_dir, filename)
        #cv2.imwrite(output_image_path, output_image)
        
        # Save the colored output image to the "Colored Outputs" folder
        colored_output_image_path = os.path.join(colored_output_dir, filename)
        cv2.imwrite(colored_output_image_path, colored_output_image)

        #print(f"Processed {filename}")

print("Processing complete.")
