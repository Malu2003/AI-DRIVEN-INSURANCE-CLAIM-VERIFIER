"""
Error Level Analysis (ELA) utility for detecting image manipulation.
ELA works by:
1. Saving image at a known quality level (e.g. 90)
2. Reloading and resaving
3. Computing the pixel difference to find areas that changed more than expected
"""

import os
import numpy as np
from PIL import Image
import cv2

def compute_ela(image_path, quality=90, scale=10):
    """
    Compute Error Level Analysis (ELA) on an image.
    
    Args:
        image_path (str): Path to input image
        quality (int): JPEG save quality (0-100)
        scale (int): Multiply the difference by this to make it more visible
        
    Returns:
        numpy.ndarray: ELA difference image (scaled up for visibility)
    """
    # Read original image
    original = Image.open(image_path).convert('RGB')
    
    # Save to temporary JPEG with specified quality
    temp_path = "temp_ela.jpg"
    original.save(temp_path, 'JPEG', quality=quality)
    
    # Read back the JPEG
    recompressed = Image.open(temp_path)
    
    # Convert both to numpy arrays
    orig_array = np.array(original)
    recom_array = np.array(recompressed)
    
    # Compute absolute difference and scale
    diff = np.abs(orig_array - recom_array) * scale
    # Clip to valid image range
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    
    # Cleanup
    os.remove(temp_path)
    
    return diff


def compute_ela_score(diff_image):
    """Compute a normalized ELA score in [0,1] from the ELA diff image.

    Uses the mean intensity of the ELA image normalized by 255.
    Higher values indicate larger compression artifacts (more suspicious).
    """
    if isinstance(diff_image, np.ndarray):
        # convert to grayscale intensity
        if diff_image.ndim == 3:
            gray = np.mean(diff_image, axis=2)
        else:
            gray = diff_image
        mean_val = float(np.mean(gray))
        score = np.clip(mean_val / 255.0, 0.0, 1.0)
        return score
    else:
        raise ValueError('diff_image must be a numpy array')


def save_ela_heatmap(diff_image, out_path, cmap=cv2.COLORMAP_JET):
    """Save ELA diff image as a heatmap to `out_path`.

    Uses OpenCV colormap for a compact dependency set.
    """
    if diff_image.ndim == 3:
        gray = np.mean(diff_image, axis=2).astype(np.uint8)
    else:
        gray = diff_image.astype(np.uint8)

    # normalize to 0-255
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cmap)
    cv2.imwrite(out_path, heat)

def process_directory(input_dir, output_dir, quality=90, scale=10):
    """
    Process all images in a directory and save their ELA versions.
    
    Args:
        input_dir (str): Directory containing original images
        output_dir (str): Directory to save ELA images
        quality (int): JPEG quality to use for ELA
        scale (int): Scale factor for ELA differences
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images in directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"ela_{filename}")
            
            try:
                # Compute ELA
                ela_image = compute_ela(input_path, quality, scale)
                
                # Save ELA result
                cv2.imwrite(output_path, cv2.cvtColor(ela_image, cv2.COLOR_RGB2BGR))
                print(f"Processed {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Example usage for CASIA2 dataset
    authentic_dir = "data/CASIA2/authentic"
    tampered_dir = "data/CASIA2/tampered"
    
    ela_authentic_dir = "data/ela/CASIA2/authentic"
    ela_tampered_dir = "data/ela/CASIA2/tampered"
    
    # Process both authentic and tampered images
    process_directory(authentic_dir, ela_authentic_dir)
    process_directory(tampered_dir, ela_tampered_dir)