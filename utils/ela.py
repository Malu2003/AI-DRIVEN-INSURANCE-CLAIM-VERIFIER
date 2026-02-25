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


def compute_ela_score(diff_image, suspicious_area_pct=None):
    """Compute a normalized ELA score in [0,1] from the ELA diff image.

    If suspicious_area_pct is provided (from localization), uses that as the primary metric.
    Otherwise, uses mean intensity of the ELA image normalized by 255.
    Higher values indicate larger compression artifacts (more suspicious).
    
    Args:
        diff_image: numpy array of ELA difference image
        suspicious_area_pct: float [0-100], percentage of image flagged as anomalous.
                            If provided, this is weighted 70% of final score.
    
    Returns:
        float [0-1]: Fraud-suspicious score
    """
    if isinstance(diff_image, np.ndarray):
        # Compute mean intensity baseline
        if diff_image.ndim == 3:
            gray = np.mean(diff_image, axis=2)
        else:
            gray = diff_image
        mean_val = float(np.mean(gray))
        mean_score = np.clip(mean_val / 255.0, 0.0, 1.0)
        
        # If we have localization metrics, use them as primary signal
        if suspicious_area_pct is not None:
            # Convert percentage (0-100) to score (0-1)
            # Threshold: 2% anomalous area = moderate risk (0.5)
            # Threshold: 5% anomalous area = high risk (0.7)
            # Threshold: 10%+ anomalous area = critical (0.9+)
            area_score = np.clip(suspicious_area_pct / 15.0, 0.0, 1.0)
            
            # Combine: 70% localization, 30% mean intensity
            final_score = 0.7 * area_score + 0.3 * mean_score
            return np.clip(final_score, 0.0, 1.0)
        else:
            # Fallback: just mean intensity (original behavior)
            return mean_score
    else:
        raise ValueError('diff_image must be a numpy array')


def save_ela_heatmap(diff_image, out_path, cmap=cv2.COLORMAP_JET):
    """Save ELA diff image as a heatmap to `out_path`.

    Uses OpenCV colormap for a compact dependency set.
    """
    heat = _build_enhanced_ela_heatmap(diff_image, cmap=cmap)
    cv2.imwrite(out_path, heat)


def save_ela_visualization(
    diff_image,
    out_path,
    scale=10,
    cmap=cv2.COLORMAP_TURBO,
    anomaly_percentile=96,
    gamma=0.75,
    overlay_strength=0.65,
):
    """Save an enhanced ELA visualization for stronger forgery highlighting.

    This function is intentionally compatible with existing callers that expect
    `save_ela_visualization(diff, out_path, scale=...)`.
    """
    _ = scale
    heat = _build_enhanced_ela_heatmap(
        diff_image,
        cmap=cmap,
        anomaly_percentile=anomaly_percentile,
        gamma=gamma,
        overlay_strength=overlay_strength,
    )
    cv2.imwrite(out_path, heat)


def _build_enhanced_ela_heatmap(
    diff_image,
    cmap=cv2.COLORMAP_TURBO,
    anomaly_percentile=96,
    gamma=0.75,
    overlay_strength=0.65,
):
    """Generate a high-contrast ELA heatmap with anomaly-mask enhancement."""
    if diff_image.ndim == 3:
        gray = np.mean(diff_image, axis=2).astype(np.float32)
    else:
        gray = diff_image.astype(np.float32)

    # Robust normalization using high-percentile clipping so subtle artifacts pop.
    p_high = np.percentile(gray, 99)
    if p_high <= 0:
        p_high = 1.0
    clipped = np.clip(gray, 0, p_high) / p_high

    # Gamma shaping boosts low-mid anomalies while preserving hotspots.
    gamma = max(0.2, float(gamma))
    boosted = np.power(clipped, gamma)

    # Local contrast enhancement via CLAHE.
    boosted_u8 = np.clip(boosted * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(boosted_u8)

    # Build color heatmap.
    heat = cv2.applyColorMap(enhanced, cmap)

    # Highlight top anomaly regions explicitly in red.
    threshold = np.percentile(enhanced, max(50, min(99.9, anomaly_percentile)))
    anomaly_mask = enhanced >= threshold
    red_overlay = np.zeros_like(heat)
    red_overlay[:, :, 2] = 255

    alpha = np.float32(np.clip(overlay_strength, 0.0, 1.0))
    heat = heat.astype(np.float32)
    heat[anomaly_mask] = (1.0 - alpha) * heat[anomaly_mask] + alpha * red_overlay[anomaly_mask]

    return np.clip(heat, 0, 255).astype(np.uint8)


def save_ela_localization_overlay(
    original_image_path,
    diff_image,
    out_path,
    anomaly_percentile=96,
    min_area_ratio=0.0008,
    max_regions=5,
):
    """Save an ELA localization overlay with boxed suspicious regions.

    Returns:
        dict: {
            suspicious_area_pct: float,
            regions: [{x,y,w,h,area_pct}],
            summary: str,
        }
    """
    original_bgr = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise ValueError(f"Could not read original image: {original_image_path}")

    if diff_image.ndim == 3:
        gray = np.mean(diff_image, axis=2).astype(np.float32)
    else:
        gray = diff_image.astype(np.float32)

    # Normalize anomaly map robustly.
    p_high = np.percentile(gray, 99)
    if p_high <= 0:
        p_high = 1.0
    norm = np.clip(gray / p_high, 0.0, 1.0)
    norm_u8 = (norm * 255).astype(np.uint8)

    # Binary mask from top anomalies.
    threshold = np.percentile(norm_u8, max(50, min(99.9, anomaly_percentile)))
    _, binary = cv2.threshold(norm_u8, int(threshold), 255, cv2.THRESH_BINARY)

    # Morphological cleanup.
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)

    height, width = binary.shape
    total_pixels = float(height * width)
    min_area = max(20, int(total_pixels * float(min_area_ratio)))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    overlay = original_bgr.copy()
    regions = []
    mask_pixels = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        area_pct = (area / total_pixels) * 100.0
        regions.append({
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'area_pct': float(round(area_pct, 3)),
        })
        mask_pixels += int(area)

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"ELA region {len(regions)} ({area_pct:.2f}%)"
        text_y = y - 8 if y - 8 > 10 else y + 18
        cv2.putText(overlay, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        if len(regions) >= int(max_regions):
            break

    # Blend transparent anomaly mask for visibility.
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 2] = binary
    overlay = cv2.addWeighted(overlay, 1.0, red_layer, 0.25, 0)

    suspicious_area_pct = (mask_pixels / total_pixels) * 100.0
    if regions:
        top = regions[0]
        summary = (
            f"Likely tampered area near (x={top['x']}, y={top['y']}, "
            f"w={top['w']}, h={top['h']}); suspicious area ≈ {suspicious_area_pct:.2f}%"
        )
    else:
        summary = "No strong localized ELA anomaly cluster detected"

    cv2.imwrite(out_path, overlay)
    return {
        'suspicious_area_pct': float(round(suspicious_area_pct, 3)),
        'regions': regions,
        'summary': summary,
    }

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