"""
Convert DICOM images from TCGA-COAD to JPEG format for training.
Applies proper windowing and normalizes pixel values.
"""

import os
import pydicom
import numpy as np
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'dicom_conversion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class DicomConverter:
    def __init__(self, input_dir="data/TCGA-COAD", output_dir="data/processed/TCGA-COAD-JPEG"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metadata_file = self.input_dir / "metadata_combined.csv"
        self.conversion_info = []
        
    def apply_window(self, image, window_center, window_width):
        """
        Apply windowing to the image to make it visible.
        Handles float values properly.
        """
        img_min = float(window_center) - float(window_width) / 2.0
        img_max = float(window_center) + float(window_width) / 2.0
        image = np.clip(image, img_min, img_max)
        
        # Normalize to 0-255 with float calculations
        if img_max != img_min:
            image = ((image - img_min) / (img_max - img_min) * 255.0)
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def convert_dicom_to_jpeg(self, dicom_path, output_path):
        """
        Convert a single DICOM file to JPEG
        """
        try:
            # Read DICOM file
            ds = pydicom.dcmread(str(dicom_path))
            
            # Get pixel array and convert to float to handle negative values
            image = ds.pixel_array.astype(float)
            
            # Get DICOM window settings or use defaults
            if ds.Modality == "CT":
                # Try to get window settings from DICOM
                try:
                    window_center = getattr(ds, 'WindowCenter', 40)
                    window_width = getattr(ds, 'WindowWidth', 400)
                    
                    # Handle different window value formats
                    if hasattr(window_center, 'value'):
                        # Handle MultiValue
                        window_center = float(window_center.value[0])
                    elif isinstance(window_center, (list, tuple)):
                        window_center = float(window_center[0])
                    else:
                        window_center = float(window_center)
                        
                    if hasattr(window_width, 'value'):
                        # Handle MultiValue
                        window_width = float(window_width.value[0])
                    elif isinstance(window_width, (list, tuple)):
                        window_width = float(window_width[0])
                    else:
                        window_width = float(window_width)
                except:
                    # Default for soft tissue
                    window_center = 40
                    window_width = 400
                
                # Apply Hounsfield Unit scaling
                if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
                    rescale_intercept = ds.RescaleIntercept
                    rescale_slope = ds.RescaleSlope
                    
                    # Handle MultiValue attributes
                    if hasattr(rescale_intercept, 'value'):
                        rescale_intercept = float(rescale_intercept.value[0])
                    else:
                        rescale_intercept = float(rescale_intercept)
                        
                    if hasattr(rescale_slope, 'value'):
                        rescale_slope = float(rescale_slope.value[0])
                    else:
                        rescale_slope = float(rescale_slope)
                        
                    image = image * rescale_slope + rescale_intercept
                
                # Apply windowing
                image = self.apply_window(image, float(window_center), float(window_width))
            else:
                # For non-CT images, normalize to 0-255
                min_val = float(image.min())
                max_val = float(image.max())
                if max_val != min_val:
                    image = ((image - min_val) / (max_val - min_val) * 255)
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Validate image before saving
            if image.size == 0:
                raise ValueError("Empty image array")
            if np.isnan(image).any():
                raise ValueError("Image contains NaN values")
            if not np.isfinite(image).all():
                raise ValueError("Image contains infinite values")
                
            # Save as JPEG
            success = cv2.imwrite(str(output_path), image)
            if not success:
                raise IOError("Failed to save JPEG image")
            
            # Record conversion info
            info = {
                'original_path': str(dicom_path),
                'jpeg_path': str(output_path),
                'modality': getattr(ds, 'Modality', 'Unknown'),
                'study_description': getattr(ds, 'StudyDescription', 'Unknown'),
                'series_description': getattr(ds, 'SeriesDescription', 'Unknown'),
                'image_shape': image.shape,
                'window_center': window_center if ds.Modality == "CT" else None,
                'window_width': window_width if ds.Modality == "CT" else None
            }
            self.conversion_info.append(info)
            
            return True
            
        except Exception as e:
            logging.error(f"Error converting {dicom_path}: {str(e)}")
            return False
    
    def process_all(self):
        """
        Convert all DICOM files in the dataset
        """
        # Load metadata
        if self.metadata_file.exists():
            df = pd.read_csv(self.metadata_file)
            logging.info(f"Loaded metadata for {len(df)} files")
            
            # Process each file
            success = 0
            failed = 0
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting DICOM files"):
                dicom_path = self.input_dir / row['file_path']
                
                # Create corresponding output path
                relative_path = Path(row['file_path'])
                output_path = self.output_dir / relative_path.parent / f"{relative_path.stem}.jpg"
                
                if self.convert_dicom_to_jpeg(dicom_path, output_path):
                    success += 1
                else:
                    failed += 1
                
            # Save conversion info
            conversion_df = pd.DataFrame(self.conversion_info)
            conversion_df.to_csv(self.output_dir / "conversion_info.csv", index=False)
            
            logging.info(f"""
            Conversion completed:
            - Successfully converted: {success}
            - Failed conversions: {failed}
            - Success rate: {success/(success+failed)*100:.2f}%
            """)
        else:
            logging.error(f"Metadata file not found: {self.metadata_file}")

if __name__ == "__main__":
    # Initialize and run converter
    converter = DicomConverter()
    converter.process_all()