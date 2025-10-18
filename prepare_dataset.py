"""
Prepare combined dataset from JPEG images and metadata.
Creates a new CSV file with paths to JPEG images and their corresponding metadata.
"""

import pandas as pd
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'dataset_preparation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class DatasetPreparer:
    def __init__(self):
        self.base_dir = Path("data")
        self.jpeg_dir = self.base_dir / "processed" / "TCGA-COAD-JPEG"
        self.metadata_file = self.base_dir / "TCGA-COAD" / "metadata_combined.csv"
        self.output_dir = self.base_dir / "processed" / "combined_dataset"
        self.output_csv = self.output_dir / "dataset_info.csv"
        
    def get_jpeg_path(self, dicom_path):
        """Convert DICOM path to corresponding JPEG path"""
        return str(self.jpeg_dir / Path(dicom_path).with_suffix('.jpg'))
    
    def prepare_dataset(self):
        """
        Prepare the combined dataset:
        1. Read metadata
        2. Add JPEG paths
        3. Verify JPEG files exist
        4. Create organized dataset structure
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read metadata
        logging.info("Reading metadata...")
        df = pd.read_csv(self.metadata_file)
        
        # Add JPEG paths
        df['jpeg_path'] = df['file_path'].apply(self.get_jpeg_path)
        
        # Verify JPEG files exist
        logging.info("Verifying JPEG files...")
        df['jpeg_exists'] = df['jpeg_path'].apply(os.path.exists)
        
        # Filter out missing JPEGs
        missing_jpegs = df[~df['jpeg_exists']]
        if not missing_jpegs.empty:
            logging.warning(f"Found {len(missing_jpegs)} missing JPEG files")
            logging.warning("First few missing files:")
            for path in missing_jpegs['jpeg_path'].head():
                logging.warning(f"  {path}")
        
        # Keep only records with existing JPEGs
        df = df[df['jpeg_exists']]
        
        # Add key image characteristics
        def safe_eval(x):
            """Safely evaluate string representations of lists"""
            if not isinstance(x, str):
                return x
            if x.lower() == 'unknown':
                return None
            try:
                return eval(x)
            except:
                return None
        
        df['image_type'] = df['imagetype'].apply(safe_eval)
        df['is_localizer'] = df['image_type'].apply(lambda x: 'LOCALIZER' in x if isinstance(x, list) else False)
        df['pixel_spacing'] = df['pixelspacing'].apply(safe_eval)
        
        # Save organized dataset info
        logging.info("Saving dataset info...")
        df.to_csv(self.output_csv, index=False)
        
        logging.info(f"""
        Dataset preparation completed:
        - Total images: {len(df)}
        - Localizer images: {df['is_localizer'].sum()}
        - Non-localizer images: {(~df['is_localizer']).sum()}
        - Output CSV: {self.output_csv}
        """)
        
        return df

if __name__ == "__main__":
    preparer = DatasetPreparer()
    dataset = preparer.prepare_dataset()