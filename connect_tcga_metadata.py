"""
Connect TCGA-COAD DICOM metadata with image files and prepare for analysis.
Handles both .xlsx and .csv metadata files and validates DICOM headers.
"""

import os
import pandas as pd
import pydicom
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'tcga_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class TCGAProcessor:
    def __init__(self, base_dir="data/TCGA-COAD", metadata_file="data/metadata.csv"):
        self.base_dir = Path(base_dir)
        self.metadata_file = Path(metadata_file)
        self.output_file = self.base_dir / "metadata_combined.csv"
        
        # Essential DICOM tags we want to extract
        self.dicom_tags = [
            'PatientID', 'SeriesInstanceUID', 'StudyDescription',
            'Manufacturer', 'Modality', 'StudyDate', 'SeriesDescription',
            'ImageType', 'PixelSpacing', 'SliceThickness'
        ]

    def load_metadata(self):
        """Load and validate the metadata file"""
        try:
            if str(self.metadata_file).endswith('.xlsx'):
                meta = pd.read_excel(self.metadata_file)
            else:
                meta = pd.read_csv(self.metadata_file)
            
            logging.info(f"Loaded metadata: {len(meta)} entries")
            logging.info(f"Columns: {meta.columns.tolist()}")
            
            # Normalize Subject ID if present
            if 'Subject ID' in meta.columns:
                meta['Subject ID'] = meta['Subject ID'].astype(str).str.strip()
            
            return meta
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            return pd.DataFrame()

    def get_dicom_value(self, ds, tag, default="Unknown"):
        """Safely extract DICOM tag value"""
        try:
            return str(getattr(ds, tag, default)).strip()
        except:
            return default

    def scan_dicom_files(self):
        """Scan and extract information from DICOM files"""
        logging.info("Scanning DICOM files...")
        dicom_records = []
        total_files = 0
        error_files = 0

        for root, _, files in os.walk(self.base_dir):
            for f in files:
                if f.lower().endswith('.dcm'):
                    total_files += 1
                    dcm_path = Path(root) / f
                    try:
                        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                        
                        # Extract all specified DICOM tags
                        record = {
                            'file_path': str(dcm_path.relative_to(self.base_dir))
                        }
                        
                        for tag in self.dicom_tags:
                            record[tag.lower()] = self.get_dicom_value(ds, tag)
                        
                        # Add folder structure information
                        parts = dcm_path.relative_to(self.base_dir).parts
                        if len(parts) >= 2:
                            record['patient_folder'] = parts[0]
                            record['study_folder'] = parts[1]
                        
                        dicom_records.append(record)
                        
                        if total_files % 100 == 0:
                            logging.info(f"Processed {total_files} files...")
                            
                    except Exception as e:
                        error_files += 1
                        logging.warning(f"Error reading {dcm_path}: {e}")

        logging.info(f"Completed scanning: {total_files} total files, {error_files} errors")
        return pd.DataFrame(dicom_records)

    def merge_metadata(self, dicom_df, meta_df):
        """Merge DICOM information with metadata"""
        logging.info("Merging metadata with DICOM info...")
        
        # First try matching on Series UID
        if 'Series UID' in meta_df.columns:
            merged = pd.merge(
                dicom_df,
                meta_df,
                how="left",
                left_on="seriesinstanceuid",
                right_on="Series UID"
            )
        # Then try matching on Patient ID
        elif 'Subject ID' in meta_df.columns:
            merged = pd.merge(
                dicom_df,
                meta_df,
                how="left",
                left_on="patientid",
                right_on="Subject ID"
            )
        else:
            logging.warning("No matching columns found for merge, using DICOM data only")
            merged = dicom_df

        # Add match status column
        merged['metadata_matched'] = ~merged['Subject ID'].isna() if 'Subject ID' in merged.columns else False
        
        return merged

    def process(self):
        """Main processing pipeline"""
        # 1. Load metadata
        meta_df = self.load_metadata()
        if meta_df.empty:
            logging.error("Failed to load metadata, aborting")
            return
        
        # 2. Scan DICOM files
        dicom_df = self.scan_dicom_files()
        if dicom_df.empty:
            logging.error("No DICOM files found, aborting")
            return
        
        # 3. Merge data
        merged_df = self.merge_metadata(dicom_df, meta_df)
        
        # 4. Save results
        try:
            merged_df.to_csv(self.output_file, index=False)
            logging.info(f"Combined metadata saved to: {self.output_file}")
            
            # Print summary statistics
            logging.info("\nSummary:")
            logging.info(f"Total DICOM files: {len(dicom_df)}")
            logging.info(f"Total metadata entries: {len(meta_df)}")
            logging.info(f"Matched entries: {merged_df['metadata_matched'].sum()}")
            
            # Print modality distribution
            logging.info("\nModality distribution:")
            logging.info(merged_df['modality'].value_counts())
            
        except Exception as e:
            logging.error(f"Error saving results: {e}")

if __name__ == "__main__":
    # Initialize and run processor
    processor = TCGAProcessor()
    processor.process()