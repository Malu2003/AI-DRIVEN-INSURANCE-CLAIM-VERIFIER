#!/usr/bin/env python3
"""
Generate pHash database from authentic CASIA2 images.
This database is used for matching images against known authentic samples.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.phash import process_directory_to_csv

def main():
    """Generate pHash database from authentic images."""
    
    # Input: authentic images from CASIA2 dataset
    authentic_dir = PROJECT_ROOT / "data" / "CASIA2" / "authentic"
    
    # Output: pHash database CSV
    output_db = PROJECT_ROOT / "data" / "phash_casia_authentic.csv"
    
    print("="*80)
    print("PHASH DATABASE GENERATION")
    print("="*80)
    
    print(f"\nInput Directory:  {authentic_dir}")
    print(f"  Exists: {authentic_dir.exists()}")
    
    if not authentic_dir.exists():
        print(f"\n❌ ERROR: Authentic images directory not found!")
        print(f"   Expected: {authentic_dir}")
        return 1
    
    print(f"\nOutput Database:  {output_db}")
    print(f"  Parent exists: {output_db.parent.exists()}")
    
    # Count images first
    image_files = list(authentic_dir.glob("**/*.jpg")) + list(authentic_dir.glob("**/*.png"))
    print(f"\nImages found: {len(image_files)}")
    
    if len(image_files) == 0:
        print("\n❌ ERROR: No images found in authentic directory!")
        return 1
    
    print(f"\nProcessing {len(image_files)} authentic images...")
    print("This may take a few minutes...\n")
    
    try:
        process_directory_to_csv(str(authentic_dir), str(output_db))
        print(f"\n✓ pHash database generated successfully!")
        print(f"  Database: {output_db}")
        print(f"  File size: {output_db.stat().st_size} bytes")
        
        # Count entries
        with open(output_db, 'r') as f:
            lines = f.readlines()
            num_entries = len(lines) - 1  # Exclude header
        
        print(f"  Entries: {num_entries}")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. The pHash database is now ready")
        print("2. Test the image forgery detection:")
        print(f"   python test_tampered_image.py")
        print("3. Run the full pipeline with forensic analysis")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
