"""
Image Manipulation Testing Module
===================================

Create tampered/manipulated images to test the forgery detection system.
Supports various manipulation techniques commonly used in image forgeries.

Usage:
    python image_manipulation_tester.py --image path/to/image.jpg --output output/
    python image_manipulation_tester.py --batch data/LC25000/train/colon_aca/ --output test_tampered/
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import random
from typing import Tuple, Optional, List

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class ImageManipulator:
    """
    Apply various manipulation techniques to images for testing forgery detection.
    """

    def __init__(self, seed: int = 42):
        """Initialize manipulator with random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

    def copy_move_forgery(
        self, 
        image: Image.Image, 
        region_size: Tuple[int, int] = (100, 100),
        offset: Tuple[int, int] = (150, 150)
    ) -> Image.Image:
        """
        Copy-Move forgery: Copy a region and paste it elsewhere in the image.
        
        Args:
            image: Input PIL Image
            region_size: (width, height) of region to copy
            offset: (x, y) offset for pasted region
            
        Returns:
            Manipulated image with copy-move forgery
        """
        img = image.copy()
        width, height = img.size
        
        # Random source region
        x1 = random.randint(0, width - region_size[0] - offset[0])
        y1 = random.randint(0, height - region_size[1] - offset[1])
        x2 = x1 + region_size[0]
        y2 = y1 + region_size[1]
        
        # Copy region
        region = img.crop((x1, y1, x2, y2))
        
        # Paste at offset location
        paste_x = x1 + offset[0]
        paste_y = y1 + offset[1]
        img.paste(region, (paste_x, paste_y))
        
        return img

    def splicing_forgery(
        self, 
        base_image: Image.Image,
        splice_image: Optional[Image.Image] = None,
        position: Optional[Tuple[int, int]] = None,
        size: Tuple[int, int] = (150, 150)
    ) -> Image.Image:
        """
        Splicing forgery: Insert a region from another image.
        
        Args:
            base_image: Base image to manipulate
            splice_image: Image to splice from (if None, uses self-splicing)
            position: (x, y) position to insert splice
            size: Size of spliced region
            
        Returns:
            Manipulated image with splicing
        """
        img = base_image.copy()
        width, height = img.size
        
        # If no splice image provided, use self-splicing
        if splice_image is None:
            splice_image = img
        else:
            splice_image = splice_image.resize((width, height))
        
        # Random splice region
        sx = random.randint(0, splice_image.width - size[0])
        sy = random.randint(0, splice_image.height - size[1])
        splice_region = splice_image.crop((sx, sy, sx + size[0], sy + size[1]))
        
        # Random or specified paste position
        if position is None:
            px = random.randint(0, width - size[0])
            py = random.randint(0, height - size[1])
        else:
            px, py = position
        
        img.paste(splice_region, (px, py))
        
        return img

    def enhancement_forgery(
        self, 
        image: Image.Image,
        brightness: float = 1.5,
        contrast: float = 1.5,
        saturation: float = 1.3,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> Image.Image:
        """
        Enhancement forgery: Modify brightness/contrast/saturation in a region.
        
        Args:
            image: Input PIL Image
            brightness: Brightness enhancement factor (1.0 = original)
            contrast: Contrast enhancement factor
            saturation: Saturation enhancement factor
            region: (x1, y1, x2, y2) to enhance, None for random
            
        Returns:
            Image with enhanced region
        """
        img = image.copy()
        width, height = img.size
        
        # Define region
        if region is None:
            x1 = random.randint(0, width // 2)
            y1 = random.randint(0, height // 2)
            x2 = random.randint(x1 + width // 4, width)
            y2 = random.randint(y1 + height // 4, height)
            region = (x1, y1, x2, y2)
        
        # Extract and enhance region
        region_img = img.crop(region)
        
        # Apply enhancements
        enhancer = ImageEnhance.Brightness(region_img)
        region_img = enhancer.enhance(brightness)
        
        enhancer = ImageEnhance.Contrast(region_img)
        region_img = enhancer.enhance(contrast)
        
        if region_img.mode in ('RGB', 'RGBA'):
            enhancer = ImageEnhance.Color(region_img)
            region_img = enhancer.enhance(saturation)
        
        # Paste back
        img.paste(region_img, (region[0], region[1]))
        
        return img

    def blur_forgery(
        self,
        image: Image.Image,
        radius: int = 5,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> Image.Image:
        """
        Blur forgery: Apply blur to hide details in a region.
        
        Args:
            image: Input PIL Image
            radius: Blur radius
            region: (x1, y1, x2, y2) to blur, None for random
            
        Returns:
            Image with blurred region
        """
        img = image.copy()
        width, height = img.size
        
        # Define region
        if region is None:
            x1 = random.randint(0, width // 2)
            y1 = random.randint(0, height // 2)
            x2 = random.randint(x1 + width // 4, width)
            y2 = random.randint(y1 + height // 4, height)
            region = (x1, y1, x2, y2)
        
        # Extract and blur region
        region_img = img.crop(region)
        region_img = region_img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Paste back
        img.paste(region_img, (region[0], region[1]))
        
        return img

    def noise_addition(
        self,
        image: Image.Image,
        noise_level: float = 0.1,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> Image.Image:
        """
        Add random noise to image or region.
        
        Args:
            image: Input PIL Image
            noise_level: Noise intensity (0-1)
            region: (x1, y1, x2, y2) for noise, None for whole image
            
        Returns:
            Image with added noise
        """
        img = image.copy()
        img_array = np.array(img).astype(np.float32)
        
        if region is None:
            region = (0, 0, img.width, img.height)
        
        # Generate noise for region
        x1, y1, x2, y2 = region
        noise = np.random.normal(0, noise_level * 255, 
                                 (y2 - y1, x2 - x1, img_array.shape[2]))
        
        # Add noise to region
        img_array[y1:y2, x1:x2] += noise
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)

    def jpeg_compression_artifacts(
        self,
        image: Image.Image,
        quality: int = 50,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> Image.Image:
        """
        Add JPEG compression artifacts to hide forgery.
        
        Args:
            image: Input PIL Image
            quality: JPEG quality (1-100, lower = more artifacts)
            region: Region to compress, None for whole image
            
        Returns:
            Image with compression artifacts
        """
        import io
        
        if region is None:
            # Compress whole image
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            return Image.open(buffer)
        else:
            # Compress only region
            img = image.copy()
            region_img = img.crop(region)
            
            buffer = io.BytesIO()
            region_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            region_img = Image.open(buffer)
            
            img.paste(region_img, (region[0], region[1]))
            return img

    def remove_object(
        self,
        image: Image.Image,
        region: Optional[Tuple[int, int, int, int]] = None,
        fill_color: Optional[Tuple[int, int, int]] = None
    ) -> Image.Image:
        """
        Remove object by filling region (simple inpainting).
        
        Args:
            image: Input PIL Image
            region: (x1, y1, x2, y2) to remove, None for random
            fill_color: Color to fill with, None for average of surroundings
            
        Returns:
            Image with removed object
        """
        img = image.copy()
        width, height = img.size
        
        # Define region
        if region is None:
            size = min(width, height) // 4
            x1 = random.randint(0, width - size)
            y1 = random.randint(0, height - size)
            x2 = x1 + size
            y2 = y1 + size
            region = (x1, y1, x2, y2)
        
        # Calculate fill color if not provided
        if fill_color is None:
            img_array = np.array(img)
            x1, y1, x2, y2 = region
            
            # Get pixels around the region
            margin = 5
            x1_m = max(0, x1 - margin)
            y1_m = max(0, y1 - margin)
            x2_m = min(width, x2 + margin)
            y2_m = min(height, y2 + margin)
            
            surrounding = img_array[y1_m:y2_m, x1_m:x2_m]
            fill_color = tuple(np.mean(surrounding, axis=(0, 1)).astype(int))
        
        # Fill region
        draw = ImageDraw.Draw(img)
        draw.rectangle(region, fill=fill_color)
        
        return img

    def apply_random_manipulation(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """
        Apply a random manipulation technique.
        
        Returns:
            Tuple of (manipulated_image, technique_name)
        """
        techniques = {
            'copy_move': lambda img: self.copy_move_forgery(img),
            'splicing': lambda img: self.splicing_forgery(img),
            'enhancement': lambda img: self.enhancement_forgery(img),
            'blur': lambda img: self.blur_forgery(img),
            'noise': lambda img: self.noise_addition(img),
            'compression': lambda img: self.jpeg_compression_artifacts(img, quality=random.randint(30, 70)),
            'removal': lambda img: self.remove_object(img),
        }
        
        technique = random.choice(list(techniques.keys()))
        manipulated = techniques[technique](image)
        
        return manipulated, technique

    def apply_multiple_manipulations(
        self,
        image: Image.Image,
        num_manipulations: int = 2
    ) -> Tuple[Image.Image, List[str]]:
        """
        Apply multiple manipulation techniques sequentially.
        
        Returns:
            Tuple of (manipulated_image, list_of_techniques)
        """
        img = image.copy()
        techniques = []
        
        for _ in range(num_manipulations):
            img, technique = self.apply_random_manipulation(img)
            techniques.append(technique)
        
        return img, techniques


def process_single_image(
    input_path: str,
    output_dir: str,
    manipulator: ImageManipulator,
    technique: Optional[str] = None
):
    """Process a single image with specified or random manipulation."""
    
    # Load image
    try:
        img = Image.open(input_path).convert('RGB')
    except Exception as e:
        print(f"❌ Failed to load {input_path}: {e}")
        return
    
    filename = Path(input_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Apply manipulation
    if technique == 'all':
        # Apply all techniques
        techniques = {
            'copy_move': manipulator.copy_move_forgery,
            'splicing': manipulator.splicing_forgery,
            'enhancement': manipulator.enhancement_forgery,
            'blur': manipulator.blur_forgery,
            'noise': manipulator.noise_addition,
            'compression': lambda x: manipulator.jpeg_compression_artifacts(x, quality=50),
            'removal': manipulator.remove_object,
        }
        
        for tech_name, tech_func in techniques.items():
            try:
                manipulated = tech_func(img)
                output_file = output_path / f"{filename}_{tech_name}.jpg"
                manipulated.save(output_file, quality=95)
                print(f"✅ Created: {output_file.name} (technique: {tech_name})")
            except Exception as e:
                print(f"❌ Failed {tech_name} on {filename}: {e}")
    
    elif technique == 'random':
        # Apply random manipulation
        manipulated, tech_name = manipulator.apply_random_manipulation(img)
        output_file = output_path / f"{filename}_random_{tech_name}.jpg"
        manipulated.save(output_file, quality=95)
        print(f"✅ Created: {output_file.name} (technique: {tech_name})")
    
    elif technique == 'multi':
        # Apply multiple random manipulations
        manipulated, techniques = manipulator.apply_multiple_manipulations(img, num_manipulations=2)
        tech_str = '_'.join(techniques)
        output_file = output_path / f"{filename}_multi_{tech_str}.jpg"
        manipulated.save(output_file, quality=95)
        print(f"✅ Created: {output_file.name} (techniques: {techniques})")
    
    else:
        # Apply specific technique
        technique_map = {
            'copy_move': manipulator.copy_move_forgery,
            'splicing': manipulator.splicing_forgery,
            'enhancement': manipulator.enhancement_forgery,
            'blur': manipulator.blur_forgery,
            'noise': manipulator.noise_addition,
            'compression': lambda x: manipulator.jpeg_compression_artifacts(x, quality=50),
            'removal': manipulator.remove_object,
        }
        
        if technique in technique_map:
            manipulated = technique_map[technique](img)
            output_file = output_path / f"{filename}_{technique}.jpg"
            manipulated.save(output_file, quality=95)
            print(f"✅ Created: {output_file.name}")
        else:
            print(f"❌ Unknown technique: {technique}")


def process_batch(
    input_dir: str,
    output_dir: str,
    manipulator: ImageManipulator,
    technique: str = 'random',
    max_images: int = 10
):
    """Process multiple images from a directory."""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Find all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(list(input_path.glob(ext)))
    
    if not images:
        print(f"❌ No images found in {input_dir}")
        return
    
    # Limit number of images
    images = images[:max_images]
    
    print(f"\n{'='*70}")
    print(f"Processing {len(images)} images from {input_dir}")
    print(f"Technique: {technique}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    for img_path in images:
        process_single_image(str(img_path), output_dir, manipulator, technique)
    
    print(f"\n✅ Batch processing complete!")
    print(f"   Processed: {len(images)} images")
    print(f"   Output directory: {output_dir}")


def test_with_detection(manipulated_image_path: str):
    """Test manipulated image with forgery detection."""
    try:
        from inference.image_forgery_score import compute_cnn_score
        import torch
        
        checkpoint = str(PROJECT_ROOT / "checkpoints" / "lc25000_forgery" / "best.pth.tar")
        
        if not Path(checkpoint).exists():
            print("⚠️  Detection model not found, skipping detection test")
            return
        
        score = compute_cnn_score(
            manipulated_image_path,
            model_ckpt=checkpoint,
            tampered_index=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"\n{'='*70}")
        print("FORGERY DETECTION TEST")
        print(f"{'='*70}")
        print(f"Image: {Path(manipulated_image_path).name}")
        print(f"CNN Score: {score:.4f}")
        print(f"Verdict: {'TAMPERED' if score > 0.5 else 'AUTHENTIC'}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"⚠️  Could not run detection: {e}")


def main():
    parser = argparse.ArgumentParser(description='Image Manipulation Testing Tool')
    
    # Input/Output
    parser.add_argument('--image', type=str, help='Single image to manipulate')
    parser.add_argument('--batch', type=str, help='Directory of images to process')
    parser.add_argument('--output', type=str, default='test_manipulated', 
                       help='Output directory for manipulated images')
    
    # Manipulation options
    parser.add_argument('--technique', type=str, default='random',
                       choices=['copy_move', 'splicing', 'enhancement', 'blur', 
                               'noise', 'compression', 'removal', 'random', 'multi', 'all'],
                       help='Manipulation technique to apply')
    parser.add_argument('--max-images', type=int, default=10,
                       help='Maximum images to process in batch mode')
    
    # Testing options
    parser.add_argument('--test-detection', action='store_true',
                       help='Test manipulated images with forgery detector')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize manipulator
    manipulator = ImageManipulator(seed=args.seed)
    
    print("\n" + "="*70)
    print("IMAGE MANIPULATION TESTING TOOL")
    print("="*70)
    
    # Process images
    if args.image:
        # Single image mode
        print(f"\nProcessing single image: {args.image}")
        process_single_image(args.image, args.output, manipulator, args.technique)
        
        # Test with detector if requested
        if args.test_detection:
            output_files = list(Path(args.output).glob("*.jpg"))
            if output_files:
                test_with_detection(str(output_files[0]))
    
    elif args.batch:
        # Batch mode
        process_batch(args.batch, args.output, manipulator, args.technique, args.max_images)
        
        # Test with detector if requested
        if args.test_detection:
            output_files = list(Path(args.output).glob("*.jpg"))
            if output_files:
                print(f"\n{'='*70}")
                print("TESTING MANIPULATED IMAGES WITH DETECTOR")
                print(f"{'='*70}\n")
                for img_file in output_files[:5]:  # Test first 5
                    test_with_detection(str(img_file))
    
    else:
        print("❌ Please specify --image or --batch")
        parser.print_help()
        return
    
    print("\n" + "="*70)
    print("✅ MANIPULATION TESTING COMPLETE")
    print("="*70)
    print(f"\nManipulated images saved to: {args.output}")
    print("\nTo test with forgery detector:")
    print(f"  python image_manipulation_tester.py --image {args.output}/image.jpg --test-detection")
    print("\nTo test in pipeline:")
    print(f"  Use manipulated images from {args.output}/ in your frontend testing")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
