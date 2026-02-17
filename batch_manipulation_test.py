"""
Batch Image Manipulation and Detection Test
============================================

Create manipulated images and test them with the forgery detector.
Generates a comprehensive report with detection accuracy.

Usage:
    python batch_manipulation_test.py --source data/LC25000/train/colon_aca/ --num-images 10
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List, Dict
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from image_manipulation_tester import ImageManipulator


def run_batch_test(
    source_dir: str,
    num_images: int = 10,
    output_base: str = "batch_test_results",
    test_detection: bool = True
):
    """
    Run comprehensive batch test:
    1. Create manipulated images with all techniques
    2. Test each with forgery detector
    3. Generate report
    """
    
    print("\n" + "="*70)
    print("BATCH MANIPULATION AND DETECTION TEST")
    print("="*70)
    
    # Setup paths
    source_path = Path(source_dir)
    output_path = Path(output_base)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find source images
    extensions = ['*.jpg', '*.jpeg', '*.png']
    source_images = []
    for ext in extensions:
        source_images.extend(list(source_path.glob(ext)))
    
    if not source_images:
        print(f"❌ No images found in {source_dir}")
        return
    
    source_images = source_images[:num_images]
    
    print(f"\nTest Configuration:")
    print(f"  Source: {source_dir}")
    print(f"  Images: {len(source_images)}")
    print(f"  Output: {output_base}")
    print(f"  Detection: {'Enabled' if test_detection else 'Disabled'}")
    
    # Initialize manipulator
    manipulator = ImageManipulator(seed=42)
    
    # Techniques to test
    techniques = [
        'copy_move',
        'splicing',
        'enhancement',
        'blur',
        'noise',
        'compression',
        'removal',
    ]
    
    # Results storage
    results = {
        'test_date': datetime.now().isoformat(),
        'source_directory': str(source_dir),
        'num_images': len(source_images),
        'authentic_scores': [],
        'manipulated_scores': {},
        'detection_enabled': test_detection,
    }
    
    for tech in techniques:
        results['manipulated_scores'][tech] = []
    
    # Process each image
    print(f"\n{'='*70}")
    print("PROCESSING IMAGES")
    print(f"{'='*70}\n")
    
    for idx, img_path in enumerate(source_images, 1):
        print(f"[{idx}/{len(source_images)}] Processing: {img_path.name}")
        
        try:
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            
            # Test authentic image
            if test_detection:
                score = test_image_detection(str(img_path))
                results['authentic_scores'].append({
                    'image': img_path.name,
                    'score': score
                })
                print(f"  Authentic score: {score:.4f}")
            
            # Test each manipulation
            for tech in techniques:
                # Apply manipulation
                if tech == 'copy_move':
                    manipulated = manipulator.copy_move_forgery(img)
                elif tech == 'splicing':
                    manipulated = manipulator.splicing_forgery(img)
                elif tech == 'enhancement':
                    manipulated = manipulator.enhancement_forgery(img)
                elif tech == 'blur':
                    manipulated = manipulator.blur_forgery(img)
                elif tech == 'noise':
                    manipulated = manipulator.noise_addition(img)
                elif tech == 'compression':
                    manipulated = manipulator.jpeg_compression_artifacts(img, quality=50)
                elif tech == 'removal':
                    manipulated = manipulator.remove_object(img)
                
                # Save manipulated image
                tech_dir = output_path / tech
                tech_dir.mkdir(exist_ok=True)
                output_file = tech_dir / f"{img_path.stem}_{tech}.jpg"
                manipulated.save(output_file, quality=95)
                
                # Test with detector
                if test_detection:
                    score = test_image_detection(str(output_file))
                    results['manipulated_scores'][tech].append({
                        'image': output_file.name,
                        'score': score
                    })
                    print(f"  {tech:15s} score: {score:.4f}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Generate report
    generate_report(results, output_path)
    
    print(f"\n{'='*70}")
    print("✅ BATCH TEST COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_path}")
    print(f"Report: {output_path / 'test_report.txt'}")
    print(f"JSON: {output_path / 'test_results.json'}")
    print(f"{'='*70}\n")


def test_image_detection(image_path: str) -> float:
    """Test image with forgery detector and return score."""
    try:
        from inference.image_forgery_score import compute_cnn_score
        import torch
        
        checkpoint = str(PROJECT_ROOT / "checkpoints" / "lc25000_forgery" / "best.pth.tar")
        
        if not Path(checkpoint).exists():
            return -1.0  # Model not available
        
        score = compute_cnn_score(
            image_path,
            model_ckpt=checkpoint,
            tampered_index=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        return float(score)
    
    except Exception as e:
        print(f"    ⚠️ Detection failed: {e}")
        return -1.0


def generate_report(results: Dict, output_path: Path):
    """Generate comprehensive test report."""
    
    # Calculate statistics
    if results['detection_enabled']:
        # Authentic statistics
        auth_scores = [r['score'] for r in results['authentic_scores'] if r['score'] >= 0]
        auth_avg = sum(auth_scores) / len(auth_scores) if auth_scores else 0
        auth_min = min(auth_scores) if auth_scores else 0
        auth_max = max(auth_scores) if auth_scores else 0
        
        # Manipulated statistics per technique
        tech_stats = {}
        for tech, scores in results['manipulated_scores'].items():
            score_values = [r['score'] for r in scores if r['score'] >= 0]
            if score_values:
                tech_stats[tech] = {
                    'avg': sum(score_values) / len(score_values),
                    'min': min(score_values),
                    'max': max(score_values),
                    'count': len(score_values),
                }
        
        # Generate text report
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("FORGERY DETECTION BATCH TEST REPORT")
        report_lines.append("="*70)
        report_lines.append(f"\nTest Date: {results['test_date']}")
        report_lines.append(f"Source Directory: {results['source_directory']}")
        report_lines.append(f"Number of Images: {results['num_images']}")
        report_lines.append("\n" + "="*70)
        report_lines.append("AUTHENTIC IMAGES")
        report_lines.append("="*70)
        report_lines.append(f"Average Score: {auth_avg:.4f}")
        report_lines.append(f"Min Score:     {auth_min:.4f}")
        report_lines.append(f"Max Score:     {auth_max:.4f}")
        report_lines.append(f"Count:         {len(auth_scores)}")
        
        report_lines.append("\n" + "="*70)
        report_lines.append("MANIPULATED IMAGES BY TECHNIQUE")
        report_lines.append("="*70)
        
        for tech, stats in sorted(tech_stats.items()):
            report_lines.append(f"\n{tech.upper()}:")
            report_lines.append(f"  Average Score: {stats['avg']:.4f}")
            report_lines.append(f"  Min Score:     {stats['min']:.4f}")
            report_lines.append(f"  Max Score:     {stats['max']:.4f}")
            report_lines.append(f"  Count:         {stats['count']}")
        
        report_lines.append("\n" + "="*70)
        report_lines.append("DETECTION ACCURACY ANALYSIS")
        report_lines.append("="*70)
        
        # Threshold analysis
        threshold = 0.5
        auth_correct = sum(1 for s in auth_scores if s < threshold)
        auth_accuracy = (auth_correct / len(auth_scores) * 100) if auth_scores else 0
        
        report_lines.append(f"\nThreshold: {threshold}")
        report_lines.append(f"\nAuthentic Images:")
        report_lines.append(f"  Correctly classified: {auth_correct}/{len(auth_scores)} ({auth_accuracy:.1f}%)")
        
        report_lines.append(f"\nManipulated Images:")
        for tech, stats in sorted(tech_stats.items()):
            tech_scores = [r['score'] for r in results['manipulated_scores'][tech] if r['score'] >= 0]
            tech_correct = sum(1 for s in tech_scores if s >= threshold)
            tech_accuracy = (tech_correct / len(tech_scores) * 100) if tech_scores else 0
            report_lines.append(f"  {tech:15s}: {tech_correct}/{len(tech_scores)} ({tech_accuracy:.1f}%)")
        
        # Overall accuracy
        total_correct = auth_correct
        for tech in tech_stats:
            tech_scores = [r['score'] for r in results['manipulated_scores'][tech] if r['score'] >= 0]
            total_correct += sum(1 for s in tech_scores if s >= threshold)
        
        total_images = len(auth_scores) + sum(len([r for r in results['manipulated_scores'][tech] if r['score'] >= 0]) for tech in tech_stats)
        overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
        
        report_lines.append(f"\nOverall Accuracy: {total_correct}/{total_images} ({overall_accuracy:.1f}%)")
        
        report_lines.append("\n" + "="*70)
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("="*70)
        
        if auth_accuracy < 90:
            report_lines.append("\n⚠️  LOW AUTHENTIC ACCURACY:")
            report_lines.append("   - Model may be over-sensitive")
            report_lines.append("   - Consider adjusting threshold")
        
        low_detection_techs = [tech for tech, stats in tech_stats.items() 
                               if stats['avg'] < 0.5]
        if low_detection_techs:
            report_lines.append(f"\n⚠️  LOW DETECTION FOR: {', '.join(low_detection_techs)}")
            report_lines.append("   - These manipulations are harder to detect")
            report_lines.append("   - Consider additional training or preprocessing")
        
        if overall_accuracy >= 90:
            report_lines.append("\n✅ EXCELLENT OVERALL PERFORMANCE!")
        elif overall_accuracy >= 75:
            report_lines.append("\n✅ GOOD OVERALL PERFORMANCE")
        else:
            report_lines.append("\n⚠️  NEEDS IMPROVEMENT")
        
        report_lines.append("\n" + "="*70)
        
        # Save text report
        report_text = '\n'.join(report_lines)
        with open(output_path / 'test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n" + report_text)
    
    # Save JSON results
    with open(output_path / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Batch Image Manipulation and Detection Test')
    
    parser.add_argument('--source', type=str, required=True,
                       help='Source directory with authentic images')
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of images to process (default: 10)')
    parser.add_argument('--output', type=str, default='batch_test_results',
                       help='Output directory for results (default: batch_test_results)')
    parser.add_argument('--no-detection', action='store_true',
                       help='Skip detection testing (only create manipulated images)')
    
    args = parser.parse_args()
    
    run_batch_test(
        source_dir=args.source,
        num_images=args.num_images,
        output_base=args.output,
        test_detection=not args.no_detection
    )


if __name__ == "__main__":
    main()
