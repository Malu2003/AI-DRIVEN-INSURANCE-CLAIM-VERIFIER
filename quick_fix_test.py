"""Quick test of the fixed detection logic."""
from pipeline.image_module import run_image_forgery

# Test with actual test images
auth_img = 'batch_test_results/blur/colonca1000_blur.jpg'
tamp_img = 'batch_test_results/compression/colonca1000_compression.jpg'

print('Testing FIXED pipeline...')
print('='*70)

result1 = run_image_forgery(auth_img, model_ckpt='checkpoints/lc25000_forgery/best.pth.tar')
print(f'Blur image:        {result1.get("forgery_verdict"):15} (should be authentic)')

result2 = run_image_forgery(tamp_img, model_ckpt='checkpoints/lc25000_forgery/best.pth.tar')
print(f'Compression image: {result2.get("forgery_verdict"):15} (should be tampered)')

print('='*70)

if result1.get('forgery_verdict') in ['authentic', 'suspicious'] and result2.get('forgery_verdict') == 'tampered':
    print('✅ FIXED!')
else:
    print('❌ STILL BROKEN')
