"""Inference script to compute ELA, pHash, CNN scores and fuse them into a final image score.

Usage (example):
    python inference/image_forgery_score.py --image data/LC25000/train/colon_aca/colonca1.jpeg \
        --model checkpoints/lc25000_forgery/best.pth.tar --phash_db data/phash_casia_authentic.csv --out_heatmap out/ela_heat.jpg
"""
import os
import argparse
import json
import numpy as np
from PIL import Image
# Defer heavy ML imports (torch, torchvision) to runtime so demo mode can run even
# in lightweight environments without full ML stack.
try:
    import torch
except Exception:
    torch = None


# Ensure project root is on sys.path so utils can be imported when running as a script
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import ela as ela_utils
from utils import phash as phash_utils
# Use pHash DB similarity for historical fraud checking (STEP 3 integration)
try:
    from icd_validation import phash_db
except Exception:
    phash_db = None


def compute_cnn_score(image_path, model_ckpt=None, tampered_index=1, device='cpu'):
    if model_ckpt is None or not os.path.exists(model_ckpt):
        return None
    
    # Import ML stack lazily so scripts can still run in environments
    # without torch if CNN scores are not required (demo/CPU-only).
    if torch is None:
        raise RuntimeError('PyTorch not installed; install torch to enable CNN scoring')
    from torchvision import transforms, models
    import torch.nn as nn
    
    # Cache models per checkpoint to avoid rebuilding for every image
    if not hasattr(compute_cnn_score, '_model_cache'):
        compute_cnn_score._model_cache = {}
    cache = compute_cnn_score._model_cache
    cache_key = f"{os.path.abspath(model_ckpt)}::{device}"
    if cache_key in cache:
        model = cache[cache_key]
    else:
        # Build DenseNet with num_classes=2 (for CASIA-style forgery detection)
        model = models.densenet121(pretrained=False)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 2)
        ckpt = torch.load(model_ckpt, map_location=device)
        state = ckpt.get('state_dict', ckpt)
        model_state = model.state_dict()
        matched = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
        model_state.update(matched)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        cache[cache_key] = model

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    # clamp index
    if tampered_index < 0 or tampered_index >= len(probs):
        raise ValueError('tampered_index out of range for model outputs')
    return float(probs[tampered_index])


def fuse_scores(cnn_score, ela_score, phash_score, weights=(0.55, 0.25, 0.20)):
    # Missing scores treated as 0.5 (neutral) so they don't dominate
    # Note: All scores follow same convention - HIGH = suspicious
    # CNN high = tampering, ELA high = anomalies, pHash high = dissimilar to authentic
    s_cnn = cnn_score if cnn_score is not None else 0.5
    s_ela = ela_score if ela_score is not None else 0.5
    s_ph = phash_score if phash_score is not None else 0.5

    final = weights[0] * s_cnn + weights[1] * s_ela + weights[2] * s_ph
    return final


def _process_image(args, image_path):
    """Process a single image and return the inference dictionary.

    This function performs the unchanged ELA and CNN computations and adds
    pHash similarity checks using `phash_db.find_similar_phash()` when
    available. It returns an `out` dict suitable for printing or saving.
    """
    out = {}
    # ELA (unchanged)
    diff = ela_utils.compute_ela(image_path, quality=args.ela_quality, scale=args.ela_scale)
    ela_score = ela_utils.compute_ela_score(diff)
    out['ela_score'] = ela_score

    # Save heatmap if requested
    if args.out_heatmap:
        os.makedirs(os.path.dirname(args.out_heatmap) or '.', exist_ok=True)
        ela_utils.save_ela_heatmap(diff, args.out_heatmap)
        out['ela_heatmap'] = args.out_heatmap

    # pHash
    ph = phash_utils.compute_phash(image_path)
    out['phash'] = ph

    # If MongoDB-backed pHash DB is available, use similarity search
    phash_result = None
    if phash_db is not None:
        try:
            phash_result = phash_db.find_similar_phash(ph)
        except Exception as e:
            # gracefully handle DB connection or query errors
            logging.warning('pHash DB query failed: %s', e)
            phash_result = None

    # Backwards-compatible CSV-based phash DB support (optional)
    if phash_result is None and args.phash_db and os.path.exists(args.phash_db):
        db = phash_utils.load_phash_db(args.phash_db)
        ph_score_csv, best_fn, best_h = phash_utils.compute_phash_score(ph, db)
        # emulate phash_result structure
        phash_result = {
            'match': best_fn is not None,
            'match_type': 'exact' if best_h == 0 else ('near' if best_h <= 10 else 'none'),
            'min_distance': best_h,
            'matched_image_id': best_fn,
        }

    # Fill output fields for pHash similarity
    if phash_result is not None:
        out['phash_match'] = bool(phash_result.get('match', False))
        out['phash_match_type'] = phash_result.get('match_type')
        out['phash_min_distance'] = phash_result.get('min_distance')
        out['phash_matched_image_id'] = phash_result.get('matched_image_id')
        # Convert match type to numeric score: exact=1.0, near=0.7, none=0.0
        mt = phash_result.get('match_type')
        phash_score_map = {'exact': 1.0, 'near': 0.7, 'none': 0.0}
        out['phash_score'] = phash_score_map.get(mt, 0.0)
    else:
        out['phash_match'] = False
        out['phash_match_type'] = 'none'
        out['phash_min_distance'] = None
        out['phash_matched_image_id'] = None
        out['phash_score'] = 0.0

    # CNN (unchanged)
    if args.model and os.path.exists(args.model):
        cnn_score = compute_cnn_score(image_path, args.model, tampered_index=args.tampered_index)
        out['cnn_score'] = cnn_score
    else:
        cnn_score = args.cnn_score
        out['cnn_score'] = cnn_score

    # fuse
    final = fuse_scores(out['cnn_score'], out['ela_score'], out['phash_score'], weights=(args.w_cnn, args.w_ela, args.w_phash))
    out['final_image_score'] = final
    out['verdict'] = 'Suspected Manipulation' if final >= args.threshold else 'Authentic'

    return out


def main(args):
    # Demo mode: create temporary images and demo DB entries to show exact/near/none
    if args.demo:
        import tempfile
        from PIL import Image, ImageChops
        tmpdir = tempfile.mkdtemp()
        # Create three images: exact, near (one pixel changed), unrelated
        img_exact = os.path.join(tmpdir, 'exact.png')
        img_near = os.path.join(tmpdir, 'near.png')
        img_unrelated = os.path.join(tmpdir, 'unrelated.png')
        Image.new('RGB', (64,64), color=(10,10,10)).save(img_exact)
        # near: copy and change one pixel
        img = Image.open(img_exact)
        img.putpixel((0,0),(11,10,10))
        img.save(img_near)
        Image.new('RGB', (64,64), color=(200,50,50)).save(img_unrelated)

        # Insert the exact image phash into DB to simulate historical fraud
        if phash_db is None:
            print('pHash DB not available (pymongo missing); demo cannot insert demo records.')
        else:
            ph_exact = phash_utils.compute_phash(img_exact)
            ph_near = phash_utils.compute_phash(img_near)
            ph_unrelated = phash_utils.compute_phash(img_unrelated)
            phash_db.insert_phash(ph_exact, 'DEMO_EXACT', 'DEMO')
            phash_db.insert_phash(ph_near, 'DEMO_NEAR', 'DEMO')
            print('Inserted demo phashes into MongoDB.')

        for path in [img_unrelated, img_near, img_exact]:
            print('\nRunning inference on', path)
            out = _process_image(args, path)
            print('Final score:', out['final_image_score'], 'verdict:', out['verdict'])
            print(json.dumps(out, indent=2))
        return

    # Normal single-image mode
    out = _process_image(args, args.image)
    print(json.dumps(out, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w', encoding='utf8') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', help='Path to CNN checkpoint (optional)')
    parser.add_argument('--tampered_index', type=int, default=1, help='Index of tampered class in CNN outputs')
    parser.add_argument('--cnn_score', type=float, help='Provide cnn score manually (0-1)')
    parser.add_argument('--phash_db', help='CSV produced by utils/phash.py (filename,phash_hex)')
    parser.add_argument('--ela_quality', type=int, default=90)
    parser.add_argument('--ela_scale', type=int, default=10)
    parser.add_argument('--out_heatmap', help='Save ELA heatmap to this path')
    parser.add_argument('--out_json', help='Save JSON output to this path')
    parser.add_argument('--threshold', type=float, default=0.5)

    # Updated default fusion weights: 0.55 * cnn + 0.25 * ela + 0.20 * phash
    parser.add_argument('--w_cnn', type=float, default=0.55)
    parser.add_argument('--w_ela', type=float, default=0.25)
    parser.add_argument('--w_phash', type=float, default=0.20)

    # Demo mode: create temporary images and demo DB entries to show exact/near/none
    parser.add_argument('--demo', action='store_true', help='Run built-in demo (inserts demo phashes into MongoDB and runs three queries)')

    args = parser.parse_args()
    main(args)
