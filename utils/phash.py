"""
Perceptual hash (pHash) utility.
Implements a DCT-based pHash compatible with common implementations.
Produces hex string output and a small CLI to save CSV of hashes.
"""
import os
import csv
import numpy as np
import cv2


def compute_phash(image_path, hash_size=8, highfreq_factor=4):
    """Compute pHash for an image and return hex string.

    Args:
        image_path (str): Path to image
        hash_size (int): size of hash (8 -> 64-bit)
        highfreq_factor (int): multiply hash_size to get DCT size

    Returns:
        str: hex representation of pHash
    """
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    img = cv2.resize(img, (hash_size * highfreq_factor, hash_size * highfreq_factor), interpolation=cv2.INTER_LINEAR)
    img = np.float32(img)

    # DCT
    dct = cv2.dct(img)

    # take top-left block
    dct_lowfreq = dct[:hash_size, :hash_size]

    med = np.median(dct_lowfreq.flatten())
    diff = dct_lowfreq > med

    # build integer from bits
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)

    # convert to hex, pad to hash_size*hash_size/4 chars
    hex_len = (hash_size * hash_size) // 4
    phash_hex = f"{bits:0{hex_len}x}"
    return phash_hex


def process_directory_to_csv(input_dir, output_csv_path):
    """Process images in a directory and write CSV with columns [filename, phash_hex]."""
    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    rows = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = os.path.join(root, fn)
            try:
                ph = compute_phash(path)
                rel = os.path.relpath(path, start=input_dir)
                rows.append((rel.replace('\\', '/'), ph))
            except Exception as e:
                print(f"Skipping {path}: {e}")

    with open(output_csv_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'phash_hex'])
        for r in rows:
            writer.writerow(r)


def load_phash_db(csv_path):
    """Load CSV from `process_directory_to_csv` and return list of (filename, phash_hex)."""
    rows = []
    with open(csv_path, newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r['filename'], r['phash_hex']))
    return rows


def hamming_distance_hex(a_hex, b_hex, hash_size=8):
    """Compute Hamming distance between two hex pHash strings."""
    # convert to integers
    a = int(a_hex, 16)
    b = int(b_hex, 16)
    x = a ^ b
    # count bits
    return x.bit_count()


def compute_phash_score(phash_hex, phash_db, hash_size=8):
    """Compute a normalized phash score in [0,1].

    phash_db: list of (filename, phash_hex)
    Score = min_hamming / max_bits; higher => more dissimilar (more suspicious).
    Returns (score, best_match_filename, best_hamming)
    """
    max_bits = hash_size * hash_size
    best = None
    best_h = max_bits + 1
    for fn, h in phash_db:
        try:
            hd = hamming_distance_hex(phash_hex, h, hash_size=hash_size)
            if hd < best_h:
                best_h = hd
                best = fn
        except Exception:
            continue
    if best is None:
        return 1.0, None, None
    score = float(best_h) / float(max_bits)
    return score, best, best_h


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute pHash for images and save CSV')
    parser.add_argument('input_dir', help='Input directory of images')
    parser.add_argument('output_csv', help='Output CSV path (e.g. data/phash.csv)')
    args = parser.parse_args()
    process_directory_to_csv(args.input_dir, args.output_csv)
