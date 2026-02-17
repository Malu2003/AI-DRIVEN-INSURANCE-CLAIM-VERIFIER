"""Generate a one-page PDF summary of current project status and key artifacts.
Saves to reports/output/progress_summary.pdf
"""
import os
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PDF = os.path.join(OUT_DIR, 'progress_summary.pdf')

# Files to include if present
CASIA_METRICS = os.path.join(ROOT, 'eval', 'casia', 'casia_metrics.json')
LC_STATS = os.path.join(ROOT, 'eval', 'lc25000', 'lc25000_stats.json')
CASIA_ROC = os.path.join(ROOT, 'eval', 'casia', 'casia_roc.png')
ELA_EXAMPLE = os.path.join(ROOT, 'out', 'ela_heat.jpg')

# Load metrics if available
casia = None
lc = None
try:
    if os.path.exists(CASIA_METRICS):
        with open(CASIA_METRICS, 'r', encoding='utf8') as f:
            casia = json.load(f)
except Exception:
    casia = None
try:
    if os.path.exists(LC_STATS):
        with open(LC_STATS, 'r', encoding='utf8') as f:
            lc = json.load(f)
except Exception:
    lc = None

# Page settings
W, H = 1240, 1754  # portrait
margin = 60
line_h = 32
title_h = 48

img = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(img)
try:
    font_b = ImageFont.truetype('arialbd.ttf', 36)
    font_r = ImageFont.truetype('arial.ttf', 20)
    font_title = ImageFont.truetype('arialbd.ttf', 48)
except Exception:
    font_b = ImageFont.load_default()
    font_r = ImageFont.load_default()
    font_title = ImageFont.load_default()

x = margin
y = margin

# Title
d.text((x, y), 'Progress Summary — Image Forgery & ICD Validation', fill='black', font=font_title)
y += title_h + 10

# Short bullets
bullets = [
    'Goal: Detect image manipulation (CASIA) and validate declared ICD codes in clinical notes',
    'Approach: DenseNet CNN + ELA + pHash fused at fixed weights (0.5/0.3/0.2). Threshold 0.5',
]
for b in bullets:
    d.text((x, y), '• ' + b, fill='black', font=font_r)
    y += line_h

y += 10

def draw_metrics(title, metrics):
    global x, y
    d.text((x, y), title, fill='black', font=font_b)
    y += line_h
    if metrics:
        for k,v in metrics.items():
            d.text((x+20, y), f'{k}: {v}', fill='black', font=font_r)
            y += line_h
    else:
        d.text((x+20, y), 'No metrics available', fill='black', font=font_r)
        y += line_h
    y += 6

# CASIA
casia_metrics = casia if isinstance(casia, dict) else None
if casia_metrics:
    # select a few
    short = {k: round(v,4) if isinstance(v,(float,int)) else v for k,v in casia_metrics.items()}
else:
    short = None

draw_metrics('CASIA evaluation (key):', short)

# LC25000
short_lc = lc if isinstance(lc, dict) else None
draw_metrics('LC25000 sanity stats:', short_lc)

# Files to check
files = [
    ('CASIA ROC', CASIA_ROC),
    ('ELA sample', ELA_EXAMPLE),
]

# Place small images on the right side if present
img_x = W - 360
img_y = margin
for label, path in files:
    if os.path.exists(path):
        try:
            mini = Image.open(path).convert('RGB')
            mini.thumbnail((320, 240))
            img.paste(mini, (img_x, img_y))
            d.text((img_x, img_y+mini.height+6), label, fill='black', font=font_r)
            img_y += mini.height + 60
        except Exception:
            pass

# Footer with artifact paths
y = H - 200
small = ImageFont.load_default()
d.text((x, y), 'Artifacts:', fill='black', font=font_b)
y += 26
artifact_lines = [
    'eval/casia/casia_results.csv  | eval/casia/casia_roc.png  | out/score.json',
    'eval/lc25000/lc25000_stats.json  | reports/progress_review.ipynb',
]
for ln in artifact_lines:
    d.text((x+10, y), ln, fill='black', font=font_r)
    y += line_h

# Save as PDF
img.save(OUT_PDF, 'PDF', resolution=150)
print('Saved PDF summary to', OUT_PDF)