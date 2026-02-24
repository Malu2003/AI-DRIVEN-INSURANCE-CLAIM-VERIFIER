from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import os
import sys
import re
from pathlib import Path
from datetime import datetime
import time
from functools import wraps

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.claim_verification_pipeline import ClaimVerificationPipeline
from backend.document_processor import extract_text_from_document
from backend.report_generator.report_builder import ReportBuilder

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'backend/temp_uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'}
ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'txt', 'docx'}
ALLOWED_IMAGE_MIME_TYPES = {
    'image/png',
    'image/jpeg',
    'image/jpg',
    'image/bmp',
    'image/tiff',
    'application/dicom',
    'application/dicom+json',
    'application/octet-stream',  # fallback for some DICOM uploads
}
ALLOWED_DOCUMENT_MIME_TYPES = {
    'application/pdf',
    'text/plain',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/octet-stream',  # fallback when browser does not provide exact type
}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

UPLOAD_RETENTION_SECONDS = int(os.getenv('UPLOAD_RETENTION_SECONDS', '900'))  # 15 minutes
AUTH_REQUIRED = os.getenv('AUTH_REQUIRED', 'true').strip().lower() in {'1', 'true', 'yes', 'on'}
API_TOKEN = os.getenv('API_TOKEN', '').strip()

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize pipeline
pipeline = ClaimVerificationPipeline()


def error_response(message, status_code=400):
    return jsonify({
        "error": True,
        "message": message,
        "status_code": status_code,
    }), status_code


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_e):
    return error_response("File too large (max: 50MB)", 413)


def cleanup_temp_uploads(max_age_seconds=UPLOAD_RETENTION_SECONDS):
    now = time.time()
    upload_dir = app.config.get('UPLOAD_FOLDER', UPLOAD_FOLDER)
    if not os.path.isdir(upload_dir):
        return
    for entry in os.listdir(upload_dir):
        path = os.path.join(upload_dir, entry)
        try:
            if not os.path.isfile(path):
                continue
            age = now - os.path.getmtime(path)
            if age > max_age_seconds:
                os.remove(path)
        except Exception:
            continue


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not AUTH_REQUIRED:
            return f(*args, **kwargs)

        if not API_TOKEN:
            return error_response('Server auth misconfiguration: API_TOKEN not set', 500)

        auth_header = request.headers.get('Authorization', '').strip()
        expected = f'Bearer {API_TOKEN}'
        if auth_header != expected:
            return error_response('Unauthorized', 401)
        return f(*args, **kwargs)
    return wrapper

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def _save_uploaded_file(uploaded_file, allowed_extensions, allowed_mime_types, field_name):
    if uploaded_file is None:
        raise ValueError(f"Missing required file field: {field_name}")
    if uploaded_file.filename is None or uploaded_file.filename.strip() == "":
        raise ValueError(f"No file selected for field: {field_name}")
    if not allowed_file(uploaded_file.filename, allowed_extensions):
        raise TypeError(f"Unsupported file format for {field_name}")

    content_type = (uploaded_file.mimetype or '').strip().lower()
    if content_type and content_type not in allowed_mime_types:
        raise TypeError(f"Unsupported MIME type for {field_name}: {content_type}")

    filename = secure_filename(uploaded_file.filename)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    unique_name = f"{timestamp}_{filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    uploaded_file.save(path)
    return path


def _extract_clinical_text_from_request(saved_paths):
    clinical_text = request.form.get('clinical_text', '').strip()
    if clinical_text:
        return clinical_text

    clinical_doc = request.files.get('clinical_document')
    if clinical_doc is None:
        raise ValueError("Missing required input: provide clinical_text or clinical_document")

    doc_path = _save_uploaded_file(
        clinical_doc,
        ALLOWED_DOCUMENT_EXTENSIONS,
        ALLOWED_DOCUMENT_MIME_TYPES,
        'clinical_document'
    )
    saved_paths.append(doc_path)
    extracted_text = extract_text_from_document(doc_path)
    if not extracted_text.strip():
        raise ValueError("Could not extract text from clinical_document")
    return extracted_text


def _extract_patient_identifier_from_text(clinical_text: str):
    """Extract likely patient identifier from uploaded clinical text."""
    if not clinical_text:
        return None

    patterns = [
        r'(?im)^\s*patient\s*id\s*[:\-]\s*([^\n\r]+)',
        r'(?im)^\s*patient\s*identifier\s*[:\-]\s*([^\n\r]+)',
        r'(?im)^\s*pid\s*[:\-]\s*([^\n\r]+)',
        r'(?im)^\s*mrn\s*[:\-]\s*([^\n\r]+)',
        r'(?im)^\s*uhid\s*[:\-]\s*([^\n\r]+)',
        r'(?im)^\s*hospital\s*id\s*[:\-]\s*([^\n\r]+)',
        r'(?im)^\s*patient\s*name\s*[:\-]\s*([^\n\r]+)',
        r'(?im)^\s*name\s*[:\-]\s*([^\n\r]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, clinical_text)
        if not match:
            continue
        candidate = match.group(1).strip()
        candidate = re.sub(r'\s+', ' ', candidate)
        if candidate:
            return candidate[:120]

    return None


def _run_pipeline_from_request(save_output=False):
    saved_paths = []
    try:
        cleanup_temp_uploads()

        if 'claim_amount' not in request.form:
            raise ValueError("Missing required field: claim_amount")

        try:
            claim_amount = float(request.form['claim_amount'])
        except Exception:
            raise ValueError("Invalid claim_amount: must be numeric")

        previous_claim_count_raw = request.form.get('previous_claim_count', '0')
        try:
            previous_claim_count = int(previous_claim_count_raw)
        except Exception:
            raise ValueError("Invalid previous_claim_count: must be integer")

        image_file = request.files.get('image')
        if image_file is None:
            raise ValueError("Missing required file field: image")

        image_path = _save_uploaded_file(
            image_file,
            ALLOWED_IMAGE_EXTENSIONS,
            ALLOWED_IMAGE_MIME_TYPES,
            'image'
        )
        saved_paths.append(image_path)

        clinical_text = _extract_clinical_text_from_request(saved_paths)

        claim_id = request.form.get('claim_id', None)
        patient_id = request.form.get('patient_id', None)
        if not patient_id:
            patient_id = _extract_patient_identifier_from_text(clinical_text)

        result = pipeline.verify_claim(
            clinical_text=clinical_text,
            image_path=image_path,
            claim_amount=claim_amount,
            previous_claim_count=previous_claim_count,
            patient_id=patient_id,
            claim_id=claim_id,
            save_output=save_output,
        )

        image_analysis = result.get("image_analysis") if isinstance(result, dict) else None
        if isinstance(image_analysis, dict) and not image_analysis.get("ela_heatmap_path"):
            try:
                from utils import ela as ela_utils

                ela_diff = ela_utils.compute_ela(image_path, quality=90, scale=10)
                image_name = Path(image_path).stem
                ela_heatmap_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    f"{image_name}_ela_heatmap.png",
                )
                ela_utils.save_ela_visualization(ela_diff, ela_heatmap_path, scale=10)
                image_analysis["ela_heatmap_path"] = ela_heatmap_path
            except Exception:
                pass

        metadata = result.get("metadata") if isinstance(result, dict) else None
        if isinstance(metadata, dict):
            compact_text = " ".join(clinical_text.split())
            metadata["clinical_text_excerpt"] = compact_text[:1200]
            if not metadata.get("patient_id"):
                metadata["patient_id"] = patient_id

        if not result.get("success", True):
            message = result.get("error", "Pipeline runtime not ready")
            return None, error_response(message, 503)

        return result, None
    finally:
        for p in saved_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Claim Verification API",
        "version": "1.0.0",
    }), 200


@app.route('/api/status', methods=['GET'])
def api_status():
    readiness = getattr(pipeline, 'startup_readiness', {})
    return jsonify({
        "service": "Claim Verification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "verify_claim": "/verify-claim",
            "verify_claim_api": "/api/verify-claim",
            "generate_report": "/generate-report",
            "generate_report_api": "/api/generate-report",
            "status": "/api/status",
        },
        "pipeline": "v1.0",
        "strict_production": bool(readiness.get("strict_production", False)),
        "runtime_ready": bool(readiness.get("ready", False)),
        "supported_image_formats": sorted(list(ALLOWED_IMAGE_EXTENSIONS)),
        "supported_document_formats": sorted(list(ALLOWED_DOCUMENT_EXTENSIONS)),
        "max_upload_mb": 50,
        "auth_required": AUTH_REQUIRED,
        "upload_retention_seconds": UPLOAD_RETENTION_SECONDS,
    }), 200

@app.route('/verify-claim', methods=['POST'])
@app.route('/api/verify-claim', methods=['POST'])
@require_auth
def verify_claim():
    """
    Complete claim verification endpoint
    Expects: 
        - clinical_document: file (PDF, TXT, DOCX)
        - image: file (medical image)
        - claim_amount: float
        - claim_id: string (optional)
        - patient_id: string (optional)
        - previous_claim_count: int (optional)
    Returns: Complete verification report with ICD, image, and fraud analysis
    """
    try:
        result, err = _run_pipeline_from_request(save_output=False)
        if err is not None:
            return err
        return jsonify(result), 200
    except ValueError as e:
        return error_response(str(e), 400)
    except TypeError as e:
        return error_response(str(e), 415)
    except Exception as e:
        return error_response(f"Verification failed: {str(e)}", 500)


@app.route('/generate-report', methods=['POST'])
@app.route('/api/generate-report', methods=['POST'])
@require_auth
def generate_report():
    """Generate a downloadable PDF report from claim verification result."""
    pdf_path = None
    try:
        result, err = _run_pipeline_from_request(save_output=False)
        if err is not None:
            return err

        builder = ReportBuilder()
        pdf_path = builder.generate_pdf(result)

        claim_id = result.get("metadata", {}).get("claim_id") or "unknown"
        download_name = f"claim_report_{claim_id}.pdf"
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='application/pdf',
        )
    except ValueError as e:
        return error_response(str(e), 400)
    except TypeError as e:
        return error_response(str(e), 415)
    except ImportError as e:
        return error_response(f"PDF generation dependency missing: {str(e)}", 500)
    except Exception as e:
        return error_response(f"Report generation failed: {str(e)}", 500)
    finally:
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception:
                pass

if __name__ == '__main__':
    cleanup_temp_uploads()
    print("="*60)
    print("🚀 Claim Verification Backend Starting...")
    print("="*60)
    print("Endpoints:")
    print("  GET  /health")
    print("  GET  /api/status")
    print("  POST /verify-claim")
    print("  POST /api/verify-claim")
    print("  POST /generate-report")
    print("  POST /api/generate-report")
    print("="*60)
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)