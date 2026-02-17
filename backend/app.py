from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.claim_verification_pipeline import ClaimVerificationPipeline
from backend.document_processor import extract_text_from_document

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'backend/temp_uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'}
ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'txt', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize pipeline
pipeline = ClaimVerificationPipeline()

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/api/verify-claim', methods=['POST'])
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
        print("\n" + "="*60, flush=True)
        print("📥 NEW CLAIM VERIFICATION REQUEST", flush=True)
        print("="*60, flush=True)
        
        # Validate files
        if 'clinical_document' not in request.files:
            print("❌ ERROR: Missing clinical_document file")
            return jsonify({'error': 'Missing clinical_document file'}), 400
        if 'image' not in request.files:
            print("❌ ERROR: Missing image file")
            return jsonify({'error': 'Missing image file'}), 400
        if 'claim_amount' not in request.form:
            print("❌ ERROR: Missing claim_amount")
            return jsonify({'error': 'Missing claim_amount'}), 400

        clinical_doc = request.files['clinical_document']
        image_file = request.files['image']
        claim_amount = float(request.form['claim_amount'])
        claim_id = request.form.get('claim_id', None)
        patient_id = request.form.get('patient_id', None)
        previous_claim_count = int(request.form.get('previous_claim_count', 0))

        print(f"Document: {clinical_doc.filename}")
        print(f"Image: {image_file.filename}")
        print(f"Claim Amount: ${claim_amount}")

        # Validate filenames
        if clinical_doc.filename == '' or image_file.filename == '':
            print("❌ ERROR: No file selected")
            return jsonify({'error': 'No file selected'}), 400

        # Save files
        doc_filename = secure_filename(clinical_doc.filename)
        img_filename = secure_filename(image_file.filename)
        doc_path = os.path.join(app.config['UPLOAD_FOLDER'], doc_filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        
        clinical_doc.save(doc_path)
        image_file.save(img_path)
        print(f"✓ Files saved: {doc_path}, {img_path}")

        # Extract text from document
        print("📄 Extracting text from document...")
        clinical_text = extract_text_from_document(doc_path)
        
        if not clinical_text.strip():
            print("❌ ERROR: Could not extract text from document")
            return jsonify({'error': 'Could not extract text from document'}), 400
        
        print(f"✓ Text extracted: {len(clinical_text)} characters")

        # Run pipeline
        print("🔍 Running verification pipeline...")
        result = pipeline.verify_claim(
            clinical_text=clinical_text,
            image_path=img_path,
            claim_amount=claim_amount,
            previous_claim_count=previous_claim_count,
            patient_id=patient_id,
            claim_id=claim_id,
            save_output=False
        )
        
        print(f"✓ Pipeline complete!")
        print(f"  Verdict: {result.get('recommendation', 'N/A')}")
        print(f"  Image: {result.get('image_analysis', {}).get('forgery_verdict', 'N/A')}")
        print("="*60 + "\n")

        # Cleanup
        try:
            os.remove(doc_path)
            os.remove(img_path)
        except:
            pass

        return jsonify(result)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("\n" + "="*60)
        print("❌ VERIFICATION ERROR:")
        print("="*60)
        print(error_details)
        print("="*60 + "\n")
        return jsonify({'error': f'Verification failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 Claim Verification Backend Starting...")
    print("="*60)
    print("Endpoints:")
    print("  POST /api/verify-claim - Complete claim verification")
    print("="*60)
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)