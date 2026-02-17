"""
PDF Report Generator for Claim Verification
============================================

Converts pipeline output to professional PDF reports using WeasyPrint.

Usage:
    builder = ReportBuilder()
    pdf_path = builder.generate_pdf(pipeline_output)
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import base64

try:
    from weasyprint import WeasyPrint, CSS
except ImportError:
    WeasyPrint = None

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class ReportBuilder:
    """Generate professional PDF reports from pipeline verification results."""

    def __init__(self):
        """Initialize the report builder."""
        if WeasyPrint is None:
            raise ImportError(
                "WeasyPrint is required for PDF generation. "
                "Install with: pip install weasyprint"
            )
        
        self.templates_dir = PROJECT_ROOT / 'backend' / 'templates'
        self.output_dir = PROJECT_ROOT / 'backend' / 'temp_uploads'
        
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_pdf(self, verification_result: Dict[str, Any]) -> str:
        """
        Generate a PDF report from pipeline verification result.

        Args:
            verification_result (dict): Complete pipeline output

        Returns:
            str: Path to generated PDF file
        """
        # Generate HTML from pipeline result
        html_content = self._build_html(verification_result)

        # Create temporary PDF file
        pdf_file = tempfile.NamedTemporaryFile(
            suffix='.pdf',
            delete=False,
            dir=str(self.output_dir)
        )
        pdf_path = pdf_file.name
        pdf_file.close()

        # Convert HTML to PDF
        try:
            WeasyPrint(
                string=html_content,
                base_url=str(self.templates_dir)
            ).write_pdf(pdf_path)
        except Exception as e:
            # Clean up on error
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            raise RuntimeError(f"PDF generation failed: {str(e)}")

        return pdf_path

    def _build_html(self, verification_result: Dict[str, Any]) -> str:
        """
        Build HTML report from pipeline result.

        Args:
            verification_result (dict): Pipeline output

        Returns:
            str: HTML content
        """
        metadata = verification_result.get('metadata', {})
        icd = verification_result.get('icd_verification', {})
        image = verification_result.get('image_analysis', {})
        fraud = verification_result.get('fraud_assessment', {})
        verdict = verification_result.get('integrated_verdict', {})
        explanation = verification_result.get('explanation', 'No explanation available')

        # Timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Build HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Claim Verification Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 900px;
            margin: 20px auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 28px;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        
        .section {{
            margin-bottom: 30px;
        }}
        
        .section-title {{
            background: #34495e;
            color: white;
            padding: 12px 15px;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            border-left: 4px solid #e74c3c;
        }}
        
        .section-content {{
            padding: 0 15px;
        }}
        
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .info-label {{
            font-weight: bold;
            color: #2c3e50;
            min-width: 200px;
        }}
        
        .info-value {{
            color: #555;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 13px;
        }}
        
        .status-valid {{
            background: #2ecc71;
            color: white;
        }}
        
        .status-uncertain {{
            background: #f39c12;
            color: white;
        }}
        
        .status-flagged {{
            background: #e74c3c;
            color: white;
        }}
        
        .status-authentic {{
            background: #27ae60;
            color: white;
        }}
        
        .status-suspicious {{
            background: #e67e22;
            color: white;
        }}
        
        .status-tampered {{
            background: #c0392b;
            color: white;
        }}
        
        .status-low {{
            background: #27ae60;
            color: white;
        }}
        
        .status-medium {{
            background: #f39c12;
            color: white;
        }}
        
        .status-high {{
            background: #e67e22;
            color: white;
        }}
        
        .status-critical {{
            background: #c0392b;
            color: white;
        }}
        
        .score-row {{
            display: flex;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .score-label {{
            min-width: 180px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .score-bar {{
            flex-grow: 1;
            height: 20px;
            background: #ecf0f1;
            border-radius: 3px;
            margin: 0 10px;
            overflow: hidden;
        }}
        
        .score-fill {{
            height: 100%;
            background: linear-gradient(to right, #e74c3c, #f39c12, #2ecc71);
            width: {self._get_score_width(fraud.get('fraud_risk_percentage', 0))}%;
        }}
        
        .score-value {{
            min-width: 50px;
            text-align: right;
            font-weight: bold;
        }}
        
        .verdict-box {{
            background: #ecf0f1;
            border: 2px solid #34495e;
            border-radius: 4px;
            padding: 20px;
            margin: 15px 0;
            text-align: center;
        }}
        
        .verdict-text {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .verdict-confidence {{
            font-size: 13px;
            color: #7f8c8d;
        }}
        
        .explanation {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
            font-size: 13px;
            line-height: 1.8;
        }}
        
        .risk-factor {{
            background: #ffe8e8;
            border-left: 4px solid #e74c3c;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 3px;
            font-size: 13px;
        }}
        
        .heatmap-section {{
            margin: 20px 0;
            text-align: center;
        }}
        
        .heatmap-image {{
            max-width: 100%;
            max-height: 400px;
            border: 1px solid #bdc3c7;
            border-radius: 4px;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            font-size: 12px;
            color: #95a5a6;
        }}
        
        .disclaimer {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 4px;
            padding: 15px;
            margin: 20px 0;
            font-size: 12px;
            color: #856404;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        
        table tr {{
            border-bottom: 1px solid #ecf0f1;
        }}
        
        table td {{
            padding: 10px 5px;
        }}
        
        table td:first-child {{
            font-weight: bold;
            width: 200px;
            color: #2c3e50;
        }}
        
        .page-break {{
            page-break-after: always;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Insurance Claim Verification Report</h1>
            <p>AI-Assisted Claim Assessment System | {timestamp}</p>
        </div>

        <!-- Claim Overview -->
        <div class="section">
            <div class="section-title">Claim Overview</div>
            <div class="section-content">
                <div class="info-row">
                    <span class="info-label">Claim ID:</span>
                    <span class="info-value">{metadata.get('claim_id', 'N/A')}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Patient ID:</span>
                    <span class="info-value">{metadata.get('patient_id', 'N/A')}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Claim Amount:</span>
                    <span class="info-value">${{{metadata.get('claim_amount', 0):.2f}}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Previous Claims:</span>
                    <span class="info-value">{metadata.get('previous_claim_count', 0)}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Assessment Date:</span>
                    <span class="info-value">{metadata.get('timestamp', 'N/A')}</span>
                </div>
            </div>
        </div>

        <!-- ICD Code Verification -->
        <div class="section">
            <div class="section-title">ICD Code Verification</div>
            <div class="section-content">
                <div class="info-row">
                    <span class="info-label">Status:</span>
                    <span class="info-value">
                        <span class="status-badge status-{icd.get('status', 'error')}">
                            {icd.get('status', 'ERROR').upper()}
                        </span>
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Match Score:</span>
                    <span class="info-value">{icd.get('match_score', 0.0):.2%}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Codes Detected:</span>
                    <span class="info-value">{icd.get('num_icds_detected', 0)}</span>
                </div>
                {self._format_icd_codes(icd.get('predicted_icds', []))}
            </div>
        </div>

        <!-- Medical Image Analysis -->
        <div class="section">
            <div class="section-title">Medical Image Analysis</div>
            <div class="section-content">
                <div class="info-row">
                    <span class="info-label">Forgery Verdict:</span>
                    <span class="info-value">
                        <span class="status-badge status-{image.get('forgery_verdict', 'error')}">
                            {image.get('forgery_verdict', 'ERROR').upper()}
                        </span>
                    </span>
                </div>
                <div class="score-row">
                    <span class="score-label">CNN Score:</span>
                    <span class="score-value">{image.get('cnn_score', 0.0):.3f}</span>
                </div>
                <div class="score-row">
                    <span class="score-label">ELA Score:</span>
                    <span class="score-value">{image.get('ela_score', 0.0):.3f}</span>
                </div>
                <div class="score-row">
                    <span class="score-label">pHash Score:</span>
                    <span class="score-value">{(f"{image.get('phash_score'):.3f}" if image.get('phash_score') is not None else 'N/A')}</span>
                </div>
                <div class="score-row">
                    <span class="score-label">Fused Score:</span>
                    <span class="score-value">{image.get('fused_score', 0.0):.3f}</span>
                </div>
            </div>
        </div>

        <!-- Fraud Risk Assessment -->
        <div class="section">
            <div class="section-title">Fraud Risk Assessment</div>
            <div class="section-content">
                <div class="info-row">
                    <span class="info-label">Risk Level:</span>
                    <span class="info-value">
                        <span class="status-badge status-{fraud.get('risk_level', 'error')}">
                            {fraud.get('risk_level', 'ERROR').upper()}
                        </span>
                    </span>
                </div>
                <div class="score-row">
                    <span class="score-label">Fraud Risk:</span>
                    <div class="score-bar">
                        <div class="score-fill"></div>
                    </div>
                    <span class="score-value">{fraud.get('fraud_risk_percentage', 0.0):.1f}%</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Recommendation:</span>
                    <span class="info-value">
                        <strong>{fraud.get('recommendation', 'MANUAL_REVIEW').upper()}</strong>
                    </span>
                </div>
                {self._format_risk_factors(fraud.get('risk_factors', []))}
            </div>
        </div>

        <!-- Integrated Verdict -->
        <div class="section">
            <div class="section-title">Integrated Verdict</div>
            <div class="section-content">
                <div class="verdict-box">
                    <div class="verdict-text">
                        {verdict.get('overall_recommendation', 'MANUAL_REVIEW').upper()}
                    </div>
                    <div class="verdict-confidence">
                        Confidence: {verdict.get('confidence', 0.0):.0%}
                    </div>
                </div>
                <div class="info-row">
                    <span class="info-label">Summary:</span>
                    <span class="info-value">{verdict.get('risk_summary', 'No summary available')}</span>
                </div>
            </div>
        </div>

        <!-- Explanation -->
        <div class="section">
            <div class="section-title">Detailed Assessment</div>
            <div class="section-content">
                <div class="explanation">
                    {self._format_explanation(explanation)}
                </div>
            </div>
        </div>

        <!-- Disclaimer -->
        <div class="disclaimer">
            <strong>⚠️ DISCLAIMER:</strong> This report is generated using an AI-assisted decision support system. 
            All recommendations should be reviewed by qualified insurance professionals before final approval. 
            The system assists in fraud detection and should not be the sole basis for claim decisions.
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Generated: {timestamp}</p>
            <p>System: AI-Driven Image Forgery Detection & Claim Verification Pipeline v1.0</p>
            <p>This is a computer-generated report. Manual verification is recommended.</p>
        </div>
    </div>
</body>
</html>
        """
        return html

    @staticmethod
    def _format_icd_codes(codes: list) -> str:
        """Format ICD codes as HTML."""
        if not codes:
            return '<div class="info-row"><span class="info-label">ICD Codes:</span><span class="info-value">None detected</span></div>'

        html = '<div class="info-row"><span class="info-label">ICD Codes:</span><span class="info-value">'
        for i, (code, conf) in enumerate(codes[:5]):  # Show top 5
            html += f'<div>{code} ({conf:.1%})</div>'
        html += '</span></div>'
        return html

    @staticmethod
    def _format_risk_factors(factors: list) -> str:
        """Format risk factors as HTML."""
        if not factors or (len(factors) == 1 and 'No major risk factors' in factors[0]):
            return ''

        html = '<div class="info-row" style="flex-direction: column;"><span class="info-label">Risk Factors:</span><span class="info-value">'
        for factor in factors[:5]:  # Show top 5
            html += f'<div class="risk-factor">• {factor}</div>'
        html += '</span></div>'
        return html

    @staticmethod
    def _format_explanation(explanation: str) -> str:
        """Format explanation text, preserving line breaks."""
        # Replace newlines with <br> tags
        return explanation.replace('\n', '<br>')

    @staticmethod
    def _get_score_width(percentage: float) -> int:
        """Get width percentage for score bar."""
        return min(int(percentage), 100)
