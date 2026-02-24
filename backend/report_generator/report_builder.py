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
import re
from html import escape

try:
    from weasyprint import HTML
except Exception:
    HTML = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
    from reportlab.lib import colors
except Exception:
    A4 = None
    getSampleStyleSheet = None
    ParagraphStyle = None
    cm = None
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    Table = None
    TableStyle = None
    RLImage = None
    PageBreak = None
    colors = None

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class ReportBuilder:
    """Generate professional PDF reports from pipeline verification results."""

    def __init__(self):
        """Initialize the report builder."""
        if HTML is None and SimpleDocTemplate is None:
            raise ImportError(
                "PDF generation requires weasyprint or reportlab. "
                "Install with: pip install weasyprint or pip install reportlab"
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
        pdf_file = tempfile.NamedTemporaryFile(
            suffix='.pdf',
            delete=False,
            dir=str(self.output_dir)
        )
        pdf_path = pdf_file.name
        pdf_file.close()

        try:
            if HTML is not None:
                html_content = self._build_html(verification_result)
                HTML(
                    string=html_content,
                    base_url=str(self.templates_dir)
                ).write_pdf(pdf_path)
            elif SimpleDocTemplate is not None:
                self._build_pdf_reportlab(verification_result, pdf_path)
            else:
                raise RuntimeError('No PDF backend available (weasyprint/reportlab).')
        except Exception as e:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            raise RuntimeError(f"PDF generation failed: {str(e)}")

        return pdf_path

    def _build_pdf_reportlab(self, verification_result: Dict[str, Any], pdf_path: str) -> None:
        metadata = verification_result.get('metadata', {})
        icd = verification_result.get('icd_verification', {})
        image = verification_result.get('image_analysis', {})
        fraud = verification_result.get('fraud_assessment', {})
        verdict = verification_result.get('integrated_verdict', {})
        final_explanation = verification_result.get('explanation', '')

        clinical_text = metadata.get('clinical_text_excerpt', '')
        explicit_icds = self._extract_explicit_icd_codes(clinical_text)

        doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm)
        styles = getSampleStyleSheet()
        body = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=10, leading=14)
        heading = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor('#1f2d3d'))
        title = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=18, leading=22)

        story = []
        story.append(Paragraph('Insurance Claim Verification Report', title))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body))
        story.append(Spacer(1, 12))

        overview_data = [
            ['Claim ID', str(metadata.get('claim_id', 'N/A'))],
            ['Patient ID', str(metadata.get('patient_id', 'N/A'))],
            ['Claim Amount', f"₹{self._to_float(metadata.get('claim_amount'), 0.0):,.2f}"],
            ['Previous Claims', str(metadata.get('previous_claim_count', 0))],
            ['Assessment Date', str(metadata.get('timestamp', 'N/A'))],
        ]
        story.append(Paragraph('Claim Overview', heading))
        story.append(self._build_table(overview_data))
        story.append(Spacer(1, 10))

        predicted_icds = icd.get('predicted_icds', []) or []
        predicted_lines = []
        for item in predicted_icds[:5]:
            code, conf = self._normalize_icd_item(item)
            predicted_lines.append(f"• {code} ({conf:.1%})")
        if not predicted_lines:
            predicted_lines = ['• None detected']

        story.append(Paragraph('ICD Code Verification', heading))
        story.append(self._build_table([
            ['Status', str(icd.get('status', 'N/A')).upper()],
            ['Match Score', f"{self._to_float(icd.get('match_score'), 0.0):.2%}"],
            ['Explicit ICDs in Notes', ', '.join(explicit_icds[:10]) if explicit_icds else 'None explicitly found'],
            ['Predicted ICDs', '<br/>'.join(escape(line) for line in predicted_lines)],
        ], allow_markup=True))
        story.append(Paragraph(self._as_reportlab_markup(self._format_explanation(icd.get('explanation', 'No ICD explanation available'))), body))
        story.append(Spacer(1, 10))

        phash = image.get('phash_score')
        phash_display = 'N/A' if phash is None else f"{self._to_float(phash, 0.0):.3f}"
        story.append(Paragraph('Medical Image Analysis', heading))
        story.append(self._build_table([
            ['Forgery Verdict', str(image.get('forgery_verdict', 'N/A')).upper()],
            ['CNN Score', f"{self._to_float(image.get('cnn_score'), 0.0):.3f}"],
            ['ELA Score', f"{self._to_float(image.get('ela_score'), 0.0):.3f}"],
            ['pHash Score', phash_display],
            ['Fused Score', f"{self._to_float(image.get('fused_score'), 0.0):.3f}"],
        ]))
        story.append(Paragraph(self._as_reportlab_markup(self._format_explanation(image.get('explanation', 'No image explanation available'))), body))

        heatmap_path = image.get('ela_heatmap_path')
        if heatmap_path:
            path = Path(heatmap_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists() and path.is_file():
                try:
                    story.append(Spacer(1, 6))
                    story.append(Paragraph('ELA Heatmap (Explainability)', body))
                    img = RLImage(str(path))
                    img._restrictSize(16 * cm, 10 * cm)
                    story.append(img)
                except Exception:
                    story.append(Paragraph('ELA Heatmap: could not render image.', body))
            else:
                story.append(Paragraph('ELA Heatmap: not found.', body))
        else:
            story.append(Paragraph('ELA Heatmap: not available for this sample.', body))

        story.append(Spacer(1, 10))

        feature_scores = fraud.get('feature_scores', {}) if isinstance(fraud.get('feature_scores', {}), dict) else {}
        feature_lines = []
        for k, v in feature_scores.items():
            display = 'N/A' if v is None else f"{self._to_float(v, 0.0):.4f}"
            feature_lines.append(f"• {k}: {display}")

        risk_factors = fraud.get('risk_factors', []) or []
        risk_factor_text = '<br/>'.join(escape(str(x)) for x in risk_factors[:8]) if risk_factors else 'No major risk factors identified'

        story.append(Paragraph('Fraud Risk Assessment', heading))
        story.append(self._build_table([
            ['Risk Level', str(fraud.get('risk_level', 'N/A')).upper()],
            ['Fraud Risk Percentage', f"{self._to_float(fraud.get('fraud_risk_percentage'), 0.0):.1f}%"],
            ['Recommendation', str(fraud.get('recommendation', 'manual_review')).upper()],
            ['Risk Factors', risk_factor_text],
            ['Feature Metrics', '<br/>'.join(escape(x) for x in feature_lines) if feature_lines else 'N/A'],
        ], allow_markup=True))
        story.append(Paragraph(self._as_reportlab_markup(self._format_explanation(fraud.get('explanation', 'No fraud explanation available'))), body))

        story.append(PageBreak())
        story.append(Paragraph('Integrated Verdict & Final Rationale', heading))
        story.append(self._build_table([
            ['Overall Recommendation', str(verdict.get('overall_recommendation', 'manual_review')).upper()],
            ['Confidence', f"{self._to_float(verdict.get('confidence'), 0.0):.0%}"],
            ['Risk Summary', escape(str(verdict.get('risk_summary', 'No summary available')))],
        ], allow_markup=True))

        story.append(Spacer(1, 8))
        story.append(Paragraph(self._as_reportlab_markup(self._build_final_reasoning(icd, image, fraud, verdict)), body))
        story.append(Spacer(1, 8))
        story.append(Paragraph(self._as_reportlab_markup(self._format_explanation(final_explanation)), body))

        story.append(Spacer(1, 12))
        story.append(Paragraph('Disclaimer: This AI-assisted report supports decision-making and should be reviewed by qualified professionals.', body))

        doc.build(story)

    def _build_table(self, rows: list, allow_markup: bool = False):
        processed = []
        for left, right in rows:
            if allow_markup:
                right_value = Paragraph(str(right), getSampleStyleSheet()['BodyText'])
            else:
                right_value = Paragraph(escape(str(right)), getSampleStyleSheet()['BodyText'])
            processed.append([Paragraph(f"<b>{escape(str(left))}</b>", getSampleStyleSheet()['BodyText']), right_value])

        table = Table(processed, colWidths=[5.0 * cm, 11.0 * cm])
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#d9d9d9')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdbdbd')),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        return table

    @staticmethod
    def _as_reportlab_markup(text: str) -> str:
        """Normalize markup for ReportLab Paragraph parser."""
        normalized = str(text or '')
        normalized = normalized.replace('<br />', '<br/>').replace('<br>', '<br/>')
        return normalized

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
        clinical_text = metadata.get('clinical_text_excerpt', '')

        explicit_icds = self._extract_explicit_icd_codes(clinical_text)
        icd_status = (icd.get('status') or 'error').lower()
        image_verdict = (image.get('forgery_verdict') or 'error').lower()
        fraud_level = (fraud.get('risk_level') or 'error').lower()
        recommendation = (verdict.get('overall_recommendation') or 'manual_review').upper()

        fraud_pct = self._to_float(fraud.get('fraud_risk_percentage'), 0.0)
        icd_match_score = self._to_float(icd.get('match_score'), 0.0)
        cnn_score = self._to_float(image.get('cnn_score'), 0.0)
        ela_score = self._to_float(image.get('ela_score'), 0.0)
        fused_score = self._to_float(image.get('fused_score'), 0.0)
        confidence = self._to_float(verdict.get('confidence'), 0.0)
        phash_score_raw = image.get('phash_score')
        phash_score = None if phash_score_raw is None else self._to_float(phash_score_raw, 0.0)

        ela_image_html = self._build_ela_heatmap_html(image.get('ela_heatmap_path'))

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
                    <span class="info-value">₹{self._to_float(metadata.get('claim_amount'), 0.0):,.2f}</span>
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
                        <span class="status-badge status-{icd_status}">
                            {icd_status.upper()}
                        </span>
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Match Score:</span>
                    <span class="info-value">{icd_match_score:.2%}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Codes Detected:</span>
                    <span class="info-value">{icd.get('num_icds_detected', 0)}</span>
                </div>
                {self._format_explicit_icds(explicit_icds)}
                {self._format_icd_codes(icd.get('predicted_icds', []))}
                <div class="explanation">{self._format_explanation(icd.get('explanation', 'No ICD explanation available'))}</div>
            </div>
        </div>

        <!-- Medical Image Analysis -->
        <div class="section">
            <div class="section-title">Medical Image Analysis</div>
            <div class="section-content">
                <div class="info-row">
                    <span class="info-label">Forgery Verdict:</span>
                    <span class="info-value">
                        <span class="status-badge status-{image_verdict}">
                            {image_verdict.upper()}
                        </span>
                    </span>
                </div>
                <div class="score-row">
                    <span class="score-label">CNN Score:</span>
                    <span class="score-value">{cnn_score:.3f}</span>
                </div>
                <div class="score-row">
                    <span class="score-label">ELA Score:</span>
                    <span class="score-value">{ela_score:.3f}</span>
                </div>
                <div class="score-row">
                    <span class="score-label">pHash Score:</span>
                    <span class="score-value">{(f"{phash_score:.3f}" if phash_score is not None else 'N/A')}</span>
                </div>
                <div class="score-row">
                    <span class="score-label">Fused Score:</span>
                    <span class="score-value">{fused_score:.3f}</span>
                </div>
                <div class="explanation">{self._format_explanation(image.get('explanation', 'No image analysis explanation available'))}</div>
                {ela_image_html}
            </div>
        </div>

        <!-- Fraud Risk Assessment -->
        <div class="section">
            <div class="section-title">Fraud Risk Assessment</div>
            <div class="section-content">
                <div class="info-row">
                    <span class="info-label">Risk Level:</span>
                    <span class="info-value">
                        <span class="status-badge status-{fraud_level}">
                            {fraud_level.upper()}
                        </span>
                    </span>
                </div>
                <div class="score-row">
                    <span class="score-label">Fraud Risk:</span>
                    <div class="score-bar">
                        <div class="score-fill"></div>
                    </div>
                    <span class="score-value">{fraud_pct:.1f}%</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Recommendation:</span>
                    <span class="info-value">
                        <strong>{fraud.get('recommendation', 'MANUAL_REVIEW').upper()}</strong>
                    </span>
                </div>
                {self._format_risk_factors(fraud.get('risk_factors', []))}
                {self._format_feature_scores(fraud.get('feature_scores', {}))}
                <div class="explanation">{self._format_explanation(fraud.get('explanation', 'No fraud module explanation available'))}</div>
            </div>
        </div>

        <!-- Integrated Verdict -->
        <div class="section">
            <div class="section-title">Integrated Verdict</div>
            <div class="section-content">
                <div class="verdict-box">
                    <div class="verdict-text">
                        {recommendation}
                    </div>
                    <div class="verdict-confidence">
                        Confidence: {confidence:.0%}
                    </div>
                </div>
                <div class="info-row">
                    <span class="info-label">Summary:</span>
                    <span class="info-value">{escape(str(verdict.get('risk_summary', 'No summary available')))}</span>
                </div>
                <div class="explanation">{self._build_final_reasoning(icd, image, fraud, verdict)}</div>
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
        for item in codes[:5]:
            code, conf = ReportBuilder._normalize_icd_item(item)
            html += f'<div>{escape(str(code))} ({conf:.1%})</div>'
        html += '</span></div>'
        return html

    @staticmethod
    def _format_explicit_icds(explicit_codes: list) -> str:
        """Format explicit ICDs extracted directly from uploaded clinical notes."""
        if not explicit_codes:
            return '<div class="info-row"><span class="info-label">Explicit ICDs in Notes:</span><span class="info-value">None explicitly found in uploaded text</span></div>'

        joined = ', '.join(escape(str(code)) for code in explicit_codes[:10])
        return f'<div class="info-row"><span class="info-label">Explicit ICDs in Notes:</span><span class="info-value">{joined}</span></div>'

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
    def _format_feature_scores(feature_scores: dict) -> str:
        """Format feature score table for fraud module explainability."""
        if not isinstance(feature_scores, dict) or not feature_scores:
            return ''

        labels = {
            'icd_match_score': 'ICD Match Score',
            'cnn_forgery_score': 'CNN Forgery Score',
            'ela_score': 'ELA Score',
            'phash_score': 'pHash Score',
            'final_image_forgery_score': 'Final Image Forgery Score',
            'patient_match_score': 'Patient Match Score',
            'claim_amount_log': 'Claim Amount (log)',
            'previous_claim_count': 'Previous Claim Count',
        }

        rows = ''
        for key in labels:
            if key not in feature_scores:
                continue
            value = feature_scores.get(key)
            if value is None:
                display = 'N/A'
            else:
                try:
                    display = f'{float(value):.4f}'
                except Exception:
                    display = escape(str(value))
            rows += f'<tr><td>{labels[key]}</td><td>{display}</td></tr>'

        if not rows:
            return ''

        return f'<div class="info-row" style="display:block;"><span class="info-label">Fraud Feature Metrics:</span><table>{rows}</table></div>'

    @staticmethod
    def _format_explanation(explanation: str) -> str:
        """Format explanation text, preserving line breaks."""
        # Replace newlines with <br> tags
        safe = escape(str(explanation or ''))
        return safe.replace('\n', '<br>')

    @staticmethod
    def _get_score_width(percentage: float) -> int:
        """Get width percentage for score bar."""
        return min(int(percentage), 100)

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _extract_explicit_icd_codes(clinical_text: str) -> list:
        """Extract explicit ICD-like tokens from uploaded clinical text."""
        if not clinical_text:
            return []
        matches = re.findall(r'\b([A-TV-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\b', clinical_text.upper())
        deduped = []
        seen = set()
        for code in matches:
            if code in seen:
                continue
            seen.add(code)
            deduped.append(code)
        return deduped

    @staticmethod
    def _normalize_icd_item(item: Any) -> tuple:
        """Normalize ICD prediction entries into (code, confidence)."""
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            code = item[0]
            conf = ReportBuilder._to_float(item[1], 0.0)
            return code, conf
        if isinstance(item, dict):
            code = item.get('code') or item.get('icd') or 'UNKNOWN'
            conf = ReportBuilder._to_float(item.get('confidence'), 0.0)
            return code, conf
        return str(item), 0.0

    def _build_ela_heatmap_html(self, heatmap_path: Optional[str]) -> str:
        """Build inline HTML for ELA heatmap image if available."""
        if not heatmap_path:
            return '<div class="info-row"><span class="info-label">ELA Heatmap:</span><span class="info-value">Not available for this sample</span></div>'

        path = Path(heatmap_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists() or not path.is_file():
            return '<div class="info-row"><span class="info-label">ELA Heatmap:</span><span class="info-value">Heatmap file not found on server</span></div>'

        try:
            with open(path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
            return (
                '<div class="heatmap-section">'
                '<div class="info-label" style="margin-bottom:8px;">ELA Heatmap (Explainability)</div>'
                f'<img class="heatmap-image" src="data:{mime};base64,{encoded}" alt="ELA Heatmap" />'
                '</div>'
            )
        except Exception:
            return '<div class="info-row"><span class="info-label">ELA Heatmap:</span><span class="info-value">Could not embed heatmap image</span></div>'

    def _build_final_reasoning(
        self,
        icd: Dict[str, Any],
        image: Dict[str, Any],
        fraud: Dict[str, Any],
        verdict: Dict[str, Any],
    ) -> str:
        """Generate explicit rationale for final outcome."""
        lines = [
            'Final outcome rationale is based on weighted evidence from all modules:',
            f"1) ICD module: status={escape(str(icd.get('status', 'N/A')))}, match_score={self._to_float(icd.get('match_score'), 0.0):.2%}.",
            f"2) Image module: verdict={escape(str(image.get('forgery_verdict', 'N/A')))}, CNN={self._to_float(image.get('cnn_score'), 0.0):.3f}, ELA={self._to_float(image.get('ela_score'), 0.0):.3f}, pHash={(f'{self._to_float(image.get('phash_score'), 0.0):.3f}' if image.get('phash_score') is not None else 'N/A')}, fused={self._to_float(image.get('fused_score'), 0.0):.3f}.",
            f"3) Fraud module: risk={self._to_float(fraud.get('fraud_risk_percentage'), 0.0):.1f}% ({escape(str(fraud.get('risk_level', 'N/A')))}).",
            f"Recommendation={escape(str(verdict.get('overall_recommendation', 'MANUAL_REVIEW')))} with confidence={self._to_float(verdict.get('confidence'), 0.0):.0%}.",
        ]
        return self._format_explanation('\n'.join(lines))
