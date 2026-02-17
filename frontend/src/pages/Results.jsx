/**
 * Results Page
 * Displays claim verification results
 */

import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { generateReport } from '../services/api';
import ResultCard from '../components/ResultCard';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  
  const state = location.state || {};
  const result = state.verificationResult || {};
  const { clinicalDocument, imageFile, claimAmount, claimId, patientId } = state;

  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState('');

  // Extract data from pipeline output
  const metadata = result.metadata || {};
  const icdVerification = result.icd_verification || {};
  const imageAnalysis = result.image_analysis || {};
  const fraudAssessment = result.fraud_assessment || {};
  const integratedVerdict = result.integrated_verdict || {};
  const explanation = result.explanation || '';

  const handleDownloadPDF = async () => {
    setDownloading(true);
    setDownloadError('');

    try {
      const pdfBlob = await generateReport(
        clinicalDocument,
        imageFile,
        claimAmount,
        claimId,
        patientId
      );

      // Create download link
      const url = window.URL.createObjectURL(pdfBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `claim_report_${claimId || 'report'}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setDownloadError('Failed to download PDF report. Please try again.');
      console.error('PDF download error:', err);
    } finally {
      setDownloading(false);
    }
  };

  const handleBackToForm = () => {
    navigate('/');
  };

  // Status badge component
  const StatusBadge = ({ status }) => {
    let bgColor = '#ecf0f1';
    let textColor = '#7f8c8d';

    if (status) {
      const lower = status.toLowerCase();
      if (lower === 'valid' || lower === 'authentic' || lower === 'low' || lower === 'approve') {
        bgColor = '#d5f4e6';
        textColor = '#27ae60';
      } else if (lower === 'mismatch' || lower === 'suspicious' || lower === 'medium' || lower === 'manual_review') {
        bgColor = '#fef5e7';
        textColor = '#f39c12';
      } else if (lower === 'error' || lower === 'tampered' || lower === 'high' || lower === 'deny') {
        bgColor = '#fadbd8';
        textColor = '#c0392b';
      }
    }

    return (
      <span style={{
        display: 'inline-block',
        padding: '6px 12px',
        backgroundColor: bgColor,
        color: textColor,
        borderRadius: '4px',
        fontWeight: '600',
        fontSize: '13px',
      }}>
        {status || 'N/A'}
      </span>
    );
  };

  // Row component for displaying key-value pairs
  const Row = ({ label, value, badge = false }) => (
    <div style={styles.row}>
      <span style={styles.rowLabel}>{label}:</span>
      <span style={styles.rowValue}>
        {badge ? <StatusBadge status={value} /> : value || 'N/A'}
      </span>
    </div>
  );

  return (
    <div style={styles.container}>
      <div style={styles.content}>
        {/* Header */}
        <div style={styles.header}>
          <h1 style={styles.heading}>Claim Verification Results</h1>
          <p style={styles.subtitle}>AI-Powered Claim Assessment</p>
        </div>

        {downloadError && <div style={styles.errorBox}>{downloadError}</div>}

        {/* Integrated Verdict */}
        <ResultCard title="Overall Verdict">
          <div style={styles.verdictBox}>
            <div style={styles.verdictTitle}>Recommendation:</div>
            <StatusBadge status={integratedVerdict.overall_recommendation} />
          </div>
          {integratedVerdict.confidence !== undefined && integratedVerdict.confidence !== null && (
            <Row label="Confidence" value={`${(integratedVerdict.confidence * 100).toFixed(0)}%`} />
          )}
          {integratedVerdict.risk_summary && (
            <Row label="Summary" value={integratedVerdict.risk_summary} />
          )}
        </ResultCard>

        {/* ICD Code Verification */}
        <ResultCard title="ICD Code Verification">
          <Row label="Status" value={icdVerification.status} badge={true} />
          {icdVerification.match_score !== undefined && icdVerification.match_score !== null && (
            <Row
              label="Match Score"
              value={`${(icdVerification.match_score * 100).toFixed(0)}%`}
            />
          )}
          {icdVerification.num_icds_detected !== undefined && (
            <Row label="Codes Detected" value={icdVerification.num_icds_detected} />
          )}
          {icdVerification.predicted_icds && icdVerification.predicted_icds.length > 0 && (
            <div style={styles.row}>
              <span style={styles.rowLabel}>Predicted Codes:</span>
              <div style={styles.codesList}>
                {icdVerification.predicted_icds.slice(0, 5).map((code, idx) => (
                  <div key={idx} style={styles.codeItem}>
                    {code[0]} ({(code[1] * 100).toFixed(0)}%)
                  </div>
                ))}
              </div>
            </div>
          )}
        </ResultCard>

        {/* Image Forgery Analysis */}
        <ResultCard title="Medical Image Analysis">
          <Row label="Forgery Verdict" value={imageAnalysis.forgery_verdict} badge={true} />
          {imageAnalysis.cnn_score !== undefined && imageAnalysis.cnn_score !== null && (
            <Row label="CNN Score" value={imageAnalysis.cnn_score.toFixed(3)} />
          )}
          {imageAnalysis.ela_score !== undefined && imageAnalysis.ela_score !== null && (
            <Row label="ELA Score" value={imageAnalysis.ela_score.toFixed(3)} />
          )}
          {imageAnalysis.phash_score !== undefined && imageAnalysis.phash_score !== null && (
            <Row label="pHash Score" value={imageAnalysis.phash_score.toFixed(3)} />
          )}
          {imageAnalysis.fused_score !== undefined && imageAnalysis.fused_score !== null && (
            <Row label="Fused Score" value={imageAnalysis.fused_score.toFixed(3)} />
          )}
        </ResultCard>

        {/* Fraud Risk Assessment */}
        <ResultCard title="Fraud Risk Assessment">
          <Row label="Risk Level" value={fraudAssessment.risk_level} badge={true} />
          {fraudAssessment.fraud_risk_percentage !== undefined && fraudAssessment.fraud_risk_percentage !== null && (
            <>
              <Row
                label="Fraud Risk"
                value={`${fraudAssessment.fraud_risk_percentage.toFixed(1)}%`}
              />
              {/* Visual indicator */}
              <div style={styles.progressBar}>
                <div
                  style={{
                    ...styles.progressFill,
                    width: `${Math.min(fraudAssessment.fraud_risk_percentage, 100)}%`,
                  }}
                />
              </div>
            </>
          )}
          {fraudAssessment.recommendation && (
            <Row label="Recommendation" value={fraudAssessment.recommendation} badge={true} />
          )}
          {fraudAssessment.risk_factors && fraudAssessment.risk_factors.length > 0 && (
            <div style={styles.row}>
              <span style={styles.rowLabel}>Risk Factors:</span>
              <div style={styles.riskFactors}>
                {fraudAssessment.risk_factors.map((factor, idx) => (
                  <div key={idx} style={styles.riskFactor}>
                    • {factor}
                  </div>
                ))}
              </div>
            </div>
          )}
        </ResultCard>

        {/* Claim Metadata */}
        {Object.keys(metadata).length > 0 && (
          <ResultCard title="Claim Information">
            {metadata.claim_id && <Row label="Claim ID" value={metadata.claim_id} />}
            {metadata.patient_id && <Row label="Patient ID" value={metadata.patient_id} />}
            {metadata.claim_amount !== undefined && metadata.claim_amount !== null && (
              <Row label="Claim Amount" value={`$${metadata.claim_amount.toFixed(2)}`} />
            )}
            {metadata.timestamp && <Row label="Verified At" value={metadata.timestamp} />}
          </ResultCard>
        )}

        {/* Explanation */}
        {explanation && (
          <ResultCard title="Detailed Assessment">
            <div style={styles.explanationText}>{explanation}</div>
          </ResultCard>
        )}

        {/* Action Buttons */}
        <div style={styles.actions}>
          <button
            onClick={handleDownloadPDF}
            style={{
              ...styles.primaryButton,
              ...(downloading ? styles.buttonDisabled : {}),
            }}
            disabled={downloading}
          >
            {downloading ? 'Generating PDF...' : 'Download PDF Report'}
          </button>
          <button onClick={handleBackToForm} style={styles.secondaryButton}>
            ← Back to Form
          </button>
        </div>

        {/* Disclaimer */}
        <div style={styles.disclaimer}>
          <strong>⚠️ Disclaimer:</strong> This report is generated using an AI-assisted decision support system.
          All recommendations should be reviewed by qualified insurance professionals before final approval.
          The system assists in fraud detection and should not be the sole basis for claim decisions.
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#f5f5f5',
    padding: '20px',
  },
  content: {
    maxWidth: '800px',
    margin: '0 auto',
  },
  header: {
    backgroundColor: 'white',
    borderRadius: '8px',
    padding: '30px',
    marginBottom: '30px',
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
    textAlign: 'center',
  },
  heading: {
    color: '#2c3e50',
    margin: '0 0 10px 0',
    fontSize: '28px',
    fontWeight: '700',
  },
  subtitle: {
    color: '#7f8c8d',
    margin: '0',
    fontSize: '14px',
  },
  row: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '10px 0',
    borderBottom: '1px solid #ecf0f1',
  },
  rowLabel: {
    fontWeight: '600',
    color: '#2c3e50',
    flex: '0 0 200px',
  },
  rowValue: {
    color: '#555',
    flex: '1',
    textAlign: 'right',
  },
  verdictBox: {
    padding: '15px',
    backgroundColor: '#ecf0f1',
    borderRadius: '4px',
    marginBottom: '15px',
  },
  verdictTitle: {
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: '8px',
  },
  codesList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px',
    textAlign: 'right',
  },
  codeItem: {
    padding: '5px 10px',
    backgroundColor: '#f0f0f0',
    borderRadius: '3px',
    fontSize: '13px',
    fontFamily: 'monospace',
  },
  progressBar: {
    width: '100%',
    height: '20px',
    backgroundColor: '#ecf0f1',
    borderRadius: '3px',
    marginTop: '10px',
    marginBottom: '10px',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    background: 'linear-gradient(to right, #e74c3c, #f39c12, #2ecc71)',
    transition: 'width 0.3s',
  },
  riskFactors: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    textAlign: 'right',
  },
  riskFactor: {
    padding: '8px',
    backgroundColor: '#ffe8e8',
    borderLeft: '3px solid #e74c3c',
    borderRadius: '3px',
    fontSize: '13px',
  },
  explanationText: {
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    lineHeight: '1.6',
    fontSize: '13px',
  },
  actions: {
    display: 'flex',
    gap: '10px',
    marginTop: '30px',
    marginBottom: '30px',
  },
  primaryButton: {
    flex: 1,
    padding: '12px',
    backgroundColor: '#27ae60',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.3s',
  },
  secondaryButton: {
    flex: 1,
    padding: '12px',
    backgroundColor: '#95a5a6',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.3s',
  },
  buttonDisabled: {
    backgroundColor: '#bdc3c7',
    cursor: 'not-allowed',
  },
  disclaimer: {
    backgroundColor: '#fff3cd',
    border: '1px solid #ffc107',
    padding: '15px',
    borderRadius: '4px',
    fontSize: '12px',
    color: '#856404',
    marginTop: '20px',
  },
  errorBox: {
    backgroundColor: '#fadbd8',
    color: '#c0392b',
    padding: '12px',
    borderRadius: '4px',
    marginBottom: '20px',
    fontSize: '14px',
    border: '1px solid #f5b7b1',
  },
};

export default Results;
