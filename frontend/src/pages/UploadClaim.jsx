/**
 * Upload Claim Page
 * Form for submitting a claim for verification
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { verifyClaim } from '../services/api';

const UploadClaim = () => {
  const navigate = useNavigate();
  
  // Form state
  const [clinicalDocument, setClinicalDocument] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [claimAmount, setClaimAmount] = useState('');
  const [claimId, setClaimId] = useState('');
  const [patientId, setPatientId] = useState('');
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleDocumentChange = (e) => {
    setClinicalDocument(e.target.files?.[0] || null);
  };

  const handleImageChange = (e) => {
    setImageFile(e.target.files?.[0] || null);
  };

  const validateForm = () => {
    if (!clinicalDocument) {
      setError('Clinical document is required');
      return false;
    }
    if (!imageFile) {
      setError('Medical image is required');
      return false;
    }
    if (!claimAmount || parseFloat(claimAmount) <= 0) {
      setError('Claim amount must be greater than 0');
      return false;
    }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      const result = await verifyClaim(
        clinicalDocument,
        imageFile,
        parseFloat(claimAmount),
        claimId || undefined,
        patientId || undefined
      );

      console.log('✅ Verification successful:', result);

      // Check if result has required structure
      if (!result || typeof result !== 'object') {
        throw new Error('Invalid response from server');
      }

      // Pass result data to Results page
      navigate('/results', { state: { verificationResult: result, clinicalDocument, imageFile, claimAmount, claimId, patientId } });
    } catch (err) {
      console.error('❌ Verification failed:', err);
      setError(err.message || err.error || 'Failed to verify claim. Please try again.');
      console.error('Verification error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.heading}>Claim Verification System</h1>
        <p style={styles.subtitle}>Submit a claim for AI-powered verification</p>

        {error && <div style={styles.errorBox}>{error}</div>}

        <form onSubmit={handleSubmit} style={styles.form}>
          {/* Clinical Document */}
          <div style={styles.formGroup}>
            <label style={styles.label}>Clinical Document / Discharge Summary *</label>
            <input
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={handleDocumentChange}
              style={styles.fileInput}
              disabled={loading}
            />
            {clinicalDocument && (
              <div style={styles.fileInfo}>
                ✓ Selected: {clinicalDocument.name}
              </div>
            )}
          </div>

          {/* Medical Image */}
          <div style={styles.formGroup}>
            <label style={styles.label}>Medical Image *</label>
            <input
              type="file"
              accept="image/jpeg,image/png,image/tiff,.dcm"
              onChange={handleImageChange}
              style={styles.fileInput}
              disabled={loading}
            />
            {imageFile && (
              <div style={styles.fileInfo}>
                ✓ Selected: {imageFile.name}
              </div>
            )}
          </div>

          {/* Claim Amount */}
          <div style={styles.formGroup}>
            <label style={styles.label}>Claim Amount (INR) *</label>
            <input
              type="number"
              value={claimAmount}
              onChange={(e) => setClaimAmount(e.target.value)}
              placeholder="e.g., 250000"
              style={styles.input}
              min="0"
              step="1"
              disabled={loading}
            />
          </div>

          {/* Optional: Claim ID */}
          <div style={styles.formGroup}>
            <label style={styles.label}>Claim ID (Optional)</label>
            <input
              type="text"
              value={claimId}
              onChange={(e) => setClaimId(e.target.value)}
              placeholder="e.g., CLM-2024-001"
              style={styles.input}
              disabled={loading}
            />
          </div>

          {/* Optional: Patient ID */}
          <div style={styles.formGroup}>
            <label style={styles.label}>Patient ID (Optional)</label>
            <input
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              placeholder="e.g., PAT-12345"
              style={styles.input}
              disabled={loading}
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            style={{
              ...styles.submitButton,
              ...(loading ? styles.submitButtonDisabled : {}),
            }}
            disabled={loading}
          >
            {loading ? 'Verifying Claim...' : 'Verify Claim'}
          </button>
        </form>

        <div style={styles.helpText}>
          * Required fields. The system will analyze the clinical text, medical image, and claim amount to provide AI-assisted verification.
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
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'flex-start',
  },
  card: {
    backgroundColor: 'white',
    borderRadius: '8px',
    padding: '40px',
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
    maxWidth: '600px',
    width: '100%',
  },
  heading: {
    color: '#2c3e50',
    marginBottom: '10px',
    fontSize: '28px',
    fontWeight: '700',
  },
  subtitle: {
    color: '#7f8c8d',
    marginBottom: '30px',
    fontSize: '14px',
  },
  form: {
    marginBottom: '20px',
  },
  formGroup: {
    marginBottom: '20px',
  },
  label: {
    display: 'block',
    marginBottom: '8px',
    fontWeight: '600',
    color: '#2c3e50',
    fontSize: '14px',
  },
  textarea: {
    width: '100%',
    padding: '10px',
    border: '1px solid #bdc3c7',
    borderRadius: '4px',
    fontFamily: 'inherit',
    fontSize: '14px',
    boxSizing: 'border-box',
    fontFamily: 'monospace',
  },
  input: {
    width: '100%',
    padding: '10px',
    border: '1px solid #bdc3c7',
    borderRadius: '4px',
    fontSize: '14px',
    boxSizing: 'border-box',
  },
  fileInput: {
    width: '100%',
    padding: '8px',
    border: '1px solid #bdc3c7',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  fileInfo: {
    marginTop: '8px',
    padding: '8px',
    backgroundColor: '#d5f4e6',
    color: '#27ae60',
    borderRadius: '4px',
    fontSize: '13px',
  },
  submitButton: {
    width: '100%',
    padding: '12px',
    backgroundColor: '#3498db',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '16px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.3s',
  },
  submitButtonDisabled: {
    backgroundColor: '#95a5a6',
    cursor: 'not-allowed',
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
  helpText: {
    fontSize: '12px',
    color: '#7f8c8d',
    marginTop: '15px',
    padding: '10px',
    backgroundColor: '#ecf0f1',
    borderRadius: '4px',
  },
};

export default UploadClaim;
