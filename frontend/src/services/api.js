/**
 * API Service
 * Handles all communication with the backend
 */

import axios from 'axios';

// Base URL for backend API
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

/**
 * Verify insurance claim
 * @param {File} clinicalDocument - Clinical document file (pdf, docx, txt)
 * @param {File} imageFile - Medical image file
 * @param {number} claimAmount - Claim amount (INR)
 * @param {string} claimId - Optional claim identifier
 * @param {string} patientId - Optional patient identifier
 * @returns {Promise} Pipeline verification result
 */
export const verifyClaim = async (clinicalDocument, imageFile, claimAmount, claimId, patientId) => {
  const formData = new FormData();
  formData.append('clinical_document', clinicalDocument);
  formData.append('image', imageFile);
  formData.append('claim_amount', claimAmount);
  
  if (claimId) {
    formData.append('claim_id', claimId);
  }
  if (patientId) {
    formData.append('patient_id', patientId);
  }

  try {
    const response = await apiClient.post('/api/verify-claim', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    console.log('API Response:', response.data);
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    const errorData = error.response?.data || {};
    throw {
      error: true,
      message: errorData.error || errorData.message || error.message || 'Failed to verify claim'
    };
  }
};

/**
 * Generate PDF report
 * @param {File} clinicalDocument - Clinical document file (pdf, docx, txt)
 * @param {File} imageFile - Medical image file
 * @param {number} claimAmount - Claim amount (INR)
 * @param {string} claimId - Optional claim identifier
 * @param {string} patientId - Optional patient identifier
 * @returns {Promise<Blob>} PDF file blob
 */
export const generateReport = async (clinicalDocument, imageFile, claimAmount, claimId, patientId) => {
  const formData = new FormData();
  formData.append('clinical_document', clinicalDocument);
  formData.append('image', imageFile);
  formData.append('claim_amount', claimAmount);
  
  if (claimId) {
    formData.append('claim_id', claimId);
  }
  if (patientId) {
    formData.append('patient_id', patientId);
  }

  try {
    const response = await apiClient.post('/generate-report', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Accept': 'application/pdf',
      },
      responseType: 'blob',
    });
    const contentDisposition = response.headers?.['content-disposition'] || '';
    const fileNameMatch = contentDisposition.match(/filename="?([^";]+)"?/i);
    const fileName = fileNameMatch?.[1] || `claim_report_${claimId || 'report'}.pdf`;

    return {
      blob: response.data,
      fileName,
      contentType: response.headers?.['content-type'] || 'application/pdf',
    };
  } catch (error) {
    if (error?.response?.data instanceof Blob) {
      try {
        const message = await error.response.data.text();
        throw { error: true, message: message || 'Failed to generate report' };
      } catch {
        throw { error: true, message: 'Failed to generate report' };
      }
    }
    throw error.response?.data || { error: true, message: 'Failed to generate report' };
  }
};

/**
 * Check API health
 * @returns {Promise} Health status
 */
export const checkHealth = async () => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    throw error.response?.data || { error: true, message: 'API is not available' };
  }
};

export default apiClient;
