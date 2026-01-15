/**
 * API endpoint functions for making requests to the backend
 */

import apiClient from './api';

/**
 * Upload a file (audio or image)
 * @param {File} file - The file to upload
 * @returns {Promise} Response with file_id, file_type, and compatible_models
 */
export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

/**
 * Classify an uploaded file with a selected model
 * @param {string} fileId - The unique file identifier
 * @param {string} modelName - Name of the model to use
 * @returns {Promise} Response with prediction results
 */
export const classifyFile = async (fileId, modelName) => {
  const response = await apiClient.post('/api/classify', {
    file_id: fileId,
    model_name: modelName,
  });

  return response.data;
};

/**
 * Generate XAI explanation for a classification
 * @param {string} fileId - The unique file identifier
 * @param {string} modelName - Name of the model used
 * @param {string} xaiMethod - XAI method to apply (LIME, SHAP, Grad-CAM)
 * @returns {Promise} Response with XAI visualization
 */
export const explainPrediction = async (fileId, modelName, xaiMethod) => {
  const response = await apiClient.post('/api/xai/explain', {
    file_id: fileId,
    model_name: modelName,
    xai_method: xaiMethod,
  });

  return response.data;
};

/**
 * Compare multiple XAI methods side-by-side
 * @param {string} fileId - The unique file identifier
 * @param {string} modelName - Name of the model used
 * @param {string[]} xaiMethods - Array of XAI methods to compare
 * @returns {Promise} Response with multiple XAI visualizations
 */
export const compareXAIMethods = async (fileId, modelName, xaiMethods) => {
  const response = await apiClient.post('/api/xai/compare', {
    file_id: fileId,
    model_name: modelName,
    xai_methods: xaiMethods,
  });

  return response.data;
};

/**
 * Get list of available models
 * @returns {Promise} Response with audio_models and image_models
 */
export const getModels = async () => {
  const response = await apiClient.get('/api/models');
  return response.data;
};

/**
 * Get list of available XAI methods
 * @returns {Promise} Response with XAI methods and metadata
 */
export const getXAIMethods = async () => {
  const response = await apiClient.get('/api/xai/methods');
  return response.data;
};

/**
 * Get file preview URL
 * @param {string} fileId - The unique file identifier
 * @param {string} fileType - Optional file type ('audio' or 'image')
 * @returns {string} URL to access the file (spectrogram for audio)
 */
export const getFileURL = (fileId, fileType = null) => {
  const baseUrl = `http://localhost:8000/api/files/${fileId}`;
  // For audio files, request spectrogram format
  if (fileType === 'audio') {
    return `${baseUrl}?format=spectrogram`;
  }
  return baseUrl;
};
