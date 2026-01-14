/**
 * FileUpload component with drag-and-drop functionality
 */

import React, { useState } from 'react';
import { useAppContext, ActionTypes } from '../../context/AppContext';
import { uploadFile } from '../../services/endpoints';
import styles from './FileUpload.module.css';

const FileUpload = () => {
  const { state, dispatch } = useAppContext();
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      await handleUpload(file);
    }
  };

  const handleFileChange = async (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      await handleUpload(file);
    }
  };

  const handleUpload = async (file) => {
    // Validate file type
    const validExtensions = ['.wav', '.jpg', '.jpeg', '.png'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(fileExtension)) {
      dispatch({
        type: ActionTypes.SET_ERROR,
        payload: 'Invalid file type. Please upload .wav (audio) or .jpg/.png (image)',
      });
      return;
    }

    setUploading(true);
    dispatch({ type: ActionTypes.SET_LOADING, payload: true });
    dispatch({ type: ActionTypes.SET_ERROR, payload: null });

    try {
      const response = await uploadFile(file);

      dispatch({
        type: ActionTypes.SET_UPLOADED_FILE,
        payload: {
          uploadedFile: file,
          ...response,
        },
      });

      dispatch({ type: ActionTypes.SET_STEP, payload: 2 });
    } catch (error) {
      console.error('Upload failed:', error);
      dispatch({
        type: ActionTypes.SET_ERROR,
        payload: error.response?.data?.detail || 'Upload failed. Please try again.',
      });
    } finally {
      setUploading(false);
      dispatch({ type: ActionTypes.SET_LOADING, payload: false });
    }
  };

  const handleReset = () => {
    dispatch({ type: ActionTypes.RESET });
  };

  if (state.fileId) {
    // File already uploaded, show preview
    return (
      <div className={styles.uploadedContainer}>
        <div className={styles.uploadSuccess}>
          <div className={styles.successIcon}>✓</div>
          <h3>File Uploaded Successfully!</h3>
          <p className={styles.fileName}>{state.fileName}</p>
          <p className={styles.fileType}>
            Type: <strong>{state.fileType === 'audio' ? 'Audio' : 'Image'}</strong>
          </p>
          <button onClick={handleReset} className={styles.resetButton}>
            Upload Different File
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.uploadContainer}>
      <h2>Upload File for Analysis</h2>
      <p className={styles.subtitle}>
        Upload audio (.wav) for deepfake detection or images (.jpg, .png) for lung cancer detection
      </p>

      <div
        className={`${styles.dropzone} ${dragActive ? styles.dragActive : ''} ${
          uploading ? styles.uploading : ''
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {uploading ? (
          <div className={styles.loadingContainer}>
            <div className={styles.spinner}></div>
            <p>Uploading and validating file...</p>
          </div>
        ) : (
          <>
            <div className={styles.uploadIcon}>📁</div>
            <p className={styles.dragText}>Drag & drop your file here</p>
            <p className={styles.orText}>or</p>
            <label className={styles.browseButton}>
              <input
                type="file"
                accept=".wav,.jpg,.jpeg,.png"
                onChange={handleFileChange}
                className={styles.fileInput}
              />
              Browse Files
            </label>
            <p className={styles.formats}>
              Supported formats: <strong>.wav</strong> (audio) | <strong>.jpg, .png</strong> (images)
            </p>
            <p className={styles.sizeLimit}>Max size: 50MB (audio), 10MB (images)</p>
          </>
        )}
      </div>

      {state.error && (
        <div className={styles.errorMessage}>
          <span className={styles.errorIcon}>⚠️</span>
          {state.error}
        </div>
      )}
    </div>
  );
};

export default FileUpload;
