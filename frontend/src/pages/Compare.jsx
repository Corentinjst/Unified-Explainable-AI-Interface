/**
 * Compare page - side-by-side comparison of multiple XAI methods
 */

import React, { useState } from 'react';
import { useAppContext, ActionTypes } from '../context/AppContext';
import { compareXAIMethods } from '../services/endpoints';
import { getFileURL } from '../services/endpoints';
import styles from './Compare.module.css';

const Compare = () => {
  const { state, dispatch } = useAppContext();
  const [selectedMethods, setSelectedMethods] = useState([]);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState(null);

  // Check if we have the necessary data to compare
  const canCompare = state.fileId && state.selectedModel && state.classificationResult;

  const handleMethodToggle = (methodName) => {
    setSelectedMethods((prev) => {
      if (prev.includes(methodName)) {
        return prev.filter((m) => m !== methodName);
      } else {
        return [...prev, methodName];
      }
    });
  };

  const handleCompare = async () => {
    if (selectedMethods.length < 2) {
      setError('Please select at least 2 XAI methods to compare');
      return;
    }

    setComparing(true);
    setError(null);

    try {
      const response = await compareXAIMethods(
        state.fileId,
        state.selectedModel.name,
        selectedMethods
      );

      dispatch({
        type: ActionTypes.SET_COMPARISON_RESULTS,
        payload: response.comparisons,
      });
      dispatch({
        type: ActionTypes.SET_COMPARISON_MODE,
        payload: true,
      });
    } catch (err) {
      console.error('Comparison failed:', err);
      setError(err.response?.data?.detail || 'Failed to compare XAI methods');
    } finally {
      setComparing(false);
    }
  };

  const handleReset = () => {
    setSelectedMethods([]);
    dispatch({
      type: ActionTypes.SET_COMPARISON_RESULTS,
      payload: [],
    });
    dispatch({
      type: ActionTypes.SET_COMPARISON_MODE,
      payload: false,
    });
    setError(null);
  };

  if (!canCompare) {
    return (
      <div className={styles.comparePage}>
        <div className={styles.emptyState}>
          <h2>No Classification Available</h2>
          <p>Please upload a file and run classification first from the Analysis page.</p>
          <a href="/" className={styles.linkButton}>Go to Analysis</a>
        </div>
      </div>
    );
  }

  const originalImageUrl = getFileURL(state.fileId, state.fileType);

  return (
    <div className={styles.comparePage}>
      <div className={styles.header}>
        <h1>Compare XAI Methods</h1>
        <p>Select multiple explainability techniques to compare side-by-side</p>
      </div>

      {/* File and Model Info */}
      <div className={styles.infoCard}>
        <div className={styles.infoRow}>
          <span className={styles.infoLabel}>File:</span>
          <span className={styles.infoValue}>{state.fileName}</span>
        </div>
        <div className={styles.infoRow}>
          <span className={styles.infoLabel}>Type:</span>
          <span className={styles.infoValue}>{state.fileType}</span>
        </div>
        <div className={styles.infoRow}>
          <span className={styles.infoLabel}>Model:</span>
          <span className={styles.infoValue}>{state.selectedModel.name}</span>
        </div>
        <div className={styles.infoRow}>
          <span className={styles.infoLabel}>Prediction:</span>
          <span className={styles.infoValue}>
            {state.classificationResult.class_name} ({(state.classificationResult.confidence * 100).toFixed(2)}%)
          </span>
        </div>
      </div>

      {/* XAI Method Selection */}
      {!state.comparisonMode && (
        <div className={styles.selectionSection}>
          <h2>Select XAI Methods to Compare</h2>
          <p className={styles.subtitle}>Choose at least 2 methods</p>

          <div className={styles.methodsGrid}>
            {state.compatibleXAIMethods.map((method) => (
              <div
                key={method.name}
                className={`${styles.methodCard} ${
                  selectedMethods.includes(method.name) ? styles.selected : ''
                }`}
                onClick={() => handleMethodToggle(method.name)}
              >
                <input
                  type="checkbox"
                  checked={selectedMethods.includes(method.name)}
                  onChange={() => {}}
                  className={styles.checkbox}
                />
                <div className={styles.methodContent}>
                  <h3>{method.name}</h3>
                  <p className={styles.fullName}>{method.full_name}</p>
                  <p className={styles.description}>{method.description}</p>
                </div>
              </div>
            ))}
          </div>

          <div className={styles.actions}>
            <button
              className={styles.compareButton}
              onClick={handleCompare}
              disabled={comparing || selectedMethods.length < 2}
            >
              {comparing ? (
                <>
                  <div className={styles.spinner}></div>
                  Comparing...
                </>
              ) : (
                `Compare ${selectedMethods.length} Method${selectedMethods.length !== 1 ? 's' : ''}`
              )}
            </button>
          </div>

          {error && (
            <div className={styles.errorMessage}>
              <span>⚠️</span> {error}
            </div>
          )}
        </div>
      )}

      {/* Comparison Results */}
      {state.comparisonMode && state.comparisonResults.length > 0 && (
        <div className={styles.resultsSection}>
          <div className={styles.resultsHeader}>
            <h2>Comparison Results</h2>
            <button className={styles.resetButton} onClick={handleReset}>
              New Comparison
            </button>
          </div>

          {/* Original Image */}
          <div className={styles.originalSection}>
            <h3>Original {state.fileType === 'audio' ? 'Spectrogram' : 'Image'}</h3>
            <div className={styles.originalImageContainer}>
              <img src={originalImageUrl} alt="Original" className={styles.originalImage} />
            </div>
          </div>

          {/* Side-by-side comparisons */}
          <div className={styles.comparisonsGrid}>
            {state.comparisonResults.map((result) => (
              <div key={result.xai_method} className={styles.comparisonCard}>
                <div className={styles.comparisonHeader}>
                  <h3>{result.xai_method}</h3>
                </div>
                <div className={styles.visualizationContainer}>
                  <img
                    src={result.visualization}
                    alt={`${result.xai_method} explanation`}
                    className={styles.visualizationImage}
                  />
                </div>
                <div className={styles.metadataSection}>
                  <h4>Metadata</h4>
                  <div className={styles.metadataItems}>
                    {Object.entries(result.metadata)
                      .filter(([key]) => !['file_type', 'model_name', 'top_features'].includes(key))
                      .map(([key, value]) => (
                        <div key={key} className={styles.metadataItem}>
                          <span className={styles.metadataKey}>
                            {key.replace(/_/g, ' ')}:
                          </span>
                          <span className={styles.metadataValue}>
                            {typeof value === 'number'
                              ? value.toFixed(4)
                              : typeof value === 'object'
                              ? JSON.stringify(value)
                              : String(value)}
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Compare;
