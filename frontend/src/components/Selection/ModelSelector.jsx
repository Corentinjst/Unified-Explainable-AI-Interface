/**
 * ModelSelector component for choosing a classification model
 */

import React, { useState } from 'react';
import { useAppContext, ActionTypes } from '../../context/AppContext';
import { classifyFile } from '../../services/endpoints';
import styles from './ModelSelector.module.css';

const ModelSelector = () => {
  const { state, dispatch } = useAppContext();
  const [classifying, setClassifying] = useState(false);

  if (!state.fileId) {
    return null;
  }

  const handleModelSelect = async (model) => {
    setClassifying(true);
    dispatch({ type: ActionTypes.SET_LOADING, payload: true });
    dispatch({ type: ActionTypes.SET_ERROR, payload: null });
    dispatch({ type: ActionTypes.SET_SELECTED_MODEL, payload: model });

    try {
      const response = await classifyFile(state.fileId, model.name);

      dispatch({
        type: ActionTypes.SET_CLASSIFICATION_RESULT,
        payload: response,
      });

      dispatch({ type: ActionTypes.SET_STEP, payload: 3 });
    } catch (error) {
      console.error('Classification failed:', error);
      dispatch({
        type: ActionTypes.SET_ERROR,
        payload: error.response?.data?.detail || 'Classification failed. Please try again.',
      });
    } finally {
      setClassifying(false);
      dispatch({ type: ActionTypes.SET_LOADING, payload: false });
    }
  };

  return (
    <div className={styles.selectorContainer}>
      <h2>Select Classification Model</h2>
      <p className={styles.subtitle}>
        Choose a model compatible with your {state.fileType} file
      </p>

      <div className={styles.modelGrid}>
        {state.compatibleModels.map((model) => (
          <div
            key={model.name}
            className={`${styles.modelCard} ${
              state.selectedModel?.name === model.name ? styles.selected : ''
            } ${classifying ? styles.disabled : ''}`}
            onClick={() => !classifying && handleModelSelect(model)}
          >
            <div className={styles.modelHeader}>
              <h3>{model.name}</h3>
              <span className={styles.accuracy}>
                {(model.accuracy * 100).toFixed(1)}% accuracy
              </span>
            </div>
            <p className={styles.modelDescription}>{model.description}</p>
            <div className={styles.modelFooter}>
              <span className={styles.modelType}>
                {model.type === 'audio' ? '🎵 Audio' : '🖼️ Image'}
              </span>
            </div>
          </div>
        ))}
      </div>

      {classifying && (
        <div className={styles.classifyingMessage}>
          <div className={styles.spinner}></div>
          <p>Running classification...</p>
        </div>
      )}

      {state.error && (
        <div className={styles.errorMessage}>
          <span>⚠️</span> {state.error}
        </div>
      )}
    </div>
  );
};

export default ModelSelector;
