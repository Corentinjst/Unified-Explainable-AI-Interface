/**
 * XAISelector component for choosing XAI explanation method
 */

import React, { useState } from 'react';
import { useAppContext, ActionTypes } from '../../context/AppContext';
import { explainPrediction } from '../../services/endpoints';
import styles from './XAISelector.module.css';

const XAISelector = () => {
  const { state, dispatch } = useAppContext();
  const [explaining, setExplaining] = useState(false);

  if (!state.classificationResult) {
    return null;
  }

  const handleXAISelect = async (method) => {
    setExplaining(true);
    dispatch({ type: ActionTypes.SET_LOADING, payload: true });
    dispatch({ type: ActionTypes.SET_ERROR, payload: null });
    dispatch({ type: ActionTypes.SET_SELECTED_XAI_METHOD, payload: method.name });

    try {
      const response = await explainPrediction(
        state.fileId,
        state.selectedModel.name,
        method.name
      );

      dispatch({
        type: ActionTypes.SET_XAI_RESULT,
        payload: response,
      });

      dispatch({ type: ActionTypes.SET_STEP, payload: 4 });
    } catch (error) {
      console.error('XAI explanation failed:', error);
      dispatch({
        type: ActionTypes.SET_ERROR,
        payload: error.response?.data?.detail || 'Failed to generate explanation.',
      });
    } finally {
      setExplaining(false);
      dispatch({ type: ActionTypes.SET_LOADING, payload: false });
    }
  };

  return (
    <div className={styles.selectorContainer}>
      <h2>Select XAI Method</h2>
      <p className={styles.subtitle}>
        Choose an explainability technique to understand the model's decision
      </p>

      <div className={styles.xaiGrid}>
        {state.compatibleXAIMethods.map((method) => (
          <div
            key={method.name}
            className={`${styles.xaiCard} ${
              state.selectedXAIMethod === method.name ? styles.selected : ''
            } ${explaining ? styles.disabled : ''}`}
            onClick={() => !explaining && handleXAISelect(method)}
          >
            <h3>{method.name}</h3>
            <p className={styles.fullName}>{method.full_name}</p>
            <p className={styles.description}>{method.description}</p>
            <span className={styles.colorScheme}>{method.color_scheme}</span>
          </div>
        ))}
      </div>

      {explaining && (
        <div className={styles.explainingMessage}>
          <div className={styles.spinner}></div>
          <p>Generating explanation...</p>
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

export default XAISelector;
