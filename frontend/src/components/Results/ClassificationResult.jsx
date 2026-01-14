/**
 * ClassificationResult component showing prediction details
 */

import React from 'react';
import { useAppContext } from '../../context/AppContext';
import styles from './ClassificationResult.module.css';

const ClassificationResult = () => {
  const { state } = useAppContext();

  if (!state.classificationResult) {
    return null;
  }

  const { class: predictedClass, confidence, probabilities } = state.classificationResult;
  const confidencePercent = (confidence * 100).toFixed(2);

  return (
    <div className={styles.resultContainer}>
      <h3>Classification Result</h3>

      <div className={styles.predictionBox}>
        <div className={styles.classLabel}>
          Predicted Class: <span className={styles.className}>{predictedClass}</span>
        </div>
        <div className={styles.confidenceBar}>
          <div className={styles.confidenceLabel}>
            Confidence: <strong>{confidencePercent}%</strong>
          </div>
          <div className={styles.barBackground}>
            <div
              className={styles.barFill}
              style={{ width: `${confidencePercent}%` }}
            ></div>
          </div>
        </div>
      </div>

      <div className={styles.probabilities}>
        <h4>Class Probabilities</h4>
        {Object.entries(probabilities).map(([className, prob]) => (
          <div key={className} className={styles.probRow}>
            <span className={styles.probClass}>{className}</span>
            <div className={styles.probBar}>
              <div
                className={styles.probFill}
                style={{ width: `${(prob * 100).toFixed(1)}%` }}
              ></div>
            </div>
            <span className={styles.probValue}>{(prob * 100).toFixed(2)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ClassificationResult;
