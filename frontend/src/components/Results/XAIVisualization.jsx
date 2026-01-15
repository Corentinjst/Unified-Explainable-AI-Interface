/**
 * XAIVisualization component showing side-by-side comparison
 */

import React from 'react';
import { useAppContext } from '../../context/AppContext';
import { getFileURL } from '../../services/endpoints';
import styles from './XAIVisualization.module.css';

const XAIVisualization = () => {
  const { state } = useAppContext();

  if (!state.xaiResult) {
    return null;
  }

  const { xai_method, visualization, metadata, processing_time_ms } = state.xaiResult;
  const originalImageUrl = getFileURL(state.fileId, state.fileType);

  return (
    <div className={styles.visualizationContainer}>
      <h2>XAI Explanation: {xai_method}</h2>
      <p className={styles.processingTime}>
        Processing time: {processing_time_ms.toFixed(2)} ms
      </p>

      <div className={styles.imagesContainer}>
        <div className={styles.imageBox}>
          <h3>Original {state.fileType === 'audio' ? 'Spectrogram' : 'Image'}</h3>
          <img src={originalImageUrl} alt="Original" className={styles.image} />
        </div>

        <div className={styles.imageBox}>
          <h3>{xai_method} Explanation</h3>
          <img src={visualization} alt={`${xai_method} visualization`} className={styles.image} />
        </div>
      </div>

      <div className={styles.metadata}>
        <h4>Explanation Metadata</h4>
        <div className={styles.metadataGrid}>
          {Object.entries(metadata).map(([key, value]) => (
            <div key={key} className={styles.metadataItem}>
              <span className={styles.metadataKey}>{key.replace(/_/g, ' ')}:</span>
              <span className={styles.metadataValue}>
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default XAIVisualization;
