/**
 * Analysis page - main workflow for file upload, classification, and XAI
 */

import React from 'react';
import { useAppContext } from '../context/AppContext';
import FileUpload from '../components/Upload/FileUpload';
import ModelSelector from '../components/Selection/ModelSelector';
import ClassificationResult from '../components/Results/ClassificationResult';
import XAISelector from '../components/Selection/XAISelector';
import XAIVisualization from '../components/Results/XAIVisualization';
import styles from './Analysis.module.css';

const Analysis = () => {
  const { state } = useAppContext();

  return (
    <div className={styles.analysisPage}>
      <div className={styles.header}>
        <h1>AI Model Analysis with XAI</h1>
        <p>Upload your file, select a model, and explore explainable AI visualizations</p>
      </div>

      {/* Step 1: Upload */}
      <div className={styles.step}>
        <FileUpload />
      </div>

      {/* Step 2: Model Selection and Classification */}
      {state.fileId && (
        <div className={styles.step}>
          <ModelSelector />
        </div>
      )}

      {/* Show Classification Results */}
      {state.classificationResult && (
        <div className={styles.step}>
          <ClassificationResult />
        </div>
      )}

      {/* Step 3: XAI Method Selection */}
      {state.classificationResult && (
        <div className={styles.step}>
          <XAISelector />
        </div>
      )}

      {/* Step 4: XAI Visualization */}
      {state.xaiResult && (
        <div className={styles.step}>
          <XAIVisualization />
        </div>
      )}
    </div>
  );
};

export default Analysis;
