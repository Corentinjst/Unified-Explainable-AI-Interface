/**
 * AudioPlayer component for playing WAV audio files
 */

import React from 'react';
import styles from './AudioPlayer.module.css';

const AudioPlayer = ({ audioUrl, fileName }) => {
  return (
    <div className={styles.audioPlayerContainer}>
      <div className={styles.audioInfo}>
        <span className={styles.audioIcon}>🔊</span>
        <span className={styles.fileName}>{fileName || 'Audio File'}</span>
      </div>
      <audio
        controls
        className={styles.audioControl}
        preload="metadata"
      >
        <source src={audioUrl} type="audio/wav" />
        Your browser does not support the audio element.
      </audio>
    </div>
  );
};

export default AudioPlayer;
