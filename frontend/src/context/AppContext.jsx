/**
 * Global state management using React Context API
 */

import React, { createContext, useReducer, useContext } from 'react';

// Initial state
const initialState = {
  // Upload state
  uploadedFile: null,
  fileType: null, // 'audio' or 'image'
  fileId: null,
  fileName: null,
  previewUrl: null,

  // Model selection
  selectedModel: null,
  compatibleModels: [],

  // Classification results
  classificationResult: null,

  // XAI state
  selectedXAIMethod: null,
  selectedXAIMethods: [], // For comparison
  compatibleXAIMethods: [],
  xaiResult: null,
  xaiResults: {}, // Map of method name to result

  // Comparison state
  comparisonMode: false,
  comparisonResults: [],

  // UI state
  loading: false,
  error: null,
  currentStep: 1, // 1: upload, 2: model, 3: xai, 4: results
};

// Action types
const ActionTypes = {
  SET_UPLOADED_FILE: 'SET_UPLOADED_FILE',
  SET_SELECTED_MODEL: 'SET_SELECTED_MODEL',
  SET_CLASSIFICATION_RESULT: 'SET_CLASSIFICATION_RESULT',
  SET_SELECTED_XAI_METHOD: 'SET_SELECTED_XAI_METHOD',
  SET_SELECTED_XAI_METHODS: 'SET_SELECTED_XAI_METHODS',
  SET_XAI_RESULT: 'SET_XAI_RESULT',
  SET_COMPARISON_RESULTS: 'SET_COMPARISON_RESULTS',
  SET_COMPARISON_MODE: 'SET_COMPARISON_MODE',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  SET_STEP: 'SET_STEP',
  RESET: 'RESET',
};

// Reducer function
const appReducer = (state, action) => {
  switch (action.type) {
    case ActionTypes.SET_UPLOADED_FILE:
      return {
        ...state,
        uploadedFile: action.payload.uploadedFile,
        fileType: action.payload.file_type,
        fileId: action.payload.file_id,
        fileName: action.payload.filename,
        previewUrl: action.payload.preview_url,
        compatibleModels: action.payload.compatible_models,
        error: null,
      };

    case ActionTypes.SET_SELECTED_MODEL:
      return {
        ...state,
        selectedModel: action.payload,
        error: null,
      };

    case ActionTypes.SET_CLASSIFICATION_RESULT:
      return {
        ...state,
        classificationResult: action.payload.prediction,
        compatibleXAIMethods: action.payload.compatible_xai_methods,
        error: null,
      };

    case ActionTypes.SET_SELECTED_XAI_METHOD:
      return {
        ...state,
        selectedXAIMethod: action.payload,
        error: null,
      };

    case ActionTypes.SET_SELECTED_XAI_METHODS:
      return {
        ...state,
        selectedXAIMethods: action.payload,
        error: null,
      };

    case ActionTypes.SET_XAI_RESULT:
      return {
        ...state,
        xaiResult: action.payload,
        xaiResults: {
          ...state.xaiResults,
          [action.payload.xai_method]: action.payload,
        },
        error: null,
      };

    case ActionTypes.SET_COMPARISON_RESULTS:
      return {
        ...state,
        comparisonResults: action.payload,
        error: null,
      };

    case ActionTypes.SET_COMPARISON_MODE:
      return {
        ...state,
        comparisonMode: action.payload,
      };

    case ActionTypes.SET_LOADING:
      return {
        ...state,
        loading: action.payload,
      };

    case ActionTypes.SET_ERROR:
      return {
        ...state,
        error: action.payload,
        loading: false,
      };

    case ActionTypes.SET_STEP:
      return {
        ...state,
        currentStep: action.payload,
      };

    case ActionTypes.RESET:
      return initialState;

    default:
      return state;
  }
};

// Create context
const AppContext = createContext();

// Provider component
export const AppProvider = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};

// Custom hook to use the AppContext
export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};

export { ActionTypes };
export default AppContext;
