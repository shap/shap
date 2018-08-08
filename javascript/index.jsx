import React from 'react';
import ReactDom from 'react-dom';
import injectTapEventPlugin from 'react-tap-event-plugin';
import SimpleListVisualizer from './visualizers/SimpleListVisualizer';
import AdditiveForceVisualizer from './visualizers/AdditiveForceVisualizer';
import AdditiveForceArrayVisualizer from './visualizers/AdditiveForceArrayVisualizer';

// Needed for onTouchTap for material-ui
// http://stackoverflow.com/a/34015469/988941
injectTapEventPlugin();

// Save some globals for the inline scripts to access
window.SHAP = {
  SimpleListVisualizer: SimpleListVisualizer,
  AdditiveForceVisualizer: AdditiveForceVisualizer,
  AdditiveForceArrayVisualizer: AdditiveForceArrayVisualizer,
  React: React,
  ReactDom: ReactDom
};
