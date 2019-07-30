import React from 'react';
import ReactDom from 'react-dom';
import injectTapEventPlugin from 'react-tap-event-plugin';
import Main from './Main';

// Needed for onTouchTap for material-ui
// http://stackoverflow.com/a/34015469/988941
injectTapEventPlugin();

// Save some globals for the inline scripts to access
window.IML = {
  Main: Main,
  React: React,
  ReactDom: ReactDom
};
