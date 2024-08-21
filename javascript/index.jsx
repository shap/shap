import React from "react";
import ReactDom from "react-dom";
import {
  SimpleListVisualizer,
  AdditiveForceVisualizer,
  AdditiveForceArrayVisualizer
} from "./visualizers";

// Save some globals for the inline scripts to access
window.SHAP = {
  SimpleListVisualizer,
  AdditiveForceVisualizer,
  AdditiveForceArrayVisualizer,
  React: React,
  ReactDom: ReactDom
};
