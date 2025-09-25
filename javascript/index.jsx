import React from "react";
import * as ReactDOM from "react-dom/client";
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
  ReactDOM: ReactDOM,
  // Provide backward compatibility for render method
  ReactDom: {
    render: (element, container) => {
      const root = ReactDOM.createRoot(container);
      root.render(element);
      return root;
    }
  }
};
