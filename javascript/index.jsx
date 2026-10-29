import React from "react";
import * as ReactDOM from "react-dom/client";
import {
  SimpleListVisualizer,
  AdditiveForceVisualizer,
  AdditiveForceArrayVisualizer
} from "./visualizers";

// Store roots to avoid recreating them for the same container
const roots = new Map();

// Save globals for inline scripts
window.SHAP = {
  SimpleListVisualizer,
  AdditiveForceVisualizer,
  AdditiveForceArrayVisualizer,
  React: React,
  ReactDOM: ReactDOM,

  // Backward compatibility render method
  ReactDom: {
    render: (element, container) => {
      if (!container) {
        console.error("Container is null or undefined");
        return null;
      }

      let root = roots.get(container);

      // Create root only once per container
      if (!root) {
        root = ReactDOM.createRoot(container);
        roots.set(container, root);
      }

      root.render(element);
      return root;
    }
  }
};
