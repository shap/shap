import { render } from '@testing-library/react';
import { AdditiveForceVisualizer } from "../visualizers";
import React from 'react';

describe('AdditiveForceVisualizer', () => {
  it('renders correctly', () => {
    const { container } = render(
      <AdditiveForceVisualizer
        baseValue={0.0}
        link={"identity"}
        featureNames={{
          "0": "Blue",
          "1": "Red",
          "2": "Green",
          "3": "Orange"
        }}
        outNames={["color rating"]}
        features={{
          "0": { value: 1.0, effect: 1.0 },
          "1": { value: 0.0, effect: 0.5 },
          "2": { value: 2.0, effect: -2.5 },
          "3": { value: 2.0, effect: -0.5 }
        }}
      />
    );

    // ✅ Snapshot
    expect(container).toMatchSnapshot();

    // ✅ Check SVG exists (D3 rendering container)
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();

    // ✅ Check main container rendered
    expect(container.firstChild).not.toBeNull();
  });
});
