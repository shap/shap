import { render, screen } from '@testing-library/react';
import { AdditiveForceVisualizer } from "../visualizers";
import React from 'react';

describe('AdditiveForceVisualizer', () => {
  it('renders correctly', () => {
    // Render the component
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

    // Compare with snapshot
    expect(container).toMatchSnapshot();
  });
});
