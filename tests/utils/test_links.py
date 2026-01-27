"""Tests for the link functions in shap.links module."""

import numpy as np

from shap.links import identity, logit


class TestLogitFunction:
    """Tests for the logit link function."""

    def test_logit_normal_range(self):
        """Test logit works correctly for normal probability range."""
        # Test values in (0, 1)
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = logit(probs)

        # All results should be finite
        assert np.all(np.isfinite(result))

        # logit(0.5) should be 0
        np.testing.assert_almost_equal(logit(0.5), 0.0)

        # logit should be symmetric around 0.5
        np.testing.assert_almost_equal(logit(0.3), -logit(0.7))

    def test_logit_edge_cases_no_inf(self):
        """Test that logit doesn't produce inf/nan for edge cases (issue #4045)."""
        # Test exact 0 and 1
        probs = np.array([0.0, 1.0])
        result = logit(probs)

        # Should not produce inf or nan
        assert not np.any(np.isinf(result)), "logit should not produce inf values"
        assert not np.any(np.isnan(result)), "logit should not produce nan values"
        assert np.all(np.isfinite(result)), "All logit values should be finite"

    def test_logit_very_close_to_boundaries(self):
        """Test logit handles probabilities very close to 0 and 1."""
        # Test values very close to boundaries
        probs = np.array([1e-20, 1e-15, 1e-10, 1 - 1e-10, 1 - 1e-15, 1 - 1e-20])
        result = logit(probs)

        # All should be finite
        assert np.all(np.isfinite(result))

    def test_logit_scalar_values(self):
        """Test logit works with scalar inputs."""
        # Test scalar inputs
        assert np.isfinite(logit(0.0)), "logit(0.0) should be finite"
        assert np.isfinite(logit(0.5)), "logit(0.5) should be finite"
        assert np.isfinite(logit(1.0)), "logit(1.0) should be finite"

        # logit(0.5) should be exactly 0
        np.testing.assert_almost_equal(logit(0.5), 0.0)

    def test_logit_downstream_computations(self):
        """Test that logit values work in downstream numerical computations."""
        # Test that results can be used in further computations
        probs = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        logit_vals = logit(probs)

        # Mean and std should be computable and finite
        mean_val = np.mean(logit_vals)
        std_val = np.std(logit_vals)

        assert np.isfinite(mean_val), "Mean of logit values should be finite"
        assert np.isfinite(std_val), "Std of logit values should be finite"

    def test_logit_inverse_composition(self):
        """Test that logit and its inverse compose correctly for safe values."""
        # Test for values in safe range (not too close to 0 or 1)
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Apply logit then inverse
        logit_vals = logit(probs)
        recovered = logit.inverse(logit_vals)  # type: ignore[attr-defined]

        # Should recover original probabilities
        np.testing.assert_allclose(recovered, probs, rtol=1e-10)


class TestIdentityFunction:
    """Tests for the identity link function."""

    def test_identity_returns_input(self):
        """Test identity function returns its input unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        result = identity(x)
        np.testing.assert_array_equal(result, x)

    def test_identity_inverse(self):
        """Test identity inverse is also identity."""
        x = np.array([1.0, 2.0, 3.0])
        result = identity.inverse(x)  # type: ignore[attr-defined]
        np.testing.assert_array_equal(result, x)
