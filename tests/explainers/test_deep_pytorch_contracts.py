"""Small, deterministic checks for the PyTorch backend's attribution shapes."""

import numpy as np
import pytest

import shap

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("ranked_outputs", [None, 1, 2])
@pytest.mark.parametrize("order", ["max", "min", "max_abs"])
def test_linear_attributions_follow_ranked_output_order(ranked_outputs, order):
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0], [-3.0, 4.0]]))
        model.bias.copy_(torch.tensor([1.0, -2.0]))
    background = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    samples = torch.tensor([[3.0, 1.0], [-1.0, 2.0]])
    explainer = shap.DeepExplainer(model, background)

    result = explainer.shap_values(
        samples,
        ranked_outputs=ranked_outputs,
        output_rank_order=order,
        check_additivity=ranked_outputs is None,
    )
    expected = (samples - background.mean(0)).numpy()[:, :, None] * model.weight.detach().numpy().T
    if ranked_outputs is None:
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected)
    else:
        values, ranks = result
        outputs = model(samples).detach()
        sorted_values = outputs.abs() if order == "max_abs" else outputs
        expected_ranks = torch.argsort(sorted_values, dim=1, descending=order != "min")[:, :ranked_outputs]
        torch.testing.assert_close(ranks, expected_ranks)
        assert values.shape == (2, 2, ranked_outputs)
        np.testing.assert_allclose(values, np.take_along_axis(expected, ranks.numpy()[:, None, :], axis=2))
        np.testing.assert_allclose(
            values.sum(axis=1) + explainer.expected_value[ranks.numpy()],
            outputs.gather(1, ranks).numpy(),
        )


def test_multiple_input_attributions_remain_separate_arrays():
    class TwoInputModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(2, 1, bias=False)
            self.right = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.left.weight.copy_(torch.tensor([[2.0, -1.0]]))
                self.right.weight.copy_(torch.tensor([[3.0]]))

        def forward(self, left, right):
            return self.left(left) + self.right(right)

    model = TwoInputModel()
    background = [torch.tensor([[0.0, 0.0], [2.0, 4.0]]), torch.tensor([[1.0], [3.0]])]
    samples = [torch.tensor([[3.0, 1.0], [-1.0, 2.0]]), torch.tensor([[4.0], [-2.0]])]
    explainer = shap.DeepExplainer(model, background)

    values = explainer.shap_values(samples)

    assert isinstance(values, list)
    assert [value.shape for value in values] == [(2, 2, 1), (2, 1, 1)]
    for value, sample, reference, layer in zip(values, samples, background, (model.left, model.right)):
        expected = (sample - reference.mean(0)) * layer.weight.detach()
        np.testing.assert_allclose(value[..., 0], expected.numpy())
    total = sum(value.sum(axis=1) for value in values)
    np.testing.assert_allclose(total + explainer.expected_value, model(*samples).detach().numpy())


def test_interim_layer_attributions_use_layer_input_shape():
    first = torch.nn.Linear(2, 3, bias=False)
    last = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 1.0], [2.0, 0.0]]))
        last.weight.copy_(torch.tensor([[2.0, -1.0, 3.0]]))
    model = torch.nn.Sequential(first, last)
    background = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    samples = torch.tensor([[3.0, 1.0], [-1.0, 2.0]])
    explainer = shap.DeepExplainer((model, last), background)

    values = explainer.shap_values(samples)

    expected = (first(samples) - first(background).mean(0)) * last.weight
    assert values.shape == (2, 3, 1)
    np.testing.assert_allclose(values[..., 0], expected.detach().numpy())
    np.testing.assert_allclose(values.sum(axis=1) + explainer.expected_value, model(samples).detach().numpy())


def test_nonlinear_backward_hook_handles_unchanged_and_changed_features():
    model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[2.0, 3.0]]))
    background = torch.tensor([[0.0, -1.0], [0.0, 1.0]])
    samples = torch.tensor([[0.0, 2.0], [1.0, -2.0]])
    explainer = shap.DeepExplainer(model, background)

    values = explainer.shap_values(samples)

    expected = (samples.relu() - background.relu().mean(0)) * model[1].weight
    np.testing.assert_allclose(values[..., 0], expected.detach().numpy())
    np.testing.assert_allclose(values.sum(axis=1) + explainer.expected_value, model(samples).detach().numpy())
