import numpy as np


def _check_additivity(explainer, model_output_values, output_phis):
    TOLERANCE = 1e-2

    assert len(explainer.expected_value) == model_output_values.shape[1], (
        "Length of expected values and model outputs does not match."
    )

    for t in range(len(explainer.expected_value)):
        if not explainer.multi_input:
            diffs = (
                model_output_values[:, t]
                - explainer.expected_value[t]
                - output_phis[t].sum(axis=tuple(range(1, output_phis[t].ndim)))
            )
        else:
            diffs = model_output_values[:, t] - explainer.expected_value[t]

            for i in range(len(output_phis[t])):
                diffs -= output_phis[t][i].sum(axis=tuple(range(1, output_phis[t][i].ndim)))

        maxdiff = np.abs(diffs).max()

        assert maxdiff < TOLERANCE, (
            "The SHAP explanations do not sum up to the model's output! This is either because of a "
            "rounding error or because an operator in your computation graph was not fully supported. If "
            "the sum difference of %f is significant compared to the scale of your model outputs, please post "
            f"as a github issue, with a reproducible example so we can debug it. Used framework: {explainer.framework} - Max. diff: {maxdiff} - Tolerance: {TOLERANCE}"
        )
