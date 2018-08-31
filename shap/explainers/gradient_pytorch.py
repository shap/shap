import numpy as np
import warnings
from .explainer import Explainer
from distutils.version import LooseVersion
torch = None


class PyTorchGradientExplainer(Explainer):
    """ Explains a model using expected gradients (an extension of integrated gradients).

    Expected gradients an extension of the integrated gradients method (Sundararajan et al. 2017), a
    feature attribution method designed for differentiable models based on an extension of Shapley
    values to infinite player games (Aumann-Shapley values). Integrated gradients values are a bit
    different from SHAP values, and require a single reference value to integrate from. As an adaptation
    to make them approximate SHAP values, expected gradients reformulates the integral as an expectation
    and combines that expectation with sampling reference values from the background dataset. This leads
    to a single combined expectation of gradients that converges to attributions that sum to the
    difference between the expected model output and the current output.
    """

    def __init__(self, model, data, batch_size=50, local_smoothing=0):
        """ An explainer object for a differentiable model using a given background dataset.

        Note that the complexity of the method scales linearly with the number of background data
        samples. Passing the entire training dataset as `data` will give very accurate expected
        values, but be unreasonably expensive. The variance of the expectation estimates scale by
        roughly 1/sqrt(N) for N background data samples. So 100 samples will give a good estimate,
        and 1000 samples a very good estimate of the expected values.

        Parameters
        ----------
        model : nn.Module
            The model to be explained, where y=model(x). Note that SHAP values are specific to a single
            output value, so the output y should be a single dimensional output (,1).

        data : [torch.Tensor]
            The background dataset to use for integrating out features. GradientExplainer integrates
            over these samples. The data passed here must match the input operations given in the
            first argument.
        """

        # try and import pytorch
        global torch
        if torch is None:
            import torch
            if LooseVersion(torch.__version__) < LooseVersion("0.4"):
                warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

        self.model = model.eval()
        # check if we have multiple inputs
        self.multi_input = False
        if type(data) == list:
            self.multi_input = True
        if type(data) != list:
            data = [data]

        self.data = data
        self.batch_size = batch_size
        self.local_smoothing = local_smoothing

        # this is wrong; multi output is multi class output
        multi_output = False
        outputs = self.model(*self.data)
        if outputs.shape[1] > 1:
            multi_output = True
        self.multi_output = multi_output

        if not self.multi_output:
            self.gradients = [None]
        else:
            self.gradients = [None for i in range(outputs.shape[1])]

    def gradient(self, idx, inputs):
        self.model.zero_grad()
        X = [x.requires_grad_() for x in inputs]
        outputs = self.model(*X)
        selected = [val for val in outputs[:, idx]]
        grads = [torch.autograd.grad(selected, x_batch)[0] for x_batch in X]
        return grads

    def shap_values(self, X, nsamples=200, ranked_outputs=None, output_rank_order="max"):
        """ Return the values for the model applied to X.

        Parameters
        ----------
        X : list, tensor
            A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
            explain the model's output.

        ranked_outputs : None or int
            If ranked_outputs is None then we explain all the outputs in a multi-output model. If
            ranked_outputs is a positive integer then we only explain that many of the top model
            outputs (where "top" is determined by output_rank_order). Note that this causes a pair
            of values to be returned (shap_values, indexes), where phi is a list of numpy arrays for each of
            the output ranks, and indexes is a matrix that tells for each sample which output indexes
            were choses as "top".

        output_rank_order : "max", "min", or "max_abs"
            How to order the model outputs when using ranked_outputs, either by maximum, minimum, or
            maximum absolute value.

        Returns
        -------
        For a models with a single output this returns a tensor of SHAP values with the same shape
        as X. For a model with multiple outputs this returns a list of SHAP value tensors, each of
        which are the same shape as X. If ranked_outputs is None then this list of tensors matches
        the number of model outputs. If ranked_outputs is a positive integer a pair is returned
        (shap_values, indexes), where shap_values is a list of tensors with a length of
        ranked_outputs, and indexes is a matrix that tells for each sample which output indexes
        were chosen as "top".
        """
        # check if we have multiple inputs
        if not self.multi_input:
            assert type(X) != list, "Expected a single tensor model input!"
            X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"
        # assert len(self.model_inputs) == len(X), "Number of model inputs does not match the number given!"

        # rank and determine the model outputs that we will explain
        with torch.no_grad():
            model_output_values = self.model(*X)
        if ranked_outputs is not None and self.multi_output:
            if output_rank_order == "max":
                _, model_output_ranks = torch.sort(model_output_values, descending=True)
            elif output_rank_order == "min":
                _, model_output_ranks = torch.sort(model_output_values, descending=False)
            elif output_rank_order == "max_abs":
                _, model_output_ranks = torch.sort(torch.abs(model_output_values), descending=True)
            else:
                assert False, "output_rank_order must be max, min, or max_abs!"
            model_output_ranks = model_output_ranks[:, :ranked_outputs]
        else:
            model_output_ranks = (torch.ones((X[0].shape[0], len(self.gradients))).int() *
                                  torch.arange(0, len(self.gradients)).int())

        # compute the attributions
        output_phis = []
        samples_input = [torch.zeros((nsamples,) + X[l].shape[1:]) for l in range(len(X))]
        samples_delta = [torch.zeros((nsamples,) + X[l].shape[1:]) for l in range(len(X))]
        rseed = np.random.randint(0, 1e6)
        for i in range(model_output_ranks.shape[1]):
            np.random.seed(rseed) # so we get the same noise patterns for each output class
            phis = []
            phi_vars = []
            for k in range(len(X)):
                phis.append(torch.zeros(X[k].shape))
                phi_vars.append(torch.zeros(X[k].shape))
            for j in range(X[0].shape[0]):
                # fill in the samples arrays
                for k in range(nsamples):
                    rind = np.random.choice(self.data[0].shape[0])
                    t = np.random.uniform()
                    for l in range(len(X)):
                        if self.local_smoothing > 0:
                            x = torch.tensor(X[l][j]) + torch.Tensor(X[l][j].shape).normal_() * self.local_smoothing
                        else:
                            x = torch.tensor(X[l][j])
                        samples_input[l][k] = t * x + (1 - t) * torch.tensor(self.data[l][rind])
                        samples_delta[l][k] = x - torch.tensor(self.data[l][rind])

                # compute the gradients at all the sample points
                find = model_output_ranks[j, i]
                grads = []
                for b in range(0, nsamples, self.batch_size):
                    batch = [samples_input[l][b:min(b+self.batch_size,nsamples)] for l in range(len(X))]
                    grads.append(self.gradient(find, batch))
                grad = [torch.cat([g[l] for g in grads], 0) for l in range(len(X))]
                # assign the attributions to the right part of the output arrays
                for l in range(len(X)):
                    samples = grad[l] * samples_delta[l]
                    phis[l][j] = samples.mean(0)
                    phi_vars[l][j] = samples.var(0) / np.sqrt(samples.shape[0]) # estimate variance of means

            output_phis.append(phis[0] if not self.multi_input else phis)
        if not self.multi_output:
            return output_phis[0]
        elif ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis
