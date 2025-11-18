import warnings

import numpy as np
from packaging import version

from .._explainer import Explainer
from .deep_utils import _check_additivity


class PyTorchDeep(Explainer):
    def __init__(self, model, data):
        import torch

        if version.parse(torch.__version__) < version.parse("0.4"):
            warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

        # check if we have multiple inputs
        self.multi_input = False
        if isinstance(data, list):
            self.multi_input = True
        if not isinstance(data, list):
            data = [data]
        self.data = data
        self.layer = None
        self.input_handle = None
        self.interim = False
        self.interim_inputs_shape = None
        self.expected_value = None  # to keep the DeepExplainer base happy
        if isinstance(model, tuple):
            self.interim = True
            model, layer = model
            model = model.eval()
            self.layer = layer
            self.add_target_handle(self.layer)

            # if we are taking an interim layer, the 'data' is going to be the input
            # of the interim layer; we will capture this using a forward hook
            with torch.no_grad():
                _ = model(*data)
                interim_inputs = self.layer.target_input
                if type(interim_inputs) is tuple:
                    # this should always be true, but just to be safe
                    self.interim_inputs_shape = [i.shape for i in interim_inputs]
                else:
                    self.interim_inputs_shape = [interim_inputs.shape]
            self.target_handle.remove()
            del self.layer.target_input
        self.model = model.eval()

        self.multi_output = False
        self.num_outputs = 1
        with torch.no_grad():
            outputs = model(*data)

            # also get the device everything is running on
            self.device = outputs.device
            if outputs.shape[1] > 1:
                self.multi_output = True
                self.num_outputs = outputs.shape[1]
            self.expected_value = outputs.mean(0).cpu().numpy()

    def add_target_handle(self, layer):
        input_handle = layer.register_forward_hook(get_target_input)
        self.target_handle = input_handle

    def add_handles(self, model, forward_handle, backward_handle):
        """Add handles to all non-container layers in the model.
        Recursively for non-container layers
        """
        handles_list = []
        model_children = list(model.children())
        if model_children:
            for child in model_children:
                handles_list.extend(self.add_handles(child, forward_handle, backward_handle))
        else:  # leaves
            handles_list.append(model.register_forward_hook(forward_handle))
            handles_list.append(model.register_full_backward_hook(backward_handle))
        return handles_list

    def remove_attributes(self, model):
        """Removes the x and y attributes which were added by the forward handles
        Recursively searches for non-container layers
        """
        for child in model.children():
            if "nn.modules.container" in str(type(child)):
                self.remove_attributes(child)
            else:
                try:
                    del child.x
                except AttributeError:
                    pass
                try:
                    del child.y
                except AttributeError:
                    pass

    def gradient(self, idx, inputs):
        import torch

        self.model.zero_grad()
        X = [x.requires_grad_() for x in inputs]
        outputs = self.model(*X)
        selected = [val for val in outputs[:, idx]]
        grads = []
        if self.interim:
            interim_inputs = self.layer.target_input
            for idx, input in enumerate(interim_inputs):
                grad = torch.autograd.grad(
                    selected, input, retain_graph=True if idx + 1 < len(interim_inputs) else None, allow_unused=True
                )[0]
                if grad is not None:
                    grad = grad.cpu().numpy()
                else:
                    grad = torch.zeros_like(X[idx]).cpu().numpy()
                grads.append(grad)
            del self.layer.target_input
            return grads, [i.detach().cpu().numpy() for i in interim_inputs]
        else:
            for idx, x in enumerate(X):
                grad = torch.autograd.grad(
                    selected, x, retain_graph=True if idx + 1 < len(X) else None, allow_unused=True
                )[0]
                if grad is not None:
                    grad = grad.cpu().numpy()
                else:
                    grad = torch.zeros_like(x).cpu().numpy()
                grads.append(grad)
            return grads

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max", check_additivity=True):
        import torch
        # X ~ self.model_input
        # X_data ~ self.data

        # check if we have multiple inputs
        if not self.multi_input:
            assert not isinstance(X, list), "Expected a single tensor model input!"
            X = [X]
        else:
            assert isinstance(X, list), "Expected a list of model inputs!"

        X = [x.detach().to(self.device) for x in X]

        model_output_values = None

        if ranked_outputs is not None and self.multi_output:
            with torch.no_grad():
                model_output_values = self.model(*X)
            # rank and determine the model outputs that we will explain
            if output_rank_order == "max":
                _, model_output_ranks = torch.sort(model_output_values, descending=True)
            elif output_rank_order == "min":
                _, model_output_ranks = torch.sort(model_output_values, descending=False)
            elif output_rank_order == "max_abs":
                _, model_output_ranks = torch.sort(torch.abs(model_output_values), descending=True)
            else:
                emsg = "output_rank_order must be max, min, or max_abs!"
                raise ValueError(emsg)
            model_output_ranks = model_output_ranks[:, :ranked_outputs]
        else:
            model_output_ranks = (
                torch.ones((X[0].shape[0], self.num_outputs)).int() * torch.arange(0, self.num_outputs).int()
            )

        # add the gradient handles
        handles = self.add_handles(self.model, add_interim_values, deeplift_grad)
        if self.interim:
            self.add_target_handle(self.layer)

        # compute the attributions
        output_phis = []
        for i in range(model_output_ranks.shape[1]):
            phis = []
            if self.interim:
                for k in range(len(self.interim_inputs_shape)):
                    phis.append(np.zeros((X[0].shape[0],) + self.interim_inputs_shape[k][1:]))
            else:
                for k in range(len(X)):
                    phis.append(np.zeros(X[k].shape))
            for j in range(X[0].shape[0]):
                # tile the inputs to line up with the background data samples
                tiled_X = [
                    X[t][j : j + 1].repeat((self.data[t].shape[0],) + tuple([1 for k in range(len(X[t].shape) - 1)]))
                    for t in range(len(X))
                ]
                joint_x = [torch.cat((tiled_X[t], self.data[t]), dim=0) for t in range(len(X))]
                # run attribution computation graph
                feature_ind = model_output_ranks[j, i]
                sample_phis = self.gradient(feature_ind, joint_x)
                # assign the attributions to the right part of the output arrays
                if self.interim:
                    sample_phis, output = sample_phis
                    x, data = [], []
                    for k in range(len(output)):
                        x_temp, data_temp = np.split(output[k], 2)
                        x.append(x_temp)
                        data.append(data_temp)
                    for t in range(len(self.interim_inputs_shape)):
                        phis[t][j] = (sample_phis[t][self.data[t].shape[0] :] * (x[t] - data[t])).mean(0)
                else:
                    for t in range(len(X)):
                        phis[t][j] = (
                            (
                                torch.from_numpy(sample_phis[t][self.data[t].shape[0] :]).to(self.device)
                                * (X[t][j : j + 1] - self.data[t])
                            )
                            .cpu()
                            .detach()
                            .numpy()
                            .mean(0)
                        )
            output_phis.append(phis[0] if not self.multi_input else phis)
        # cleanup; remove all gradient handles
        for handle in handles:
            handle.remove()
        self.remove_attributes(self.model)
        if self.interim:
            self.target_handle.remove()

        # check that the SHAP values sum up to the model output
        if check_additivity:
            if model_output_values is None:
                with torch.no_grad():
                    model_output_values = self.model(*X)

            _check_additivity(self, model_output_values.cpu(), output_phis)

        if isinstance(output_phis, list):
            # in this case we have multiple inputs and potentially multiple outputs
            if isinstance(output_phis[0], list):
                output_phis = [np.stack([phi[i] for phi in output_phis], axis=-1) for i in range(len(output_phis[0]))]
            # multiple outputs case
            else:
                output_phis = np.stack(output_phis, axis=-1)
        if ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis


# Module hooks


def deeplift_grad(module, grad_input, grad_output):
    """The backward hook which computes the deeplift
    gradient for an nn.Module
    """
    # first, get the module type
    module_type = module.__class__.__name__
    # first, check the module is supported
    if module_type in op_handler:
        if op_handler[module_type].__name__ not in ["passthrough", "linear_1d"]:
            return op_handler[module_type](module, grad_input, grad_output)
    else:
        warnings.warn(f"unrecognized nn.Module: {module_type}")
        return grad_input


def add_interim_values(module, input, output):
    """The forward hook used to save interim tensors, detached
    from the graph. Used to calculate the multipliers
    """
    import torch

    try:
        del module.x
    except AttributeError:
        pass
    try:
        del module.y
    except AttributeError:
        pass
    module_type = module.__class__.__name__
    if module_type in op_handler:
        func_name = op_handler[module_type].__name__
        # First, check for cases where we don't need to save the x and y tensors
        if func_name == "passthrough":
            pass
        elif func_name == "lstm_cell_handler":
            # Special handling for LSTMCell which has multiple varying inputs: (x, (h, c))
            if type(input) is tuple and len(input) >= 2:
                x_in = input[0]
                if type(input[1]) is tuple:
                    h_in, c_in = input[1]
                else:
                    h_in = input[1] if len(input) > 1 else None
                    c_in = input[2] if len(input) > 2 else None

                # Save all inputs as a tuple
                module.x = (
                    x_in.detach(),
                    h_in.detach() if h_in is not None else None,
                    c_in.detach() if c_in is not None else None,
                )

                # Save outputs (h_new, c_new)
                if type(output) is tuple:
                    module.y = (output[0].detach(), output[1].detach())
                else:
                    module.y = output.detach()
        else:
            # check only the 0th input varies
            for i in range(len(input)):
                if i != 0 and type(output) is tuple:
                    assert input[i] == output[i], "Only the 0th input may vary!"
            # if a new method is added, it must be added here too. This ensures tensors
            # are only saved if necessary
            if func_name in ["maxpool", "nonlinear_1d"]:
                # only save tensors if necessary
                if type(input) is tuple:
                    module.x = torch.nn.Parameter(input[0].detach())
                else:
                    module.x = torch.nn.Parameter(input.detach())
                if type(output) is tuple:
                    module.y = torch.nn.Parameter(output[0].detach())
                else:
                    module.y = torch.nn.Parameter(output.detach())


def get_target_input(module, input, output):
    """A forward hook which saves the tensor - attached to its graph.
    Used if we want to explain the interim outputs of a model
    """
    try:
        del module.target_input
    except AttributeError:
        pass
    module.target_input = input


def passthrough(module, grad_input, grad_output):
    """No change made to gradients"""
    return None


def maxpool(module, grad_input, grad_output):
    import torch

    pool_to_unpool = {
        "MaxPool1d": torch.nn.functional.max_unpool1d,
        "MaxPool2d": torch.nn.functional.max_unpool2d,
        "MaxPool3d": torch.nn.functional.max_unpool3d,
    }
    pool_to_function = {
        "MaxPool1d": torch.nn.functional.max_pool1d,
        "MaxPool2d": torch.nn.functional.max_pool2d,
        "MaxPool3d": torch.nn.functional.max_pool3d,
    }
    delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2) :]
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # we also need to check if the output is a tuple
    y, ref_output = torch.chunk(module.y, 2)
    cross_max = torch.max(y, ref_output)
    diffs = torch.cat([cross_max - ref_output, y - cross_max], 0)

    # all of this just to unpool the outputs
    with torch.no_grad():
        _, indices = pool_to_function[module.__class__.__name__](
            module.x, module.kernel_size, module.stride, module.padding, module.dilation, module.ceil_mode, True
        )
        xmax_pos, rmax_pos = torch.chunk(
            pool_to_unpool[module.__class__.__name__](
                grad_output[0] * diffs, indices, module.kernel_size, module.stride, module.padding, list(module.x.shape)
            ),
            2,
        )

    grad_input = [None for _ in grad_input]
    grad_input[0] = torch.where(
        torch.abs(delta_in) < 1e-7, torch.zeros_like(delta_in), (xmax_pos + rmax_pos) / delta_in
    ).repeat(dup0)

    return tuple(grad_input)


def linear_1d(module, grad_input, grad_output):
    """No change made to gradients."""
    return None


def nonlinear_1d(module, grad_input, grad_output):
    import torch

    delta_out = module.y[: int(module.y.shape[0] / 2)] - module.y[int(module.y.shape[0] / 2) :]

    delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2) :]
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # handles numerical instabilities where delta_in is very small by
    # just taking the gradient in those cases
    grads = [None for _ in grad_input]
    grads[0] = torch.where(
        torch.abs(delta_in.repeat(dup0)) < 1e-6, grad_input[0], grad_output[0] * (delta_out / delta_in).repeat(dup0)
    )
    return tuple(grads)


def lstm_cell_handler(module, grad_input, grad_output):
    """
    Backward hook handler for LSTMCell that computes SHAP values manually.

    Returns gradients in the format: shap_values / (input - baseline)
    so that when DeepExplainer aggregates with (grad * (X - baseline)).mean(0),
    it produces the correct SHAP values.
    """
    import torch

    # Check if we have saved tensors
    if not hasattr(module, "x") or not hasattr(module, "y"):
        warnings.warn("LSTM handler: No saved tensors, using standard gradients")
        return None

    # Extract inputs (doubled batch: [actual; baseline])
    if isinstance(module.x, tuple):
        x_doubled = module.x[0]
        h_doubled = module.x[1] if len(module.x) > 1 else None
        c_doubled = module.x[2] if len(module.x) > 2 else None
    else:
        return None

    # Split actual and baseline
    batch_size = x_doubled.shape[0] // 2
    x = x_doubled[:batch_size]
    x_base = x_doubled[batch_size:]

    if h_doubled is not None:
        h = h_doubled[:batch_size]
        h_base = h_doubled[batch_size:]
    else:
        hidden_size = module.hidden_size
        h = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)
        h_base = h.clone()

    if c_doubled is not None:
        c = c_doubled[:batch_size]
        c_base = c_doubled[batch_size:]
    else:
        hidden_size = module.hidden_size
        c = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)
        c_base = c.clone()

    # Extract weights from LSTMCell
    hidden_size = module.hidden_size
    W_ii, W_if, W_ig, W_io = torch.chunk(module.weight_ih, 4, dim=0)
    W_hi, W_hf, W_hg, W_ho = torch.chunk(module.weight_hh, 4, dim=0)

    if module.bias:
        b_ii, b_if, b_ig, b_io = torch.chunk(module.bias_ih, 4, dim=0)
        b_hi, b_hf, b_hg, b_ho = torch.chunk(module.bias_hh, 4, dim=0)
    else:
        zeros = torch.zeros(hidden_size, device=x.device, dtype=x.dtype)
        b_ii = b_if = b_ig = b_io = zeros
        b_hi = b_hf = b_hg = b_ho = zeros

    # Helper function for gate SHAP - keep hidden dimension separate
    def manual_shap_gate(W_i, W_h, b_i, b_h, x, h, x_base, h_base, activation="sigmoid"):
        linear_current = torch.matmul(x, W_i.T) + torch.matmul(h, W_h.T) + b_i + b_h
        linear_base = torch.matmul(x_base, W_i.T) + torch.matmul(h_base, W_h.T) + b_i + b_h

        act_fn = torch.sigmoid if activation == "sigmoid" else torch.tanh
        output = act_fn(linear_current)
        output_base = act_fn(linear_base)

        # DeepLift rescale rule
        denom = torch.matmul(x, W_i.T) + torch.matmul(h, W_h.T)

        x_diff = (x - x_base).unsqueeze(1)
        h_diff = (h - h_base).unsqueeze(1)

        W_i_expanded = W_i.unsqueeze(0)
        W_h_expanded = W_h.unsqueeze(0)

        numerator_x = W_i_expanded * x_diff
        numerator_h = W_h_expanded * h_diff

        denom_expanded = denom.unsqueeze(-1)

        Z_x = numerator_x / (denom_expanded + 1e-10)
        Z_h = numerator_h / (denom_expanded + 1e-10)

        output_diff = (output - output_base).unsqueeze(-1)

        # Return element-wise relevances WITHOUT summing over hidden dimension yet
        # r_x: (batch, hidden_size, input_size)
        # r_h: (batch, hidden_size, hidden_size)
        r_x = output_diff * Z_x  # (batch, hidden_size, input_size)
        r_h = output_diff * Z_h  # (batch, hidden_size, hidden_size)

        return r_x, r_h, output, output_base

    # Helper function for element-wise multiplication SHAP
    def manual_shap_multiplication(a, b, a_base, b_base):
        r_a = 0.5 * (a * b - a_base * b + a * b_base - a_base * b_base)
        r_b = 0.5 * (a * b - a * b_base + a_base * b - a_base * b_base)
        return r_a, r_b

    # Calculate SHAP values for each gate (element-wise)
    r_x_i, r_h_i, i_t, i_t_base = manual_shap_gate(W_ii, W_hi, b_ii, b_hi, x, h, x_base, h_base, activation="sigmoid")

    r_x_f, r_h_f, f_t, f_t_base = manual_shap_gate(W_if, W_hf, b_if, b_hf, x, h, x_base, h_base, activation="sigmoid")

    r_x_g, r_h_g, c_tilde, c_tilde_base = manual_shap_gate(
        W_ig, W_hg, b_ig, b_hg, x, h, x_base, h_base, activation="tanh"
    )

    # Cell state update: c_new = f_t ⊙ c + i_t ⊙ c_tilde
    # Shapley values for multiplications
    r_f_from_mult, r_c_from_f = manual_shap_multiplication(f_t, c, f_t_base, c_base)
    r_i_from_mult, r_ctilde_from_mult = manual_shap_multiplication(i_t, c_tilde, i_t_base, c_tilde_base)

    # Get output selection from grad_output
    # grad_output[1] is for c_new, shape (doubled_batch, hidden_size)
    # Each row indicates which output dimension is being explained
    if len(grad_output) > 1 and grad_output[1] is not None:
        c_output_selector = grad_output[1][:batch_size]  # (batch, hidden_size)
    else:
        # If no grad_output, explain all outputs equally
        c_output_selector = torch.ones_like(c)

    # Now r_x_f, r_h_f etc have shape (batch, hidden_size, feature_size)
    # We need to:
    # 1. Multiply multiplication relevances by output selector
    # 2. Distribute back to input features
    # 3. Sum over hidden dimension

    # Weight multiplication relevances by output selector
    # r_f_from_mult: (batch, hidden_size)
    # c_output_selector: (batch, hidden_size)
    # Result: (batch, hidden_size)
    r_f_weighted = r_f_from_mult * c_output_selector
    r_i_weighted = r_i_from_mult * c_output_selector
    r_ctilde_weighted = r_ctilde_from_mult * c_output_selector

    # For forget gate path:
    # r_x_f: (batch, hidden_size, input_size)
    # r_h_f: (batch, hidden_size, hidden_size)
    # r_f_weighted: (batch, hidden_size)

    # Distribute r_f_weighted[b,k] to input features based on their contribution to f_t[k]
    # Total contribution to f_t[k] from all inputs: sum over input_size and hidden_size
    total_r_f_per_hidden = r_x_f.sum(dim=2) + r_h_f.sum(dim=2)  # (batch, hidden_size)

    # Avoid division by zero
    total_r_f_per_hidden = torch.where(
        torch.abs(total_r_f_per_hidden) < 1e-10, torch.ones_like(total_r_f_per_hidden), total_r_f_per_hidden
    )

    # Scale gate relevances by multiplication relevances
    # (batch, hidden_size, 1) * (batch, hidden_size, input_size) / (batch, hidden_size, 1)
    scale_f = (r_f_weighted / total_r_f_per_hidden).unsqueeze(-1)
    shap_x_from_f = (r_x_f * scale_f).sum(dim=1)  # Sum over hidden_size → (batch, input_size)
    shap_h_from_f = (r_h_f * scale_f).sum(dim=1)  # Sum over hidden_size → (batch, hidden_size)

    # For input gate path
    total_r_i_per_hidden = r_x_i.sum(dim=2) + r_h_i.sum(dim=2)
    total_r_i_per_hidden = torch.where(
        torch.abs(total_r_i_per_hidden) < 1e-10, torch.ones_like(total_r_i_per_hidden), total_r_i_per_hidden
    )
    scale_i = (r_i_weighted / total_r_i_per_hidden).unsqueeze(-1)
    shap_x_from_i = (r_x_i * scale_i).sum(dim=1)
    shap_h_from_i = (r_h_i * scale_i).sum(dim=1)

    # For candidate gate path
    total_r_g_per_hidden = r_x_g.sum(dim=2) + r_h_g.sum(dim=2)
    total_r_g_per_hidden = torch.where(
        torch.abs(total_r_g_per_hidden) < 1e-10, torch.ones_like(total_r_g_per_hidden), total_r_g_per_hidden
    )
    scale_g = (r_ctilde_weighted / total_r_g_per_hidden).unsqueeze(-1)
    shap_x_from_g = (r_x_g * scale_g).sum(dim=1)
    shap_h_from_g = (r_h_g * scale_g).sum(dim=1)

    # Sum all contributions
    shap_x = shap_x_from_f + shap_x_from_i + shap_x_from_g  # (batch, input_size)
    shap_h = shap_h_from_f + shap_h_from_i + shap_h_from_g  # (batch, hidden_size)
    shap_c = (r_c_from_f * c_output_selector).sum(
        dim=1, keepdim=True
    )  # (batch, 1) but broadcast to (batch, hidden_size)

    # Actually, shap_c should be (batch, hidden_size) with relevance for each c[k]
    # But we're explaining c_new, not c_in, so we need element-wise relevance
    shap_c = r_c_from_f * c_output_selector  # (batch, hidden_size)

    # Compute deltas
    delta_x = x - x_base
    delta_h = h - h_base
    delta_c = c - c_base

    dup0 = [2] + [1 for i in delta_x.shape[1:]]

    # Compute gradients with numerical stability (avoid division by zero)
    # Where delta is very small, the SHAP value should also be very small (no change = no contribution)
    # So we can safely set gradient to 0 in those cases
    eps = 1e-6
    grad_x_value = torch.where(torch.abs(delta_x) < eps, torch.zeros_like(shap_x), shap_x / delta_x)
    grad_h_value = torch.where(torch.abs(delta_h) < eps, torch.zeros_like(shap_h), shap_h / delta_h)
    grad_c_value = torch.where(torch.abs(delta_c) < eps, torch.zeros_like(shap_c), shap_c / delta_c)

    # Return gradients repeated for doubled batch
    # grad_input structure: (grad_x, (grad_h, grad_c))
    grads_x = grad_x_value.repeat(dup0)
    grads_h = grad_h_value.repeat(dup0)
    grads_c = grad_c_value.repeat(dup0)

    return (grads_x, (grads_h, grads_c))


op_handler = {}

# passthrough ops, where we make no change to the gradient
op_handler["Dropout3d"] = passthrough
op_handler["Dropout2d"] = passthrough
op_handler["Dropout"] = passthrough
op_handler["AlphaDropout"] = passthrough
op_handler["Identity"] = passthrough
op_handler["Flatten"] = passthrough

op_handler["Conv1d"] = linear_1d
op_handler["Conv2d"] = linear_1d
op_handler["Conv3d"] = linear_1d
op_handler["ConvTranspose1d"] = linear_1d
op_handler["ConvTranspose2d"] = linear_1d
op_handler["ConvTranspose3d"] = linear_1d
op_handler["Linear"] = linear_1d
op_handler["AvgPool1d"] = linear_1d
op_handler["AvgPool2d"] = linear_1d
op_handler["AvgPool3d"] = linear_1d
op_handler["AdaptiveAvgPool1d"] = linear_1d
op_handler["AdaptiveAvgPool2d"] = linear_1d
op_handler["AdaptiveAvgPool3d"] = linear_1d
op_handler["BatchNorm1d"] = linear_1d
op_handler["BatchNorm2d"] = linear_1d
op_handler["BatchNorm3d"] = linear_1d

op_handler["LeakyReLU"] = nonlinear_1d
op_handler["ReLU"] = nonlinear_1d
op_handler["ELU"] = nonlinear_1d
op_handler["Sigmoid"] = nonlinear_1d
op_handler["Tanh"] = nonlinear_1d
op_handler["Softplus"] = nonlinear_1d
op_handler["Softmax"] = nonlinear_1d
op_handler["SELU"] = nonlinear_1d
op_handler["GELU"] = nonlinear_1d

op_handler["MaxPool1d"] = maxpool
op_handler["MaxPool2d"] = maxpool
op_handler["MaxPool3d"] = maxpool

op_handler["LSTMCell"] = lstm_cell_handler
