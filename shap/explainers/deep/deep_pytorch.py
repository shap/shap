import numpy as np
import warnings
from shap.explainers.explainer import Explainer
from distutils.version import LooseVersion
torch = None


class PyTorchDeepExplainer(Explainer):

    def __init__(self, model, data):
        # try and import pytorch
        global torch
        if torch is None:
            import torch
            if LooseVersion(torch.__version__) < LooseVersion("0.4"):
                warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

        # check if we have multiple inputs
        self.multi_input = False
        if type(data) == list:
            self.multi_input = True
        if type(data) != list:
            data = [data]
        # for consistency, the method signature calls for data as the model input.
        # However, within this class, self.model_inputs is the input (i.e. the data passed by the user)
        # and self.data is the background data for the layer we want to assign importances to. If this layer is
        # the input, then self.data = self.model_inputs
        self.model_inputs = data
        self.layer = None
        self.input_handle = None
        self.interim = False
        if type(model) == tuple:
            self.interim = True
            model, layer = model
            model = model.eval()
            self.add_target_handle(layer)
            self.layer = layer

            # now, if we are taking an interim layer, the 'data' is going to be the input
            # of the interim layer; we will capture this using a forward hook
            with torch.no_grad():
                _ = model(*data)
                interim_inputs = self.layer.target_input
                if type(interim_inputs) is tuple:
                    # this should always be true, but just to be safe
                    self.data = [torch.tensor(i) for i in interim_inputs]
                else:
                    self.data = [torch.tensor(interim_inputs)]
        else:
            self.data = data
        self.model = model.eval()

        # now, get the expected model mean and outputs
        # note that this will also conveniently add all the reference values
        self.multi_output = False
        with torch.no_grad():
            outputs = model(*data)
            self.expected_value = outputs.mean(0)
            if outputs.shape[1] > 1:
                self.multi_output = True
                self.num_outputs = outputs.shape[1]
        if self.interim:
            self.target_handle.remove()

    def add_target_handle(self, layer):
        input_handle = layer.register_forward_hook(self.get_target_input)
        self.target_handle = input_handle

    @staticmethod
    def get_target_input(module, input, output):
        """Saves the tensor - attached to its graph.
        Used if we want to explain the interim outputs of a model
        """
        try:
            del module.target_input
        except AttributeError:
            pass
        setattr(module, 'target_input', input)

    @staticmethod
    def add_interim_values(module, input, output):
        """Saves interim tensors detached from the graph.
        Used to calculate multipliers
        """
        try:
            del module.x
        except AttributeError:
            pass
        if type(input) is tuple:
            setattr(module, 'x', tuple(i.detach() for i in input))
        else:
            setattr(module, 'x', input.detach())

        try:
            del module.y
        except AttributeError:
            pass
        if type(output) is tuple:
            setattr(module, 'y', tuple(o.detach() for o in output))
        else:
            setattr(module, 'y', output.detach())

    @staticmethod
    def deeplift_grad(module, grad_input, grad_output):
        # first, get the module type
        type = module.__class__.__name__
        # first, check its not a container
        if type in op_handler:
            return op_handler[type](module, grad_input, grad_output)
        else:
            print('Warning: unrecognized nn.Module: {}'.format(type))
            return grad_input

    def gradient(self, idx, inputs):
        self.model.zero_grad()
        X = [x.requires_grad_() for x in inputs]
        outputs = self.model(*X)
        selected = [val for val in outputs[:, idx]]
        if self.input_handle is not None:
            interim_inputs = self.layer.target_input
            grads = [torch.autograd.grad(selected, input)[0].cpu().numpy() for input in interim_inputs]
            del self.layer.target_input
        else:
            grads = [torch.autograd.grad(selected, x)[0].cpu().numpy() for x in X]
        return grads

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max"):

        # X ~ self.model_input
        # X_data ~ self.data

        # check if we have multiple inputs
        if not self.multi_input:
            assert type(X) != list, "Expected a single tensor model input!"
            X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"

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
                assert False, "output_rank_order must be max, min, or max_abs!"
            model_output_ranks = model_output_ranks[:, :ranked_outputs]
        else:
            model_output_ranks = (torch.ones((X[0].shape[0], self.num_outputs)).int() *
                                  torch.arange(0, self.num_outputs).int())

        # add the gradient handles
        gradient_handles = []
        interim_handles = []
        for child in self.model.children():
            if 'nn.modules.container' in str(type(child)):
                for subchild in child.children():
                    interim_handles.append(subchild.register_forward_hook(self.add_interim_values))
                    gradient_handles.append(subchild.register_backward_hook(self.deeplift_grad))
            else:
                interim_handles.append(child.register_forward_hook(self.add_interim_values ))
                gradient_handles.append(child.register_backward_hook(self.deeplift_grad))

        # compute the attributions
        output_phis = []
        for i in range(model_output_ranks.shape[1]):
            phis = []
            for k in range(len(X)):
                phis.append(np.zeros(X[k].shape))
            for j in range(X[0].shape[0]):
                # tile the inputs to line up with the background data samples
                tiled_X = [X[l][j:j + 1].repeat(
                                   (self.data[l].shape[0],) + tuple([1 for k in range(len(X[l].shape) - 1)])) for l
                           in range(len(X))]
                joint_x = [torch.cat((tiled_X[l], self.model_inputs[l]), dim=0) for l in range(len(X))]
                # run attribution computation graph
                feature_ind = model_output_ranks[j, i]
                sample_phis = self.gradient(feature_ind, joint_x)
                # assign the attributions to the right part of the output arrays
                for l in range(len(X)):
                    phis[l][j] = (sample_phis[l][self.data[l].shape[0]:] * (X[l][j: j + 1] - self.data[l])).mean(0)
            output_phis.append(phis[0] if not self.multi_input else phis)
        # cleanup; remove all gradient handles
        for handle in gradient_handles:
            handle.remove()
        # TODO remove all attributes given to the modules

        if not self.multi_output:
            return output_phis[0]
        elif ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis


def passthrough(module, grad_input, grad_output):
    """No change made to gradients"""
    return grad_input


def maxpool(module, grad_input, grad_output):

    pool_to_unpool = {
        'MaxPool1d': torch.nn.functional.max_unpool1d,
        'MaxPool2d': torch.nn.functional.max_unpool2d,
        'MaxPool3d': torch.nn.functional.max_unpool1d
    }
    x, ref_input = torch.chunk(module.x[0], 2)
    delta_in = x - ref_input
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # we also need to check if the output is a tuple
    if type(module.y) is tuple:
        y, ref_output = torch.chunk(module.y[0], 2)
    else:
        y, ref_output = torch.chunk(module.y, 2)

    cross_max = torch.where(y > ref_output, y, ref_output)

    diffs = torch.cat([cross_max - ref_output, y - cross_max], 0)

    # all of this just to unpool the outputs
    temp_module = getattr(torch.nn, module.__class__.__name__)(
        module.kernel_size, module.stride, module.padding, module.dilation,
        return_indices=True, ceil_mode=module.ceil_mode)
    _, indices = temp_module(module.x[0])

    unpooled = pool_to_unpool[module.__class__.__name__](
        grad_output[0] * diffs, indices, module.kernel_size, module.stride,
        module.padding, delta_in.shape)
    xmax_pos, rmax_pos = torch.chunk(unpooled, 2)
    # handles numerical instabilities where delta_in is very small by
    # just taking the gradient in those cases
    grads = [None for _ in grad_input]
    grads[0] = torch.where(torch.abs(delta_in) < 1e-7, torch.zeros_like(delta_in),
                           (xmax_pos + rmax_pos) / delta_in).repeat(dup0)
    return tuple(grads)


def linear_1d(module, grad_input, grad_output):
    for i in range(len(module.x)):
        if i != 0 and type(module.y) is tuple:
            assert module.x[i] == module.y[i], "Only the 0th input may vary!"
    return grad_input


def nonlinear_1d(module, grad_input, grad_output):
    # check only the 0th input varies
    for i in range(len(module.x)):
        if i != 0 and type(module.y) is tuple:
            assert module.x[i] == module.y[i], "Only the 0th input may vary!"

    # we also need to check if the output is a tuple
    if type(module.y) is tuple:
        y, ref_output = torch.chunk(module.y[0], 2)
    else:
        y, ref_output = torch.chunk(module.y, 2)

    x, ref_input = torch.chunk(module.x[0], 2)
    delta_in = x - ref_input
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # handles numerical instabilities where delta_in is very small by
    # just taking the gradient in those cases
    grads = [None for _ in grad_input]
    grads[0] = torch.where(torch.abs(delta_in.repeat(dup0)) < 1e-6, grad_input[0],
                           grad_output[0] * ((y - ref_output) / delta_in).repeat(dup0))
    return tuple(grads)


op_handler = {}

# passthrough ops, where we make no change to the gradient
op_handler['Dropout2d'] = passthrough

op_handler['Conv2d'] = linear_1d
op_handler['Linear'] = linear_1d
op_handler['AvgPool1d'] = linear_1d
op_handler['AvgPool2d'] = linear_1d
op_handler['AvgPool3d'] = linear_1d

op_handler['ReLU'] = nonlinear_1d
op_handler['ELU'] = nonlinear_1d
op_handler['Sigmoid'] = nonlinear_1d
op_handler["Tanh"] = nonlinear_1d
op_handler["Softplus"] = nonlinear_1d
op_handler['Softmax'] = nonlinear_1d

op_handler['MaxPool1d'] = maxpool
op_handler['MaxPool2d'] = maxpool
op_handler['MaxPool3d'] = maxpool
