import random
import numpy as np


def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)

    from tests.explainers.test_deep import _skip_if_no_pytorch
    _skip_if_no_pytorch()

    import torch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True


def set_seed():
    set_random_seeds()
