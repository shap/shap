import random
import torch
import pytest

import numpy as np


def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True


@pytest.fixture()
def set_seed():
    set_random_seeds()
