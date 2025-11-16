from __future__ import annotations

from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt

from .._serializable import Deserializer, Serializable, Serializer
from ..utils import safe_isinstance


class Model(Serializable):
    """This is the superclass of all models."""

    def __init__(self, model: Any = None) -> None:
        """Wrap a callable model as a SHAP Model object."""
        if isinstance(model, Model):
            self.inner_model: Any = model.inner_model
        else:
            self.inner_model = model

        if hasattr(model, "output_names"):
            self.output_names = model.output_names

    def __call__(self, *args: Any) -> npt.NDArray[Any]:
        out = self.inner_model(*args)
        is_tensor = safe_isinstance(out, "torch.Tensor")
        out = out.cpu().detach().numpy() if is_tensor else np.array(out)
        return out

    def save(self, out_file: BinaryIO) -> None:
        """Save the model to the given file stream."""
        super().save(out_file)
        with Serializer(out_file, "shap.Model", version=0) as s:
            s.save("model", self.inner_model)

    @classmethod
    def load(cls, in_file: BinaryIO, instantiate: bool = True) -> Model | dict[str, Any]:
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.Model", min_version=0, max_version=0) as s:
            kwargs["model"] = s.load("model")
        return kwargs
