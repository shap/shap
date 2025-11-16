from __future__ import annotations

import types
from typing import Any

from ..utils._exceptions import InvalidMaskerError
from ._masker import Masker


class Composite(Masker):
    """This merges several maskers for different inputs together into a single composite masker.

    This is not yet implemented.
    """

    def __init__(self, *maskers: Masker) -> None:
        self.maskers: tuple[Masker, ...] = maskers

        self.arg_counts: list[int] = []
        self.total_args: int = 0
        self.text_data: bool = False
        self.image_data: bool = False
        all_have_clustering: bool = True
        for masker in self.maskers:
            all_args: int = masker.__call__.__code__.co_argcount

            if masker.__call__.__defaults__ is not None:  # in case there are no kwargs
                kwargs: int = len(masker.__call__.__defaults__)
            else:
                kwargs = 0
            num_args: int = all_args - kwargs - 2
            self.arg_counts.append(num_args)  # -2 is for the self and mask arg
            self.total_args += num_args

            if not hasattr(masker, "clustering"):
                all_have_clustering = False

            self.text_data = self.text_data or getattr(masker, "text_data", False)
            self.image_data = self.image_data or getattr(masker, "image_data", False)

        if all_have_clustering:
            self.clustering = types.MethodType(joint_clustering, self)

    def shape(self, *args: Any) -> tuple[int | None, int]:
        """Compute the shape of this masker as the sum of all the sub masker shapes."""
        assert len(args) == self.total_args, "The number of passed args is incorrect!"

        rows: int | None = None
        cols: int = 0
        pos: int = 0
        for i, masker in enumerate(self.maskers):
            if callable(masker.shape):
                shape: tuple[int | None, int] = masker.shape(*args[pos : pos + self.arg_counts[i]])
            else:
                shape = masker.shape
            if rows is None:
                rows = shape[0]
            else:
                assert shape[1] == 0 or rows == shape[0], (
                    "All submaskers of a Composite masker must return the same number of rows!"
                )
            cols += shape[1]
            pos += self.arg_counts[i]
        return rows, cols

    def mask_shapes(self, *args: Any) -> list[Any]:
        """The shape of the masks we expect."""
        out: list[Any] = []
        pos: int = 0
        for i, masker in enumerate(self.maskers):
            out.extend(masker.mask_shapes(*args[pos : pos + self.arg_counts[i]]))
        return out

    def data_transform(self, *args: Any) -> list[Any]:
        """Transform the argument"""
        arg_pos: int = 0
        out: list[Any] = []
        for i, masker in enumerate(self.maskers):
            masker_args: tuple[Any, ...] = args[arg_pos : arg_pos + self.arg_counts[i]]
            if hasattr(masker, "data_transform"):
                out.extend(masker.data_transform(*masker_args))
            else:
                out.extend(masker_args)
            arg_pos += self.arg_counts[i]

        return out

    def __call__(self, mask: Any, *args: Any) -> tuple[Any, ...]:  # type: ignore[override]
        mask = self._standardize_mask(mask, *args)
        assert len(args) == self.total_args, "The number of passed args is incorrect!"

        # compute all the shapes and confirm they align
        arg_pos: int = 0
        shapes: list[tuple[int | None, int]] = []
        num_rows: int | None = None
        for i, masker in enumerate(self.maskers):
            masker_args: tuple[Any, ...] = args[arg_pos : arg_pos + self.arg_counts[i]]
            if callable(masker.shape):
                shapes.append(masker.shape(*masker_args))
            else:
                shapes.append(masker.shape)

            if num_rows is None:
                num_rows = shapes[-1][0]
            elif num_rows == 1 and shapes[-1][0] is not None:
                num_rows = shapes[-1][0]

            if shapes[-1][0] != num_rows and shapes[-1][0] != 1 and shapes[-1][0] is not None:
                raise InvalidMaskerError(
                    "The composite masker can only join together maskers with a compatible number of background rows!"
                )
            arg_pos += self.arg_counts[i]

        # call all the submaskers and combine their outputs
        arg_pos = 0
        mask_pos: int = 0
        masked: list[Any] = []
        for i, masker in enumerate(self.maskers):
            masker_args: tuple[Any, ...] = args[arg_pos : arg_pos + self.arg_counts[i]]  # type: ignore[no-redef]
            masked_out: tuple[Any, ...] = masker(mask[mask_pos : mask_pos + shapes[i][1]], *masker_args)
            # num_rows is guaranteed to be an int at this point (not None) by the logic above
            assert num_rows is not None
            if num_rows > 1 and (shapes[i][0] == 1 or shapes[i][0] is None):
                masked_out = tuple([m[0] for _ in range(num_rows)] for m in masked_out)
            masked.extend(masked_out)

            mask_pos += shapes[i][1]
            arg_pos += self.arg_counts[i]

        return tuple(masked)


def joint_clustering(self: Composite, *args: Any) -> Any:
    """Return a joint clustering that merges the clusterings of all the submaskers."""
    single_clustering: Any = []
    arg_pos: int = 0
    for i, masker in enumerate(self.maskers):
        masker_args: tuple[Any, ...] = args[arg_pos : arg_pos + self.arg_counts[i]]
        if callable(masker.clustering):
            clustering: Any = masker.clustering(*masker_args)
        else:
            clustering = masker.clustering

        if len(single_clustering) == 0:
            single_clustering = clustering
        elif len(clustering) != 0:
            raise NotImplementedError(
                "Joining two non-trivial clusterings is not yet implemented in the Composite masker!"
            )
    return single_clustering
