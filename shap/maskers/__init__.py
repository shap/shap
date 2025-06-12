# from ._composite import Composite
# from ._fixed import Fixed
# from ._fixed_composite import FixedComposite
# from ._image import Image
# from ._masker import Masker
# from ._output_composite import OutputComposite
# from ._tabular import Impute, Independent, Partition
# from ._text import Text

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "_composite": ["Composite"],
        "_fixed": ["Fixed"],
        "_fixed_composite": ["FixedComposite"],
        "_image": ["Image"],
        "_masker": ["Masker"],
        "_output_composite": ["OutputComposite"],
        "_tabular": ["Impute", "Independent", "Partition"],
        "_text": ["Text"],
    },
)

# __all__ = [
#     "Composite",
#     "Fixed",
#     "FixedComposite",
#     "Image",
#     "Masker",
#     "OutputComposite",
#     "Impute",
#     "Independent",
#     "Partition",
#     "Text",
# ]
