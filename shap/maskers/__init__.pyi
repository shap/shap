# Type stub for shap.maskers module
# Import-based type stub for lazy loading with attach_stub()

# Base masker class
# Composite maskers
from ._composite import Composite as Composite

# Simple maskers
from ._fixed import Fixed as Fixed
from ._fixed_composite import FixedComposite as FixedComposite

# Domain-specific maskers
from ._image import Image as Image
from ._masker import Masker as Masker
from ._output_composite import OutputComposite as OutputComposite

# Tabular maskers
from ._tabular import Impute as Impute
from ._tabular import Independent as Independent
from ._tabular import Partition as Partition
from ._text import Text as Text
