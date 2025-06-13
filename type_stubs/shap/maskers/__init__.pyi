# Type stubs for shap.maskers
from ._composite import Composite as Composite
from ._fixed import Fixed as Fixed
from ._fixed_composite import FixedComposite as FixedComposite
from ._image import Image as Image
from ._masker import Masker as Masker
from ._output_composite import OutputComposite as OutputComposite
from ._tabular import Independent as Independent
from ._tabular import Partition as Partition
from ._text import Text as Text

# Legacy aliases
Impute = Independent
Tabular = Independent
