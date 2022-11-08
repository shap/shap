try:
    import matplotlib
except ImportError:
    raise ImportError("matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this.")

from ._bar import bar
from ._heatmap import heatmap
from ._decision import decision
from ._scatter import scatter
from ._embedding import embedding
from ._force import force, initjs
from ._group_difference import group_difference
from ._image import image, image_to_text
from ._monitoring import monitoring
from ._partial_dependence import partial_dependence
from ._beeswarm import beeswarm
from ._violin import violin
from ._text import text
from ._waterfall import waterfall
from ._benchmark import benchmark

