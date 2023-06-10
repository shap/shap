try:
    import matplotlib
except ImportError:
    raise ImportError("matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this.")

from ._bar import bar
from ._beeswarm import beeswarm
from ._benchmark import benchmark
from ._decision import decision
from ._embedding import embedding
from ._force import force, initjs
from ._group_difference import group_difference
from ._heatmap import heatmap
from ._image import image, image_to_text
from ._monitoring import monitoring
from ._partial_dependence import partial_dependence
from ._scatter import scatter
from ._text import text
from ._violin import violin
from ._waterfall import waterfall
