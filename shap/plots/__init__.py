import lazy_loader as lazy

try:
    matplotlib = lazy.load("matplotlib", error_on_import=True)
except ImportError:
    raise ImportError(
        "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this."
    )

# from ._bar import bar
# from ._beeswarm import beeswarm
# from ._benchmark import benchmark
# from ._decision import decision
# from ._embedding import embedding
# from ._force import force, initjs
# from ._group_difference import group_difference
# from ._heatmap import heatmap
# from ._image import image, image_to_text
# from ._monitoring import monitoring
# from ._partial_dependence import partial_dependence
# from ._scatter import scatter
# from ._text import text
# from ._violin import violin
# from ._waterfall import waterfall

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "_bar": ["bar"],
        "_beeswarm": ["beeswarm"],
        "_benchmark": ["benchmark"],
        "_decision": ["decision"],
        "_embedding": ["embedding"],
        "_force": ["force", "initjs"],
        "_group_difference": ["group_difference"],
        "_heatmap": ["heatmap"],
        "_image": ["image", "image_to_text"],
        "_monitoring": ["monitoring"],
        "_partial_dependence": ["partial_dependence"],
        "_scatter": ["scatter"],
        "_text": ["text"],
        "_violin": ["violin"],
        "_waterfall": ["waterfall"],
    },
)


# __all__ = [
#     "bar",
#     "beeswarm",
#     "benchmark",
#     "decision",
#     "embedding",
#     "force",
#     "initjs",
#     "group_difference",
#     "heatmap",
#     "image",
#     "image_to_text",
#     "monitoring",
#     "partial_dependence",
#     "scatter",
#     "text",
#     "violin",
#     "waterfall",
# ]
