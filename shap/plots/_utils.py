from .. import Explanation
from ..utils import OpChain
from . import colors

def convert_color(color):
    try:
        color = pl.get_cmap(color)
    except:
        pass
    
    if color == "shap_red":
        color = colors.red_rgb
    elif color == "shap_blue":
        color = colors.blue_rgb
    
    return color

def convert_ordering(ordering, shap_values):
    if issubclass(type(ordering), OpChain):
        ordering = ordering.apply(Explanation(shap_values))
    if issubclass(type(ordering), Explanation):
        if "argsort" in [n for n,a in ordering.transform_history]:
            ordering = ordering.values
        else:
            ordering = ordering.argsort.flip.values
    return ordering