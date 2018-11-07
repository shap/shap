""" This defines some common colors.
"""

import numpy as np

try:
    import matplotlib.pyplot as pl
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    red_blue = LinearSegmentedColormap('red_blue', { # #1E88E5 -> #ff0052
        'red': ((0.0, 30./255, 30./255),
                (1.0, 255./255, 255./255)),

        'green': ((0.0, 136./255, 136./255),
                  (1.0, 13./255, 13./255)),

        'blue': ((0.0, 229./255, 229./255),
                 (1.0, 87./255, 87./255)),

        'alpha': ((0.0, 1, 1),
                  (0.5, 0.3, 0.3),
                  (1.0, 1, 1))
    })

    red_blue_solid = LinearSegmentedColormap('red_blue_solid', {
        'red': ((0.0, 30./255, 30./255),
                (1.0, 255./255, 255./255)),

        'green': ((0.0, 136./255, 136./255),
                  (1.0, 13./255, 13./255)),

        'blue': ((0.0, 229./255, 229./255),
                 (1.0, 87./255, 87./255)),

        'alpha': ((0.0, 1, 1),
                  (0.5, 1, 1),
                  (1.0, 1, 1))
    })

    colors = []
    for l in np.linspace(1, 0, 100):
        colors.append((30./255, 136./255, 229./255,l))
    for l in np.linspace(0, 1, 100):
        colors.append((255./255, 13./255, 87./255,l))
    red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

    colors = []
    for l in np.linspace(0, 1, 100):
        colors.append((30./255, 136./255, 229./255,l))
    transparent_blue = LinearSegmentedColormap.from_list("transparent_blue", colors)

    colors = []
    for l in np.linspace(0, 1, 100):
        colors.append((255./255, 13./255, 87./255,l))
    transparent_red = LinearSegmentedColormap.from_list("transparent_red", colors)


except ImportError:
    pass

default_colors = ["#1E88E5", "#ff0d57", "#13B755", "#7C52FF", "#FFC000", "#00AEEF"]

#blue_rgba = np.array([0.11764705882352941, 0.5333333333333333, 0.8980392156862745, 1.0])
blue_rgba = np.array([30, 136, 229, 255]) / 255
blue_rgb = np.array([30, 136, 229]) / 255
red_rgb = np.array([255, 13, 87]) / 255

default_blue_colors = []
tmp = blue_rgba.copy()
for i in range(10):
    default_blue_colors.append(tmp.copy())
    if tmp[-1] > 0.1:
        tmp[-1] *= 0.7
