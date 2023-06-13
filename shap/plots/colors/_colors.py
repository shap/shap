""" This defines some common colors.
"""


import numpy as np

from ._colorconv import lab2rgb, lch2lab

try:
    import matplotlib  # noqa: F401
    from matplotlib.colors import LinearSegmentedColormap

    def lch2rgb(x):
        return lab2rgb(lch2lab([[x]]))[0][0]

    # define our colors using Lch
    # note that we intentionally vary the lightness during interpolation so as to better
    # enable the eye to see patterns (since patterns are most easily recognized through
    # lightness variability)
    blue_lch = [54., 70., 4.6588]
    l_mid = 40.
    red_lch = [54., 90., 0.35470565 + 2* np.pi]
    gray_lch = [55., 0., 0.]
    blue_rgb = lch2rgb(blue_lch)
    red_rgb = lch2rgb(red_lch)
    gray_rgb = lch2rgb(gray_lch)
    white_rgb = np.array([1.,1.,1.])


    light_blue_rgb = np.array([127., 196, 252])/255
    light_red_rgb = np.array([255., 127, 167])/255

    # define a perceptually uniform color scale using the Lch color space
    reds = []
    greens = []
    blues = []
    alphas = []
    nsteps = 100
    l_vals = list(np.linspace(blue_lch[0], l_mid, nsteps//2)) + list(np.linspace(l_mid, red_lch[0], nsteps//2))
    c_vals = np.linspace(blue_lch[1], red_lch[1], nsteps)
    h_vals = np.linspace(blue_lch[2], red_lch[2], nsteps)
    for pos,l,c,h in zip(np.linspace(0, 1, nsteps), l_vals, c_vals, h_vals):
        lch = [l, c, h]
        rgb = lch2rgb(lch)
        reds.append((pos, rgb[0], rgb[0]))
        greens.append((pos, rgb[1], rgb[1]))
        blues.append((pos, rgb[2], rgb[2]))
        alphas.append((pos, 1.0, 1.0))

    red_blue = LinearSegmentedColormap('red_blue', {
        "red": reds,
        "green": greens,
        "blue": blues,
        "alpha": alphas
    })
    red_blue.set_bad(gray_rgb, 1.0)
    red_blue.set_over(gray_rgb, 1.0)
    red_blue.set_under(gray_rgb, 1.0) # "under" is incorrectly used instead of "bad" in the scatter plot

    red_blue_no_bounds = LinearSegmentedColormap('red_blue_no_bounds', {
        "red": reds,
        "green": greens,
        "blue": blues,
        "alpha": alphas
    })

    red_blue_transparent = LinearSegmentedColormap('red_blue_no_bounds', {
        "red": reds,
        "green": greens,
        "blue": blues,
        "alpha": [(a[0], 0.5, 0.5) for a in alphas]
    })

    # define a circular version of the color scale for categorical coloring
    reds = []
    greens = []
    blues = []
    alphas = []
    nsteps = 100
    c_vals = np.linspace(blue_lch[1], red_lch[1], nsteps)
    h_vals = np.linspace(blue_lch[2], red_lch[2], nsteps)
    for pos,c,h in zip(np.linspace(0, 0.5, nsteps), c_vals, h_vals):
        lch = [blue_lch[0], c, h]
        rgb = lch2rgb(lch)
        reds.append((pos, rgb[0], rgb[0]))
        greens.append((pos, rgb[1], rgb[1]))
        blues.append((pos, rgb[2], rgb[2]))
        alphas.append((pos, 1.0, 1.0))
    c_vals = np.linspace(red_lch[1], blue_lch[1], nsteps)
    h_vals = np.linspace(red_lch[2] - 2 * np.pi, blue_lch[2], nsteps)
    for pos,c,h in zip(np.linspace(0.5, 1, nsteps), c_vals, h_vals):
        lch = [blue_lch[0], c, h]
        rgb = lch2rgb(lch)
        reds.append((pos, rgb[0], rgb[0]))
        greens.append((pos, rgb[1], rgb[1]))
        blues.append((pos, rgb[2], rgb[2]))
        alphas.append((pos, 1.0, 1.0))

    red_blue_circle = LinearSegmentedColormap('red_blue_circle', {
        "red": reds,
        "green": greens,
        "blue": blues,
        "alpha": alphas
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

    old_blue_rgb = np.array([30, 136, 229]) / 255
    old_red_rgb = np.array([255, 13, 87]) / 255

    colors = []
    for alpha in np.linspace(1, 0, 100):
        c = blue_rgb * alpha + (1 - alpha) * white_rgb
        colors.append(c)
    for alpha in np.linspace(0, 1, 100):
        c = red_rgb * alpha + (1 - alpha) * white_rgb
        colors.append(c)
    red_white_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


except ImportError:
    pass

#default_colors = ["#1E88E5", "#ff0d57", "#13B755", "#7C52FF", "#FFC000", "#00AEEF"]

#blue_rgba = np.array([0.11764705882352941, 0.5333333333333333, 0.8980392156862745, 1.0])
# blue_rgba = np.array([30, 136, 229, 255]) / 255
# blue_rgb = np.array([30, 136, 229]) / 255
# red_rgb = np.array([255, 13, 87]) / 255

# default_blue_colors = []
# tmp = blue_rgba.copy()
# for i in range(10):
#     default_blue_colors.append(tmp.copy())
#     if tmp[-1] > 0.1:
#         tmp[-1] *= 0.7
