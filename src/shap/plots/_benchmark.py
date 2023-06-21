import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np

from . import colors

xlabel_names = {
    "remove absolute": "Fraction removed",
    "remove positive": "Fraction removed",
    "remove negative": "Fraction removed",
    "keep absolute": "Fraction kept",
    "keep positive": "Fraction kept",
    "keep negative": "Fraction kept",
    "explanation error": "Explanation error as std dev.",
    "compute time": "Seconds per. sample"
}

def benchmark(benchmark, show=True):
    """ Plot a BenchmarkResult or list of such results.
    """

    if hasattr(benchmark, "__iter__"):
        benchmark = list(benchmark)

        # see if we have multiple metrics or just a single metric
        single_metric = True
        metric_name = None
        has_curves = True
        for b in benchmark:
            if metric_name is None:
                metric_name = b.metric
            elif metric_name != b.metric:
                single_metric = False

            if b.curve_x is None or b.curve_y is None:
                has_curves = False

        methods = list({b.method for b in benchmark})
        methods.sort()
        method_color = {}

        for i, m in enumerate(methods):
            method_color[m] = colors.red_blue_circle(i/len(methods))

        # plot a single metric benchmark result
        if single_metric and has_curves:
            benchmark.sort(key=lambda b: -b.value_sign * b.value)
            for i, b in enumerate(benchmark):
                plt.fill_between(
                    b.curve_x, b.curve_y - b.curve_y_std, b.curve_y + b.curve_y_std,
                    color=method_color[b.method], alpha=0.1, linewidth=0
                )
            for i, b in enumerate(benchmark):
                plt.plot(
                    b.curve_x, b.curve_y,
                    color=method_color[b.method],
                    linewidth=2,
                    label=b.method + f" ({b.value:0.3})"
                )
                #plt.fill_between(b.curve_x, b.curve_y - b.curve_y_std, b.curve_y + b.curve_y_std, color=method_color[b.method], alpha=0.2)
                ax = plt.gca()
            ax.set_xlabel(xlabel_names[metric_name], fontsize=13)
            ax.set_ylabel("Model output", fontsize=13)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.title(metric_name.capitalize())
            plt.legend(fontsize=11)
            if show:
                plt.show()

        elif single_metric:
            benchmark.sort(key=lambda b: -b.value_sign * b.value)

            values = np.array([b.value for b in benchmark])
            total_width = 0.7
            bar_width = total_width
            # for i, b in enumerate(benchmark):
            #     ypos_offset = 0#- ((i - len(values) / 2) * bar_width + bar_width / 2)
            plt.barh(
                np.arange(len(values)), values,
                bar_width, align='center',
                color=[method_color[b.method] for b in benchmark],
                edgecolor=(1,1,1,0.8)
            )
                # plt.plot(
                #     b.curve_x, b.curve_y,
                #     color=method_color[b.method],
                #     linewidth=2,
                #     label=b.method + f" ({b.value:0.3})"
                # )
            ax = plt.gca()
            ax.set_yticks(np.arange(len(methods)))
            ax.set_yticklabels([b.method for b in benchmark], rotation=0, fontsize=11)
            ax.set_xlabel(xlabel_names[metric_name], fontsize=13)
            # ax.set_ylabel("Model output", fontsize=13)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.title(metric_name.capitalize())
            # plt.legend(fontsize=11)
            plt.gca().invert_yaxis()
            if show:
                plt.show()

        # plot a multi-metric benchmark result
        else:

            # get a list of all the metrics in the order they first appear
            metrics = []
            for b in benchmark:
                if b.metric not in metrics:
                    metrics.append(b.metric)

            # compute normalized values
            max_value = {n: -np.inf for n in metrics}
            min_value = {n: np.inf for n in metrics}
            for b in benchmark:
                if max_value[b.metric] < b.value_sign * b.value:
                    max_value[b.metric] = b.value_sign * b.value
                if min_value[b.metric] > b.value_sign * b.value:
                    min_value[b.metric] = b.value_sign * b.value
            norm_values = {}
            for b in benchmark:
                norm_values[b.full_name] = (b.value_sign * b.value - min_value[b.metric]) / (max_value[b.metric] - min_value[b.metric])

            # compute the average value for each method and sort by it
            # global_values = {}
            # global_counts = {}
            # for b in benchmark:
            #     global_values[b.method] = global_values.get(b.method, 0) + norm_values[b.full_name]
            #     global_counts[b.method] = global_counts.get(b.method, 0) + 1
            # for k in global_values:
            #     global_values[k] /= global_counts[k]

            # sort by the first and then second metric
            metric_0 = {}
            metric_1 = {}
            for b in benchmark:
                if b.metric == metrics[0]:
                    metric_0[b.method] = b.value
                elif b.metric == metrics[1]:
                    metric_1[b.method] = b.value

            methods.sort(key=lambda method: (np.round(metric_0[method], 3), metric_1[method]))
            xs = [-0.03 * (len(methods) - 1)] + list(range(len(metrics)+1))
            for i, method in enumerate(methods):
                scores = [1 - i/(len(methods)-1), 1 - i/(len(methods)-1)]
                values = [None, None]
                for metric in metrics:
                    for b in benchmark:
                        if b.method == method and b.metric == metric:
                            scores.append(norm_values[b.full_name])
                            values.append(b.value)
                plt.plot(
                    xs,
                    scores,
                    color=method_color[method],
                    label=method
                )

                for x, y, value in zip(xs, scores, values):
                    if value is None:
                        continue
                    label = f"{value:.2f}"
                    txt = plt.annotate(
                        label, # this is the text
                        (x, y), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0, -3), # distance from text to points (x,y)
                        ha='center', # horizontal alignment can be left, right or center
                        color=method_color[method],
                        fontsize=9
                    )
                    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

            ax = plt.gca()
            ax.set_yticks([1 - i / (len(methods) - 1) for i in range(0, len(methods))])
            ax.set_yticklabels(methods, rotation=0, fontsize=11)

            ax.set_xticks(np.arange(len(metrics) + 1))
            # from matplotlib import rcParams
            # rcParams['text.latex.preamble'] = [r'\boldmath']
            ax.set_xticklabels([''] + [m.capitalize() for m in metrics], rotation=45, ha='left', fontsize=11)

            ax.xaxis.tick_top()
            plt.grid(which='major', axis='x', linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('none')
            plt.xlim(xs[0], len(metrics))
            # for l in ax.get_xticklabels():
            #     l.set_fontweight('bold')
            ax.get_xticklabels()[1].set_fontweight('bold')
            # plt.gca().invert_yaxis()
            # plt.ylabel("\nAll scores are relative")
            # ax.yaxis.set_label_position("right")
            if show:
                plt.show()

    # plot a single benchmark result
    else:
        plt.fill_between(
            benchmark.curve_x, benchmark.curve_y - benchmark.curve_y_std,
            benchmark.curve_y + benchmark.curve_y_std,
            color=colors.blue_rgb, alpha=0.1, linewidth=0
        )
        plt.plot(
            benchmark.curve_x, benchmark.curve_y,
            color=colors.blue_rgb,
            linewidth=2,
            label=benchmark.method + f" ({benchmark.value:0.3})"
        )
        ax = plt.gca()
        ax.set_xlabel(xlabel_names[benchmark.metric], fontsize=13)
        ax.set_ylabel("Model output", fontsize=13)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.legend(fontsize=11)
        if show:
            plt.show()
