from typing import List, Union
from matplotlib import pyplot as plt


def simple_bar_histogram(bin_data: List[Union[int, str]], count_data: List[int], y_scale: str = "linear"):
    """
    plots a simple bar histogram
    :param bin_data: data for x axis
    :param count_data: data for y axis
    :param y_scale: the scale of the y axis(linear vs log)
    :return:
    """
    plt.bar(x=bin_data, height=count_data, color='lawngreen', align='center', alpha=0.5)
    plt.yscale(y_scale)
    plt.show()


def multi_bar_histogram(multiple_bin_data: List[List[int]],
                        multiple_count_data: List[List[int]],
                        labels: List[str],
                        normalize: bool = True,
                        corpus_lengths: List[int] = None,
                        y_scale: str = "linear"):
    """
    plots a multi bar histogram from different datasources
    example:
    multi_bar_histogram(mult_years, mult_counts, labels=["bundestag", "abstracts", "sustainability"],
    normalize=True, corpus_lengths=[877973, 407961, 221034])
    :param multiple_bin_data: numeric data of different series organized in lists for x axis
    :param multiple_count_data: count data of different series organized in lists for y axis
    :param labels: labels for the different series
    :param normalize: normalizes the count data by sum of series or other provided values with corpus_lengths
    :param corpus_lengths: provided corpus lengths for normalization
    :param y_scale: the scale of the y axis(linear vs log)
    :return:
    """
    c = []
    if normalize:
        for i, count_data in enumerate(multiple_count_data):
            if corpus_lengths:
                sum_counts = corpus_lengths[i]
            else:
                sum_counts = sum(count_data)
            c.append([count / sum_counts for count in count_data])
    else:
        c = multiple_count_data
    for bin_data, count_data, label in zip(multiple_bin_data, c, labels):
        plt.bar(x=bin_data, height=count_data, align='center', alpha=0.5, label=label)
    plt.legend(loc='best')
    plt.yscale(y_scale)
    plt.show()
