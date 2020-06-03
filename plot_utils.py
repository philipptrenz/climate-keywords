from typing import Callable, List, Union
from matplotlib import pyplot as plt


def simple_bar_histogram(bin_data: List[Union[int, str]], count_data: List[int]):
    plt.bar(x=bin_data, height=count_data, color='lawngreen', align='center', alpha=0.5)
    plt.show()


def multi_bar_histogram(multiple_bin_data: List[List[Union[int, str]]],
                        multiple_count_data: List[List[int]],
                        labels: List[str],
                        normalize: bool = True,
                        corpus_lengths: List[int] = None):
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
    plt.show()