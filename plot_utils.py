from typing import List, Union
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import pandas as pd


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


def heatmap(df, algorithm, yearwise=True):
    df_f = df.loc[df["Yearwise"]==yearwise]
    df_f = df_f.loc[df["Algorithm"]==algorithm]
#     display(df_f)
    res = {}
    sources = set()
    for i, row in df_f.iterrows():
        res[(row["Source1"], row["Source2"])] = (row["TF Precision"], row["DF Precision"])
        res[(row["Source2"], row["Source1"])] = (row["TF Precision"], row["DF Precision"])
        sources.add(row["Source1"])
        sources.add(row["Source2"])
    res

    sources = list(sources)
    sources.sort()
    matrix = []

    for i, source in enumerate(sources):
        inner_matrix = []
        for j, other_source in enumerate(sources):
            tup = (source, other_source)
            if i > j:
                index = 1
            else:
                index = 0
            tf_df = res.get(tup)
            if not tf_df:
                tf_df = (0, 0)

            inner_matrix.append(tf_df[index])
        matrix.append(np.array(inner_matrix))

    matrix = np.array(matrix)
    ax = plt.axes()
    ax = sns.heatmap(matrix, xticklabels=sources, yticklabels=sources, annot=True, cmap="YlGn", ax = ax)
    ax.set_title(f'{algorithm} {"yearwise" if yearwise else ""}: ↓ DF \ TF →')
    plt.show()

df = pd.read_csv('data/evaluation/precision.csv')
heatmap(df, "tfidfskl", yearwise=True)
heatmap(df, "rake", yearwise=True)
heatmap(df, "rake", yearwise=False)
heatmap(df, "tfidfskl", yearwise=False)