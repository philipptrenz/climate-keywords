import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def main():
    parser = argparse.ArgumentParser(description='Extracts keywords for given algorithm on given corpora')
    parser.add_argument('-i1', '--input1', help='Input path 1',
                        default="data/evaluation/annotated_keywords.csv")
    parser.add_argument('-i2', '--input2', help='Input path 2',
                        default="data/evaluation/unannotated_keywords_sample_pt.csv")
    args = vars(parser.parse_args())

    df_1 = pd.read_csv(args["input1"])
    df_2 = pd.read_csv(args["input2"])

    if len(df_1.index) > len(df_2.index):
        bigger_df = df_1
        smaller_df = df_2
    else:
        bigger_df = df_2
        smaller_df = df_1
    bigger_df_filtered = bigger_df.loc[bigger_df["Keyword"].isin(smaller_df["Keyword"].values)]
    bigger_df_filtered = bigger_df_filtered.sort_values("Keyword")
    smaller_df = smaller_df.sort_values("Keyword")
    labels_1 = bigger_df_filtered["Label"].values
    labels_2 = smaller_df["Label"].values
    print(cohen_kappa_score(labels_1, labels_2))


if __name__ == '__main__':
    main()
