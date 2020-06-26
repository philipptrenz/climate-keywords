import os

import pandas as pd


def calculate_precision_of_file(path, top_k):
    complete_df = pd.read_csv(path, nrows=top_k)

    complete_df.columns = ["tf_keyword", "tf_score", "tf_label", "df_keyword", "df_score", "df_label"]
    true_tf_df = complete_df[complete_df['tf_label'] == 1]
    true_df_df = complete_df[complete_df['df_label'] == 1]

    return len(true_tf_df.index)/len(complete_df.index), len(true_df_df.index)/len(complete_df.index)


def main():
    top_k = 100
    root = 'data/evaluation'
    files = [
        'rake_state_of_the_union_abstract'
    ]

    results = []
    for file in files:
        tf_prec, df_prec = calculate_precision_of_file(os.path.join(root, f'{file}_an.csv'), top_k)
        # print(file.replace('_', ' '), tf_prec, df_prec)
        results.append((file.replace('_', ' '), tf_prec, df_prec))

    result_df = pd.DataFrame(results, columns=['Config', 'TF Precision', 'DF Precision'])
    print(result_df)
    result_df.to_csv(os.path.join(root, 'precision.csv'), index=False)


if __name__ == '__main__':
    main()
