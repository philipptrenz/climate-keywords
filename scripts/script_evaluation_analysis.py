import os

import pandas as pd


def get_meta_from_path(path: str):
    path = path.replace('_an.csv', '')
    data_sets = ["abstract", "state_of_the_union", "bundestag", "sustainability", "un"]
    yearwise = False
    if "yearwise" in path:
        yearwise = True
    path = path.replace('yearwise', '')

    actual_data_sets = []
    for data_set in data_sets:
        if data_set in path:
            actual_data_sets.append(data_set)

    for data_set in actual_data_sets:
        path = path.replace(data_set, '')

    algorithm = path.replace('_', '')

    return algorithm, actual_data_sets[0], actual_data_sets[1], yearwise


def calculate_precision_of_file(path, top_k):
    complete_df = pd.read_csv(path, nrows=top_k)

    complete_df.columns = ["tf_keyword", "tf_score", "tf_label", "df_keyword", "df_score", "df_label"]
    true_tf_df = complete_df[complete_df['tf_label'] == 1]
    true_df_df = complete_df[complete_df['df_label'] == 1]

    return len(true_tf_df.index) / len(complete_df.index), len(true_df_df.index) / len(complete_df.index)


def calculate_recall_of_file(path, top_k, annotated_list_path):
    complete_df = pd.read_csv(path, nrows=top_k)

    complete_df.columns = ["tf_keyword", "tf_score", "tf_label", "df_keyword", "df_score", "df_label"]
    true_tf_df = complete_df[complete_df['tf_label'] == 1]
    true_df_df = complete_df[complete_df['df_label'] == 1]

    annotated_list_df = pd.read_csv(annotated_list_path)
    complete_positives = annotated_list_df[annotated_list_df['Label'] == 1.0]
    return len(true_tf_df.index) / len(complete_positives.index), len(true_df_df.index) / len(complete_positives.index)


def calculate_f(prec, rec):
    if (prec + rec) > 0:
        return 2 * (prec * rec) / (prec + rec)
    else:
        return 0


def main():
    top_k = 100
    root = 'data/evaluation'
    exclude = ['precision.csv']
    # files = [
    #     'rake_state_of_the_union_abstract',
    #     'rake_state_of_the_union_sustainability',
    #     'rake_state_of_the_union_sustainability_yearwise',
    #     'tfidf_skl_state_of_the_union_sustainability',
    # ]

    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    files = [file for file in files if file not in exclude and file.endswith('_an.csv')]

    results = []
    for file in files:
        algorithm, source_1, source_2, yearwise = get_meta_from_path(file)
        tf_prec, df_prec = calculate_precision_of_file(os.path.join(root, file), top_k)
        tf_rec, df_rec = calculate_recall_of_file(os.path.join(root, file), top_k, os.path.join(root, "annotated_keywords.csv"))
        tf_f = calculate_f(tf_prec, tf_rec)
        df_f = calculate_f(df_prec, df_rec)

        # print(file.replace('_', ' '), tf_prec, df_prec)
        results.append((file.replace('_', ' '), tf_prec, df_prec, tf_rec, df_rec, tf_f, df_f, algorithm, source_1, source_2, yearwise))

    result_df = pd.DataFrame(results, columns=['Config', 'TF Precision', 'DF Precision',
                                               'TF Recall', 'DF Recall',
                                               'TF F-measure', 'DF F-measure',
                                               'Algorithm', 'Source1',
                                               'Source2', "Yearwise"])
    print(result_df)
    result_df.to_csv(os.path.join(root, 'precision.csv'), index=False)


if __name__ == '__main__':
    main()
