import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Extracts annotations of annotated files')
    parser.add_argument('-i', '--in', help='in directory', default="data/evaluation")
    parser.add_argument('-s', '--standard', help='in directory', default="data/evaluation/annotated_keywords.csv")
    args = vars(parser.parse_args())
    directory = args['in']
    keyword_list_file = args['standard']
    exclude = ['precision.csv', 'annotated_keywords.csv', 'unannotated_keywords.csv', 'unannotated_keywords_sample.csv',
               'unannotated_keywords_sample_pt.csv']

    dir_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    dir_files = [file for file in dir_files if file not in exclude and not file.endswith('_an.csv')
                 and file.endswith('.csv') and not file.endswith('_ov.csv')]
    print(dir_files)
    label_df = pd.read_csv(keyword_list_file)

    labels = {row['Keyword']: row['Label'] for i, row in label_df.iterrows()}
    for file in dir_files:
        path = os.path.join(directory, file)
        df = pd.read_csv(path)
        # df = pd.read_csv(path, dtype={'Word_1': np.str, 'Score_1': 'float64', 'Label_1': 'Int32',
        #                               'Word_2': np.str, 'Score_2': 'float64', 'Label_2': 'Int32'})
        # df.columns = ['Word_1', 'Score_1', 'Label_1', 'Word_2', 'Score_2', 'Label_2']
        override = []
        for i, row in df.iterrows():
            label_1 = row['Label_1']
            label_2 = row['Label_2']
            if str(label_1).lower() == "nan":
                label_1 = labels.get(row['Word_1'])
                if label_1:
                    label_1 = int(label_1)

            if str(label_2).lower() == "nan":
                label_2 = labels.get(row['Word_2'])
                if label_2:
                    label_2 = int(str(label_2))

            override.append((row['Word_1'], row['Score_1'], label_1, row['Word_2'], row['Score_2'], label_2))
        override_df = pd.DataFrame(override, columns=df.columns)
        # override_df[["Label_1", "Label_2"]] = df[["Label_1", "Label_2"]].astype(pd.Int64Dtype())
        override_df.to_csv(os.path.join(directory, f"{file.replace('.csv', '_ov.csv')}"), index=False)
        print(override_df)


if __name__ == '__main__':
    main()
