import argparse
from collections import defaultdict
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(description='Extracts annotations of annotated files')
    parser.add_argument('-i', '--in', help='in directory', default="data/evaluation")
    parser.add_argument('-o', '--out', help='outfile', default="data/evaluation/annotated_keywords.csv")
    args = vars(parser.parse_args())
    directory = args['in']
    output_path = args['out']
    exclude = ['precision.csv']

    dir_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    dir_files = [file for file in dir_files if file not in exclude and file.endswith('_an.csv')]
    print(dir_files)
    erg_dict = defaultdict(set)
    for file in dir_files:
        path = os.path.join(directory, file)
        df = pd.read_csv(path, header=None)
        df.columns = ['Word_1', 'Score_1', 'Label_1', 'Word_2', 'Score_2', 'Label_2']
        for i, row in df.iterrows():
            erg_dict[row['Word_1']].add(row['Label_1'])
            erg_dict[row['Word_2']].add(row['Label_2'])

            if len(erg_dict[row['Word_1']]) > 1:
                print(f'{row["Word_1"]} of {file} is labeled ambiguous.')
                erg_dict[row['Word_1']].clear()
                erg_dict[row['Word_1']].add('ambiguous')
            if len(erg_dict[row['Word_2']]) > 1:
                print(f'{row["Word_2"]} of {file} is labeled ambiguous.')
                erg_dict[row['Word_2']].clear()
                erg_dict[row['Word_2']].add('ambiguous')

    erg_dict = {word: next(iter(label)) for word, label in erg_dict.items()}

    sorted_ergs = list(sorted(erg_dict.items(), key=lambda x: (x[1], x[0]), reverse=False))
    erg_df = pd.DataFrame(sorted_ergs, columns=['Keyword', 'Label'])

    erg_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
