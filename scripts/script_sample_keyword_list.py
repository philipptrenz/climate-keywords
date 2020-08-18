import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Extracts keywords for given algorithm on given corpora')
    parser.add_argument('-i', '--input', help='Input path', default="data/evaluation/unannotated_keywords.csv")
    parser.add_argument('-o', '--output', help='Output path', default="data/evaluation/unannotated_keywords_sample.csv")
    args = vars(parser.parse_args())

    input_path = args["input"]
    output_path = args["output"]
    df = pd.read_csv(input_path)
    total = 200
    positives = 0.65

    number_positves = int(total*positives)
    number_negatives = total-number_positves
    df_n = df.loc[df["Label"] == 0]
    df_p = df.loc[df["Label"] == 1]

    sample_p = df_p.sample(number_positves)
    sample_n = df_n.sample(number_negatives)

    sample = sample_p.append(sample_n).sort_values("Keyword")
    sample["Label"] = np.nan
    print(sample)
    sample.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
