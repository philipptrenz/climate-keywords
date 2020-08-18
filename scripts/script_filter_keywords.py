import argparse
import json

from tqdm import tqdm

from collections import defaultdict


def modify_path(old_path: str):
    return old_path.replace('.json', '_old.json')


def main():
    parser = argparse.ArgumentParser(description='Extracts keywords for given algorithm on given corpora')
    parser.add_argument('-p', '--paths', help='Paths of keyword files to translate', nargs='+',
                        default=['data/bundestag_corpus_rake_keywords.json'])
    args = vars(parser.parse_args())
    paths = args['paths']

    for path in paths:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        with open(modify_path(path), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1, default=lambda o: o.__dict__)
        print('copied file')

        ger_counter = defaultdict(int)
        en_counter = defaultdict(int)
        for doc_id, keywords in data.items():
            if keywords:
                for keyword in keywords:
                    ger_counter[keyword["german_translation"]] += 1
                    en_counter[keyword["english_translation"]] += 1

        ger_counter = {keyword_string: counter_var for keyword_string, counter_var in tqdm(ger_counter.items())
                       if counter_var > 50 and counter_var < 5000 and keyword_string != '' and keyword_string is not None}

        en_counter = {keyword_string: counter_var for keyword_string, counter_var in tqdm(en_counter.items())
                      if counter_var > 15 and counter_var < 7000 and keyword_string != '' and keyword_string is not None}
        print(ger_counter)
        print(len(ger_counter.keys()))
        filtered = {doc_id: [keyword for keyword in keywords
                             if keyword['german_translation'] in ger_counter
                             # or keyword['english_translation'] in en_counter
                             ]
                    for doc_id, keywords in data.items() if keywords}

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, ensure_ascii=False, indent=1, default=lambda o: o.__dict__)
        print(f'wrote file {path}')


if __name__ == '__main__':
    main()
