import argparse
from utils import ConfigLoader
import json


def main():
    parser = argparse.ArgumentParser(description='Translates keywords in keyword files of given paths')
    parser.add_argument('-p', '--paths', help='Paths of keyword files to translate', nargs='+',
                        default=['data/state_of_the_union_corpus_rake_keywords.json'])
    args = vars(parser.parse_args())

    config = ConfigLoader.get_config()
    paths = args['paths']

    for path in paths:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
            for doc_id, keywords in data.items():
                for keyword in keywords:
                    en_translation = keyword["english_translation"]
                    ger_translation = keyword["german_translation"]
                    if en_translation is None:
                        pass
                    if ger_translation is None:
                        pass


if __name__ == '__main__':
    main()
