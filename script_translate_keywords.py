import argparse
from collections import namedtuple

from utils import ConfigLoader, Corpus, Language

import json

def main():
    # parser = argparse.ArgumentParser(description='Extracts keywords for given algorithm on given corpora')
    # parser.add_argument('-a', '--algorithm', help='Algorithm to use like rake or tfidf', default="rake")
    # parser.add_argument('-c', '--corpora', help='Corpora to annotate as list', nargs='+',
    #                     default=['state_of_the_union'])
    # parser.add_argument('-t', '--translate', help='Translate keywords', action='store_true')
    # args = vars(parser.parse_args())
    #
    # config = ConfigLoader.get_config()
    #
    # #  remove and use actual args
    # chosen_corpora = [
    #     # 'state_of_the_union',
    #     'bundestag', 'abstract', 'sustainability'
    # ]  # args['corpora']
    #
    # PathMetaData = namedtuple('PathMetaData', 'path corpus_name language')
    # paths_and_meta_data = [
    #     PathMetaData(config["corpora"]["bundestag_corpus"], "bundestag", Language.DE),
    #     PathMetaData(config["corpora"]["abstract_corpus"], "abstract", Language.EN),
    #     PathMetaData(config["corpora"]["sustainability_corpus"], "sustainability", Language.EN),
    #     PathMetaData(config["corpora"]["state_of_the_union_corpus"], "state_of_the_union", Language.EN),
    #     PathMetaData(config["corpora"]["united_nations_corpus"], "united_nations", Language.EN)
    # ]
    # paths_and_meta_data = [path_meta for path_meta in paths_and_meta_data if path_meta.corpus_name in chosen_corpora]

    paths = ['data/state_of_the_union_corpus_rake_keywords.json']
    for path in paths:
        with open(path) as f:
            data = json.load(f)
            for doc_id, keywords in data:
                en_translation = keywords["english_translation"]
                ger_translation = keywords["german_translation"]
                if en_translation is None:
                    pass
                if ger_translation is None:
                    pass


if __name__ == '__main__':
    main()
