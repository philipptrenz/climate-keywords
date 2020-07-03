import argparse
from collections import namedtuple

from utils import ConfigLoader, Corpus, Language


def modify_path(path: str, without_text: bool = False):
    if without_text:
        return path.replace('.json', f'_yearwise_no_text.json')
    return path.replace('.json', f'_yearwise.json')


def main():
    parser = argparse.ArgumentParser(description='Extracts keywords for given algorithm on given corpora')
    parser.add_argument('-c', '--corpora', help='Corpora to annotate as list', nargs='+',
                        default=['bundestag', 'abstract'])
    parser.add_argument('-t', '--translate', help='Translate keywords', action='store_true')
    args = vars(parser.parse_args())

    config = ConfigLoader.get_config()

    chosen_corpora = args['corpora']

    PathMetaData = namedtuple('PathMetaData', 'path corpus_name language')
    paths_and_meta_data = [
        PathMetaData(config["corpora"]["state_of_the_union_corpus"], "state_of_the_union", Language.EN),
        PathMetaData(config["corpora"]["bundestag_corpus"], "bundestag", Language.DE),
        PathMetaData(config["corpora"]["abstract_corpus"], "abstract", Language.EN),
        PathMetaData(config["corpora"]["sustainability_corpus"], "sustainability", Language.EN),
        PathMetaData(config["corpora"]["united_nations_corpus"], "united_nations", Language.EN)
    ]

    paths_and_meta_data = [path_meta for path_meta in paths_and_meta_data if path_meta.corpus_name in chosen_corpora]

    print(f'Yearwise of {chosen_corpora}')

    corpora = [Corpus(source=path_meta.path, name=path_meta.corpus_name, language=path_meta.language)
               for path_meta in paths_and_meta_data]

    corpora = [corpus.year_wise_pseudo_documents() for corpus in corpora]

    for corpus, path_meta in zip(corpora, paths_and_meta_data):
        corpus.save_corpus(modify_path(path_meta.path))
        corpus.save_corpus_without_text(modify_path(path_meta.path, without_text=True))


if __name__ == '__main__':
    main()
