import argparse

from corpora_processing import KeyPhraseExtractor
from collections import namedtuple

from utils import ConfigLoader, Corpus, Language, KeywordTranslator


def main():
    parser = argparse.ArgumentParser(description='Extracts keywords for given algorithm on given corpora')
    parser.add_argument('-a', '--algorithm', help='Algorithm to use like rake or tfidf', default="rake")
    parser.add_argument('-c', '--corpora', help='Corpora to annotate as list', nargs='+', default=['state_of_the_union'])
    parser.add_argument('-t', '--translate', help='Translate keywords', action='store_true')
    args = vars(parser.parse_args())
    print(args)

    config = ConfigLoader.get_config()

    algorithm = "rake"
    translate_keywords = False
    chosen_corpora = ['state_of_the_union']

    PathMetaData = namedtuple('PathMetaData', 'path corpus_name language')
    paths_and_meta_data = [
        PathMetaData(config["corpora"]["bundestag_corpus"], "bundestag", Language.DE),
        PathMetaData(config["corpora"]["abstract_corpus"], "abstract", Language.EN),
        PathMetaData(config["corpora"]["sustainability_corpus"], "sustainability", Language.EN),
        PathMetaData(config["corpora"]["state_of_the_union_corpus"], "state_of_the_union", Language.EN),
        PathMetaData(config["corpora"]["united_nations_corpus"], "united_nations", Language.EN)
    ]
    paths_and_meta_data = [path_meta for path_meta in paths_and_meta_data if path_meta.corpus_name in chosen_corpora]

    use = {
        "rake": KeyPhraseExtractor.rake,
        "tfidf_skl": KeyPhraseExtractor.tfidf_skl,
    }

    keyword_extractor = use[algorithm]

    corpora = [Corpus(source=path_meta.path, name=path_meta.corpus_name, language=path_meta.language)
               for path_meta in paths_and_meta_data]

    if translate_keywords:
        kwt = KeywordTranslator(cache_file=config["translator"]["cache_file"])
        # todo: add translation

    for corpus, path_meta in zip(corpora, paths_and_meta_data):
        keyword_extractor(corpus=corpus)
        new_path = str(path_meta.path).replace('.json', f"_{algorithm}.json")
        corpus.save_corpus(new_path)


if __name__ == '__main__':
    main()
