import argparse
import json

from corpora_processing import KeyPhraseExtractor
from collections import namedtuple

from utils import ConfigLoader, Corpus, Language, KeywordTranslator, Keyword


def main():
    parser = argparse.ArgumentParser(description='Extracts keywords for given algorithm on given corpora')
    parser.add_argument('-a', '--algorithm', help='Algorithm to use like rake or tfidf', default="rake")
    parser.add_argument('-c', '--corpora', help='Corpora to annotate as list', nargs='+',
                        default=['state_of_the_union'])
    parser.add_argument('-t', '--translate', help='Translate keywords', action='store_true')
    args = vars(parser.parse_args())

    config = ConfigLoader.get_config()

    #  remove and use actual args
    algorithm = args['algorithm']
    translate_keywords = False  #args['translate']
    chosen_corpora = args['corpora']
    assign_keywords = False

    PathMetaData = namedtuple('PathMetaData', 'path corpus_name language')
    paths_and_meta_data = [
        PathMetaData(config["corpora"]["state_of_the_union_corpus"], "state_of_the_union", Language.EN),
        PathMetaData(config["corpora"]["bundestag_corpus"], "bundestag", Language.DE),
        PathMetaData(config["corpora"]["abstract_corpus"], "abstract", Language.EN),
        PathMetaData(config["corpora"]["sustainability_corpus"], "sustainability", Language.EN),
        PathMetaData(config["corpora"]["united_nations_corpus"], "united_nations", Language.EN)
    ]

    paths_and_meta_data = [path_meta for path_meta in paths_and_meta_data if path_meta.corpus_name in chosen_corpora]

    use = {
        "rake": KeyPhraseExtractor.rake,
        "tfidf_skl": KeyPhraseExtractor.tfidf_skl,
        "tfidf_pke": KeyPhraseExtractor.tfidf_pke,
        "text_rank": KeyPhraseExtractor.text_rank_pke,
        "yake": KeyPhraseExtractor.yake_pke,
        "single_rank": KeyPhraseExtractor.single_rank_pke,
        "topic_rank": KeyPhraseExtractor.topic_rank_pke,
        "topical_page_rank": KeyPhraseExtractor.topical_page_rank_pke,
        "position_rank": KeyPhraseExtractor.position_rank_pke,
        "multipartite_rank": KeyPhraseExtractor.multipartite_rank_pke

    }

    keyword_extractor = use[algorithm]

    print(f'Applied {algorithm} on {chosen_corpora} with translation={translate_keywords}')

    corpora = [Corpus(source=path_meta.path, name=path_meta.corpus_name, language=path_meta.language)
               for path_meta in paths_and_meta_data]

    for corpus, path_meta in zip(corpora, paths_and_meta_data):
        if translate_keywords:
            kwt = KeywordTranslator(cache_file=config["translator"]["cache_file"])
            corpus.translate_keywords(kwt)
            # todo: add translation
        keyword_extractor(corpus=corpus)
        if assign_keywords:
            new_path = str(path_meta.path).replace('.json', f"_{algorithm}.json")
            corpus.save_corpus(new_path)
        else:
            new_path = str(path_meta.path).replace('.json', f"_{algorithm}_keywords.json")
            keyword_storage = {doc_id: document.keywords for doc_id, document in corpus.documents.items()}
            with open(new_path, 'w', encoding='utf-8') as f:
                json.dump(keyword_storage, f, ensure_ascii=False, indent=1, default=lambda o: o.__dict__)
            print('wrote file')


if __name__ == '__main__':
    main()
