import argparse
import json

from collections import namedtuple, defaultdict
from typing import Dict, List

from tqdm import tqdm

from utils import ConfigLoader, Corpus, Language, KeywordMatcher, Keyword


def modify_path(path: str, algorithm: str, use_unassigned: bool):
    if use_unassigned:
        return path
    return path.replace('.json', f'_{algorithm}.json')


def modify_keyword_path(path: str, algorithm: str):
    return path.replace('.json', f'_{algorithm}_keywords.json')


def modify_meta_path(path: str):
    return path.replace('_keywords.json', f'_inverse_keywords.json')


def revert_list_dict(list_dict: Dict[str, List[Keyword]]):
    reverted = defaultdict(list)
    for key, values in tqdm(list_dict.items(), desc="Revert dict", total=len(list_dict)):
        if values:
            for value in values:
                if value.english_translation:
                    reverted[value.english_translation].append(key)
                if value.german_translation:
                    reverted[value.german_translation].append(key)
    return reverted


def get_keyword_counts(keyword_dict: Dict[str, List[Keyword]], corpus: Corpus):
    result = defaultdict(list)
    for keyword, doc_ids in tqdm(keyword_dict.items(), desc="Get keyword counts", total=len(keyword_dict)):
        count_dict = {}
        for doc_id in doc_ids:
            text = str(corpus.documents[doc_id].text).lower()
            count_dict[doc_id] = text.count(keyword)
        result[keyword].append(count_dict)
    return result


def get_visualization_meta_data(config, algorithm, chosen_corpora, use_unassigned):
    output_path = f"data/evaluation/{algorithm}_{'_'.join(chosen_corpora)}.csv"
    PathMetaData = namedtuple('PathMetaData', 'path corpus_name language')
    paths_and_meta_data = [
        PathMetaData(modify_path(config["corpora"]["bundestag_corpus"], algorithm, use_unassigned),
                     "bundestag", Language.DE),
        PathMetaData(modify_path(config["corpora"]["abstract_corpus"], algorithm, use_unassigned),
                     "abstract", Language.EN),
        PathMetaData(modify_path(config["corpora"]["sustainability_corpus"], algorithm, use_unassigned),
                     "sustainability", Language.EN),
        PathMetaData(modify_path(config["corpora"]["state_of_the_union_corpus"], algorithm, use_unassigned),
                     "state_of_the_union", Language.EN),
        PathMetaData(modify_path(config["corpora"]["united_nations_corpus"], algorithm, use_unassigned),
                     "united_nations", Language.EN)
    ]

    keyword_files = [
        PathMetaData(modify_keyword_path(config["corpora"]["bundestag_corpus"], algorithm),
                     "bundestag", Language.DE),
        PathMetaData(modify_keyword_path(config["corpora"]["abstract_corpus"], algorithm),
                     "abstract", Language.EN),
        PathMetaData(modify_keyword_path(config["corpora"]["sustainability_corpus"], algorithm),
                     "sustainability",
                     Language.EN),
        PathMetaData(modify_keyword_path(config["corpora"]["state_of_the_union_corpus"], algorithm),
                     "state_of_the_union", Language.EN),
        PathMetaData(modify_keyword_path(config["corpora"]["united_nations_corpus"], algorithm),
                     "united_nations", Language.EN)
    ]

    paths_and_meta_data = [path_meta for path_meta in paths_and_meta_data if path_meta.corpus_name in chosen_corpora]
    keyword_files = [path_meta for path_meta in keyword_files if path_meta.corpus_name in chosen_corpora]

    print(f'Invert {algorithm} keywords on {chosen_corpora}')

    corpora = [Corpus(source=path_meta.path, name=path_meta.corpus_name, language=path_meta.language)
               for path_meta in paths_and_meta_data]

    print('invert keyword dicts...')
    for corpus, keyword_path in zip(corpora, keyword_files):
        with open(keyword_path.path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
            doc_id2keywords = {doc_id: Keyword.parse_keywords(keywords)
                               for doc_id, keywords in tqdm(data.items(),
                                                            desc="build index...",
                                                            total=len(data))}

        keywords2doc_id = revert_list_dict(doc_id2keywords)

        meta_data = get_keyword_counts(keywords2doc_id, corpus)

        with open(modify_meta_path(keyword_path.path), 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=1)


def main():
    parser = argparse.ArgumentParser(description='Extracts keywords for given algorithm on given corpora')
    parser.add_argument('-a', '--algorithm', help='Algorithm to use like rake or tfidf', default="rake")
    parser.add_argument('-c', '--corpora', help='Two Corpora to operate on ', nargs='+',
                        default=['state_of_the_union', 'abstract'])
    parser.add_argument('-k', '--top_k', help='number of elements for output', type=int, default=100)
    args = vars(parser.parse_args())

    config = ConfigLoader.get_config()

    #  remove and use actual args
    chosen_corpora = ['bundestag']  # args['corpora']
    algorithm = "rake"  # args['algorithm']

    get_visualization_meta_data(config, algorithm, chosen_corpora, use_unassigned=True)


if __name__ == '__main__':
    main()
