import argparse

from collections import namedtuple

from utils import ConfigLoader, Corpus, Language, KeywordMatcher


def modify_path(path: str, algorithm: str, use_unassigned: bool):
    if use_unassigned:
        return path
    return path.replace('.json', f'_{algorithm}.json')


def modify_keyword_path(path: str, algorithm: str):
    return path.replace('.json', f'_{algorithm}_keywords.json')


def evaluate_single(config, algorithm, chosen_corpora, top_k, use_unassigned):
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

    print(f'Evaluate {algorithm} on {chosen_corpora}')

    corpora = [Corpus(source=path_meta.path, name=path_meta.corpus_name, language=path_meta.language)
               for path_meta in paths_and_meta_data]

    print('assign copora...')
    for corpus, keyword_path in zip(corpora, keyword_files):
        corpus.assign_corpus_with_keywords_from_file(keyword_path.path)

    print('assigned copora')

    km = KeywordMatcher(corpora[0], corpora[1])
    km.match_corpora(lemmatize=False, simplify_result=False)
    # common = km.get_common_keyword_vocab()
    # km.get_first_mentions(common)
    common = km.get_common_keyword_vocab()
    mentions = km.get_first_mentions(common)
    count_a = 0
    count_b = 0
    count_same = 0
    for keyword, years in mentions.items():
        print(keyword, years)
        year_1, year_2 = years
        if year_1 < year_2:
            count_a += 1
        elif year_1 == year_2:
            count_same += 1
        else:
            count_b += 1

    print(count_a / len(mentions))
    print(count_b / len(mentions))
    print(count_same / len(mentions))



    keyword_counts = km.get_keyword_counts(common)
    for keyword_count in keyword_counts:
        print(keyword_count)
    return mentions, keyword_counts


def main():
    parser = argparse.ArgumentParser(description='Extracts keywords for given algorithm on given corpora')
    parser.add_argument('-a', '--algorithm', help='Algorithm to use like rake or tfidf', default="rake")
    parser.add_argument('-c', '--corpora', help='Two Corpora to operate on ', nargs='+',
                        default=['state_of_the_union', 'abstract'])
    parser.add_argument('-k', '--top_k', help='number of elements for output', type=int, default=100)
    args = vars(parser.parse_args())

    config = ConfigLoader.get_config()

    #  remove and use actual args
    chosen_corpora = ['state_of_the_union', 'sustainability']  # args['corpora']
    algorithm = "rake"  # args['algorithm']
    top_k = 100  # args['top_k']

    evaluate_single(config, algorithm, chosen_corpora, top_k, use_unassigned=True)


if __name__ == '__main__':
    main()
