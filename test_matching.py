from typing import Dict, Tuple, List, Union

from utils import Keyword, KeywordMatcher, Document, Language


def get_ngrams_of_matched(matched_corpus_ids, keyword_collection: Union[Dict[int, List[Keyword]], List[Document]]):
    print(matched_corpus_ids)
    ngrams = set()

    # for instance in keyword_collection:
    #     if isinstance(keyword_collection, dict):
    #         keywords = keyword_collection[instance]
    #     elif isinstance(keyword_collection, list):
    #         keywords = instance.keywords

    for instance in keyword_collection:
        if isinstance(keyword_collection, dict):
            keywords = keyword_collection[instance]
            identifier = instance
        elif isinstance(keyword_collection, list):
            keywords = instance.keywords
            identifier = instance.doc_id
        else:
            raise NotImplementedError("Not supported type!")

        if identifier in matched_corpus_ids:
            # todo: add real ngrams?
            # ngrams.update(document.text.split())
            candidates = [keyword.german_translation for keyword in keywords] + \
                         [keyword.english_translation for keyword in keywords]
            ngrams.update(candidates)

    return ngrams


def get_common_keyword_vocab(matches: Dict[Keyword, Tuple[List[int], List[int]]], keyword_collection_1,
                             keyword_collection_2):
    matched_ids_src1 = set()
    matched_ids_src2 = set()
    for matched_keyword, documents in matches.items():
        src1, src2 = documents
        matched_ids_src1.update(src1)
        matched_ids_src2.update(src2)

    ngrams_1 = get_ngrams_of_matched(matched_ids_src1, keyword_collection_1)
    ngrams_2 = get_ngrams_of_matched(matched_ids_src2, keyword_collection_2)
    print(ngrams_1.difference(ngrams_2))
    print(ngrams_2.difference(ngrams_1))
    print(ngrams_1.intersection(ngrams_2))
    return ngrams_1.intersection(ngrams_2)


def main():
    grouped_dict1 = {2020: [Keyword(german_translation="klimawandel", english_translation="climate change"),
                            Keyword(german_translation="nachhaltigkeit", english_translation="sustainability"),
                            Keyword(german_translation="Wasser", english_translation="water")]}

    grouped_dict2 = {2011: [Keyword(german_translation="nachhaltigkeit", english_translation="sustainability"),
                            Keyword(german_translation="klimawandel", english_translation="climate change"),
                            Keyword(german_translation="Erde", english_translation="Earth")],

                     2018: [Keyword(german_translation="Verkehr", english_translation="traffic"),
                            Keyword(german_translation="Geld", english_translation="money"),
                            Keyword(german_translation="Auto", english_translation="car")],

                     2017: [Keyword(german_translation="Geld", english_translation="money"),
                            Keyword(german_translation="Wasser", english_translation="sea"),
                            Keyword(german_translation="kernfusion", english_translation="fusion")]

                     }

    km = KeywordMatcher()
    print(km.match_corpora(grouped_dict1, grouped_dict2))

    documents1 = [
        Document(doc_id="B", keywords=[Keyword(german_translation="klimawandel", english_translation="climate change"),
                                       Keyword(german_translation="nachhaltigkeit",
                                               english_translation="sustainability"),
                                       Keyword(german_translation="Wasser", english_translation="water")],
                 date=2020, language=Language.DE, text="climate change pollution clean energy sustainability water")]

    documents2 = [Document(doc_id="A",
                           keywords=[Keyword(german_translation="nachhaltigkeit", english_translation="sustainability"),
                                     Keyword(german_translation="klimawandel", english_translation="climate change"),
                                     Keyword(german_translation="Erde", english_translation="Earth")],
                           date=2011, language=Language.EN, text="Tiere zuhause Erde"),

                  Document(doc_id="C", keywords=[Keyword(german_translation="Verkehr", english_translation="traffic"),
                                                 Keyword(german_translation="Geld", english_translation="money"),
                                                 Keyword(german_translation="Auto", english_translation="car")],
                           date=2018, language=Language.EN, text="Verkehr ist wichtig besonders das Auto"),
                  Document(doc_id="D", keywords=[Keyword(german_translation="Geld", english_translation="money"),
                                                 Keyword(german_translation="Wasser", english_translation="sea"),
                                                 Keyword(german_translation="kernfusion",
                                                         english_translation="fusion")],
                           date=2017, language=Language.EN, text="Wasser kühlt Kernfusion aber kostet Geld")
                  ]

    km = KeywordMatcher()
    print(km.match_corpora(documents1, documents2))
    matches, _ = km.match_corpora(documents1, documents2)
    get_common_keyword_vocab(matches, documents1, documents2)

    matches, _ = km.match_corpora(grouped_dict1, grouped_dict2)
    get_common_keyword_vocab(matches, grouped_dict1, grouped_dict2)


if __name__ == '__main__':
    main()
