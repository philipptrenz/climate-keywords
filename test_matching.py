from utils import Keyword, KeywordMatcher, Document


def main():
    grouped_dict1 = {2020: [Keyword(german_translation="Affe", english_translation="monkey"),
                            Keyword(german_translation="Affe", english_translation="ape"),
                            Keyword(german_translation="Wasser", english_translation="water")]}

    grouped_dict2 = {2011: [Keyword(german_translation="Pferd", english_translation="Horse"),
                            Keyword(german_translation="Affen", english_translation="apes"),
                            Keyword(german_translation="Erde", english_translation="Earth")],

                     2018: [Keyword(german_translation="Licht", english_translation="Light"),
                            Keyword(german_translation="Wasser", english_translation="sea"),
                            Keyword(german_translation="Erde", english_translation="earth")],

                     }

    km = KeywordMatcher()
    print(km.match_corpora(grouped_dict1, grouped_dict2))

    documents1 = [Document(doc_id=2020, keywords=[Keyword(german_translation="Affe", english_translation="monkey"),
                                                   Keyword(german_translation="Affe", english_translation="ape"),
                                                   Keyword(german_translation="Wasser", english_translation="water")],
                           date=2020, language="English", text="...")]

    documents2 = [Document(doc_id=2011, keywords=[Keyword(german_translation="Pferd", english_translation="Horse"),
                                                    Keyword(german_translation="Affen", english_translation="apes"),
                                                    Keyword(german_translation="Erde", english_translation="Earth")],
                           date=2011, language="English", text="..."),

                  Document(doc_id=2018, keywords=[Keyword(german_translation="Licht", english_translation="Light"),
                                                    Keyword(german_translation="Wasser", english_translation="sea"),
                                                    Keyword(german_translation="Erde", english_translation="earth")],
                           date=2018, language="English", text="...")
                  ]

    km = KeywordMatcher()
    print(km.match_corpora(documents1, documents2))


if __name__ == '__main__':
    main()