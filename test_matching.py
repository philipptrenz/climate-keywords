from utils import Keyword, KeywordMatcher

def main():
    grouped_dict1 = {2020: [Keyword(german_translation="Affe", english_translation="monkey"),
    Keyword(german_translation="Affe", english_translation="ape"),
    Keyword(german_translation="Wasser", english_translation="water")]}

    grouped_dict2 = {2019: [Keyword(german_translation="Affe", english_translation="Monkey"),
                            Keyword(german_translation="Affen", english_translation="apes"),
                            Keyword(german_translation="Erde", english_translation="Earth")],
                     2018: [Keyword(german_translation="Licht", english_translation="Light"),
                            Keyword(german_translation="Wasser", english_translation="sea"),
                            Keyword(german_translation="Erde", english_translation="earth")],

                     }

    km = KeywordMatcher()
    print(km.match_grouped_dicts(grouped_dict1, grouped_dict2))


if __name__ == '__main__':
    main()