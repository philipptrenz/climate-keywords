from utils import DataHandler, ConfigLoader, DocumentsFilter


def test_filter():
    config = ConfigLoader.get_config("../")

    corpora = [
        DataHandler.load_corpus(config["corpora"]["abstract_corpus"]),
        DataHandler.load_corpus(config["corpora"]["bundestag_corpus"]),
        DataHandler.load_corpus(config["corpora"]["sustainability_corpus"])
    ]

    for corpus in corpora:
        corpus = corpus[:200]

        # test text_contains
        test_text_words = ['climate', 'klima']
        test = DocumentsFilter.filter(corpus, text_contains=test_text_words)
        for t in test:
            for ttw in test_text_words:
                if ttw not in t.text:
                    assert False

        # test is_one_of_languages
        test_date_range = range(2015,2016)
        test = DocumentsFilter.filter(corpus, date_in_range=test_date_range)
        for t in test:
            if t.date not in test_date_range:
                assert False

        # test is_one_of_languages
        test_languages = ['english', 'en']
        test = DocumentsFilter.filter(corpus, is_one_of_languages=test_languages)
        for t in test:
            if t.language.lower() is not test_languages:
                assert False

        # test is_one_of_doc_ids
        test_doc_id = '0'
        test = DocumentsFilter.filter(corpus, is_one_of_doc_ids=[test_doc_id])
        for t in test:
            if test_doc_id is not t.doc_id:
                assert False

        # test has_authors
        test_author = 'test'
        test = DocumentsFilter.filter(corpus, has_authors=[test_author])
        for t in test:
            if test_author not in t.author:
                assert False

        # test has_tags
        test_tags = 'test'
        test = DocumentsFilter.filter(corpus, has_tags=[test_tags])
        for t in test:
            if test_tags not in t.tags:
                assert False

        # test is_one_of_parties
        test_parties = ["cdu", "FdP"]
        test = DocumentsFilter.filter(corpus, is_one_of_parties=[test_parties])
        for t in test:
            if t.party.lower() not in [x.lower() for x in test_parties]:
                assert False

        # test ratings_in_range
        test_rating_range = range(0, 7)
        test = DocumentsFilter.filter(corpus, ratings_in_range=test_rating_range)
        for t in test:
            if t.rating not in test_rating_range:
                assert False

        # TODO: Test for keywords

        assert True

