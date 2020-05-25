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

        # test tags
        test_tags= 'test'
        test = DocumentsFilter.filter(corpus, has_tags=[test_tags])
        for t in test:
            if test_tags not in t.tags:
                assert False

        assert True

