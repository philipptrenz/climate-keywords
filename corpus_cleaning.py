from utils import DataHandler, ConfigLoader


def cleaning_abstracts(config):
    corpus = DataHandler.load_corpus(config["corpora"]["abstract_corpus"])
    print("1", len(corpus))
    corpus = [d for d in corpus if d.date and len(str(d.date)) == 4 and d.date.isnumeric()]
    for d in corpus:
        d.date = int(d.date)
    print("2", len(corpus))
    DataHandler.save_corpus(corpus, "abstract_corpus.json")


def cleaning_sustainability(config):
    corpus = DataHandler.load_corpus(config["corpora"]["sustainability_corpus"])
    print("1", len(corpus))
    corpus = [d for d in corpus if d.date]
    for d in corpus:
        d.date = int(d.date)
    print("2", len(corpus))
    DataHandler.save_corpus(corpus, "sustainability_corpus.json")


def cleaning_bundestag(config):
    corpus = DataHandler.load_corpus(config["corpora"]["bundestag_corpus"])
    corpus = [d for d in corpus if d.date]
    print("1", len(corpus))
    for d in corpus:
        d.date = int(d.date)
    print("2", len(corpus))
    DataHandler.save_corpus(corpus, "bundestag_corpus.json")


def main():
    # load configuration parameters from config file
    config = ConfigLoader.get_config()

    # deletes unusable documents and replaces date with year int
    cleaning_abstracts(config)
    cleaning_sustainability(config)
    cleaning_bundestag(config)


if __name__ == '__main__':
    main()