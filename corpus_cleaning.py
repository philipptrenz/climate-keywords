import os

from utils import DataHandler, ConfigLoader


def create_new_filepath_uncleaned(file_path):
    path, filename = os.path.split(file_path)
    name, ending = os.path.splitext(filename)
    return os.path.join(path, name + '_uncleaned' + ending)


def cleaning_abstracts(config, overwrite=True):
    corpus = DataHandler.load_corpus(config["corpora"]["abstract_corpus"])
    print("1", len(corpus))
    corpus = [d for d in corpus if d.date and len(str(d.date)) == 4 and d.date.isnumeric()]
    for d in corpus:
        d.date = int(d.date)
    print("2", len(corpus))

    if not overwrite:
        os.rename(src=config["corpora"]["abstract_corpus"], dst=create_new_filepath_uncleaned(config["corpora"]["abstract_corpus"]))

    DataHandler.save_corpus(corpus, config["corpora"]["abstract_corpus"])


def cleaning_sustainability(config, overwrite=True):
    corpus = DataHandler.load_corpus(config["corpora"]["sustainability_corpus"])
    print("1", len(corpus))
    corpus = [d for d in corpus if d.date]
    for d in corpus:
        d.date = int(d.date)
    print("2", len(corpus))

    if not overwrite:
        os.rename(src=config["corpora"]["sustainability_corpus"], dst=create_new_filepath_uncleaned(config["corpora"]["sustainability_corpus"]))

    DataHandler.save_corpus(corpus, config["corpora"]["sustainability_corpus"])


def cleaning_bundestag(config, overwrite=True):
    corpus = DataHandler.load_corpus(config["corpora"]["bundestag_corpus"])
    corpus = [d for d in corpus if d.date]
    print("1", len(corpus))
    for d in corpus:
        d.date = int(d.date)
    print("2", len(corpus))

    if not overwrite:
        os.rename(src=config["corpora"]["bundestag_corpus"], dst=create_new_filepath_uncleaned(config["corpora"]["bundestag_corpus"]))

    DataHandler.save_corpus(corpus, config["corpora"]["bundestag_corpus"])


def main():
    # load configuration parameters from config file
    config = ConfigLoader.get_config()

    # deletes unusable documents and replaces date with year int
    cleaning_abstracts(config)
    cleaning_sustainability(config)
    cleaning_bundestag(config)


if __name__ == '__main__':
    main()
