import os

from utils import DataHandler, ConfigLoader
import numpy as np


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


def cleaning_authors(config, overwrite=False):
    corpus_names = [
        # "bundestag_corpus",
        # "sustainability_corpus",
        "abstract_corpus"
    ]
    wlc = 0
    m_a = 0
    s_a = 0
    for corpus_name in corpus_names:
        corpus = DataHandler.load_corpus(config["corpora"][corpus_name])
        for d in corpus:
            if d.author:
                if isinstance(d.author, float) and np.isnan(d.author):
                    d.author = None
                else:
                    if corpus_name == "bundestag_corpus":
                        authors = [d.author]
                    elif corpus_name == "sustainability_corpus":
                        authors = [a.strip() for a in d.author.split(',')]
                        authors = [f'{j}. {i}' for i, j in zip(authors[::2], authors[1::2])]
                    else:
                        if d.language != "English":
                            wlc += 1
                            continue
                        authors = [a.strip() for a in d.author.split(',')]
                        authors = [f'{j}. {i}' for i, j in zip(authors[::2], authors[1::2])]
                        if len(authors) > 1:
                            m_a += 1
                            print(d.author, authors)
                        else:
                            s_a += 1
                    d.author = authors

        if not overwrite:
            os.rename(src=config["corpora"][corpus_name], dst=create_new_filepath_uncleaned(config["corpora"][corpus_name]))

        DataHandler.save_corpus(corpus, config["corpora"][corpus_name])
    print(wlc, m_a, s_a)




def main():
    # load configuration parameters from config file
    config = ConfigLoader.get_config()

    # deletes unusable documents and replaces date with year int
    # cleaning_abstracts(config, overwrite=False)
    # cleaning_sustainability(config, overwrite=False)
    # cleaning_bundestag(config, overwrite=False)

    cleaning_authors(config, overwrite=False)


if __name__ == '__main__':
    main()
