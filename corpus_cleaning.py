import os
import re

import numpy as np

from utils import ConfigLoader, Corpus, Language


def create_new_filepath_uncleaned(file_path):
    path, filename = os.path.split(file_path)
    name, ending = os.path.splitext(filename)
    return os.path.join(path, name + '_uncleaned' + ending)


def cleaning_abstracts(config, overwrite=True):
    corpus = Corpus(source=config["corpora"]["abstract_corpus"], language=Language.EN, name="abstract_corpus")
    # corpus = DataHandler.load_corpus(config["corpora"]["abstract_corpus"])
    print("1", len(corpus))
    corpus = Corpus([d for d in corpus.get_documents() if d.date and len(str(d.date)) == 4 and d.date.isnumeric()],
                    name=corpus.name, language=Language.EN)
    for d in corpus.get_documents():
        d.date = int(d.date)
    print("2", len(corpus))

    if not overwrite:
        os.rename(src=config["corpora"]["abstract_corpus"],
                  dst=create_new_filepath_uncleaned(config["corpora"]["abstract_corpus"]))

    corpus.save_corpus(config["corpora"]["abstract_corpus"])


def cleaning_sustainability(config, overwrite=True):
    corpus = Corpus(source=config["corpora"]["sustainability_corpus"],
                    language=Language.EN, name="sustainability_corpus")
    # corpus = DataHandler.load_corpus(config["corpora"]["sustainability_corpus"])
    print("1", len(corpus))
    corpus = Corpus(source=[d for d in corpus.documents if d.date], language=corpus.language, name=corpus.name)
    for d in corpus.get_documents():
        d.date = int(d.date)
    print("2", len(corpus))

    if not overwrite:
        os.rename(src=config["corpora"]["sustainability_corpus"],
                  dst=create_new_filepath_uncleaned(config["corpora"]["sustainability_corpus"]))

    corpus.save_corpus(config["corpora"]["sustainability_corpus"])


def cleaning_bundestag(config, overwrite=True):
    corpus = Corpus(source=config["corpora"]["bundestag_corpus"], language=Language.DE, name="bundestag_corpus")
    # corpus = DataHandler.load_corpus(config["corpora"]["bundestag_corpus"])
    corpus = Corpus(source=[d for d in corpus.get_documents() if d.date], language=corpus.language, name=corpus.name)
    print("1", len(corpus))
    for d in corpus.get_documents():
        d.date = int(d.date)
    print("2", len(corpus))

    if not overwrite:
        os.rename(src=config["corpora"]["bundestag_corpus"],
                  dst=create_new_filepath_uncleaned(config["corpora"]["bundestag_corpus"]))

    corpus.save_corpus(config["corpora"]["bundestag_corpus"])


def cleaning_un(config, overwrite=True):
    corpus = Corpus(source=config["corpora"]["united_nations_corpus"], language=Language.DE, name="united_nations_corpus")
    corpus = Corpus(source=[d for d in corpus.get_documents() if d.date], language=corpus.language, name=corpus.name)
    print("1", len(corpus))
    for d in corpus.get_documents():
        d.date = int(d.date)
    print("2", len(corpus))

    if not overwrite:
        os.rename(src=config["corpora"]["united_nations_corpus"],
                  dst=create_new_filepath_uncleaned(config["corpora"]["united_nations_corpus"]))

    corpus.save_corpus(config["corpora"]["united_nations_corpus"])


def cleaning_authors(config, overwrite=False):
    corpus_names = [
        "bundestag_corpus",
        # "sustainability_corpus",
        # "abstract_corpus"
    ]
    languages = [
        Language.DE,
        Language.EN,
        Language.EN]
    wlc = 0
    m_a = 0
    s_a = 0
    for i, corpus_name in enumerate(corpus_names):
        corpus = Corpus(source=config["corpora"][corpus_name], language=languages[i], name=corpus_name)
        # corpus = DataHandler.load_corpus(config["corpora"][corpus_name])
        for d in corpus.get_documents():
            if d.author:
                if isinstance(d.author, float) and np.isnan(d.author):
                    d.author = None
                else:
                    if corpus_name == "bundestag_corpus":
                        authors = [d.author]
                    elif corpus_name == "sustainability_corpus":
                        if isinstance(d.author, str):
                            authors = [a.strip() for a in d.author.split(',')]
                            authors = [f'{j}. {i}' for i, j in zip(authors[::2], authors[1::2])]
                        else:
                            authors = d.author
                    else:
                        if d.language != "English":
                            wlc += 1
                            continue
                        if isinstance(d.author, str):
                            authors = [a.strip() for a in d.author.split(',')]
                            authors = [f'{j}. {i}' for i, j in zip(authors[::2], authors[1::2])]
                        else:
                            authors = d.author
                        if len(authors) > 1:
                            m_a += 1
                            print(d.author, authors)
                        else:
                            s_a += 1
                    d.author = authors

        if not overwrite:
            os.rename(src=config["corpora"][corpus_name],
                      dst=create_new_filepath_uncleaned(config["corpora"][corpus_name]))

        corpus.save_corpus(config["corpora"][corpus_name])
    print(wlc, m_a, s_a)


def remove_punctuation(corpus: Corpus):
    for d in corpus.get_documents():
        res = re.sub(r"[^a-zA-ZäöüÖÄÜß\-\s.!?]", '', d.text)
        res = re.sub(r" +", ' ', res)
        d.text = res


def cleaning_punctuation(config, overwrite=False):
    corpus_names = [
        "bundestag_corpus",
        "sustainability_corpus",
        "abstract_corpus"
    ]
    languages = [
        Language.DE,
        Language.EN,
        Language.EN]
    for i, corpus_name in enumerate(corpus_names):
        corpus = Corpus(source=config["corpora"][corpus_name], language=languages[i], name=corpus_name)
        remove_punctuation(corpus)

        if not overwrite:
            os.rename(src=config["corpora"][corpus_name],
                      dst=create_new_filepath_uncleaned(config["corpora"][corpus_name]))

        corpus.save_corpus(config["corpora"][corpus_name])


def main():
    # load configuration parameters from config file
    config = ConfigLoader.get_config()

    # deletes unusable documents and replaces date with year int
    # cleaning_abstracts(config, overwrite=False)
    # cleaning_sustainability(config, overwrite=False)
    # cleaning_bundestag(config, overwrite=True)
    #
    # cleaning_authors(config, overwrite=True)
    cleaning_un(config, overwrite=False)


if __name__ == '__main__':
    main()
