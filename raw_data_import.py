from typing import List
from tqdm import tqdm
import spacy

from utils import Document, ConfigLoader, DataHandler


class Preprocessor:

    @staticmethod
    def preprocess(documents: List[Document], lemmatize: bool = True, lower: bool = False,
                   pos_filter: list = None, remove_stopwords: bool = False,
                   remove_punctuation: bool = False):

        def token_representation(token):
            if lemmatize:
                representation = str(token.lemma_)
            else:
                representation = str(token)

            if lower:
                representation = representation.lower()
            return representation

        if documents[0].language == "German":
            lan = "German"
            nlp = spacy.load("de_core_news_sm")
        else:
            lan = "English"
            nlp = spacy.load("en_core_web_sm")
        tokenized = []
        texts = [document.text for document in documents]
        if pos_filter is None:
            for doc in tqdm(nlp.pipe(texts, disable=['parser', 'ner', 'tagger']),
                            desc=f"Preprocess documents in {lan}",
                            total=len(texts)):
                tokenized.append(
                    [token_representation(token)
                     for token in doc
                     if (not remove_stopwords or not token.is_stop)
                     and (not remove_punctuation or token.is_alpha)]
                )

        else:
            for doc in tqdm(nlp.pipe(texts, disable=['parser', 'ner']), desc="Preprocess documents", total=len(texts)):
                tokenized.append(
                    [token_representation(token)
                     for token in doc if (not remove_stopwords or not token.is_stop)
                     and (not remove_punctuation or token.is_alpha)
                     and token.pos_ in pos_filter]
                )

        for i, document in tqdm(enumerate(documents)):
            document.text = ' '.join(tokenized[i])


def parse_and_preprocess_src(data_source, corpus_destination):
    if "bundestag" in data_source.lower():
        raw_corpus = DataHandler.get_bundestag_speeches(dir=data_source)
    elif "sustainability" in data_source.lower():
        raw_corpus = DataHandler.get_sustainability_data(path=data_source)
    else:
        raw_corpus = DataHandler.get_abstracts(path=data_source)
    Preprocessor.preprocess(raw_corpus)
    DataHandler.save_corpus(raw_corpus, corpus_destination)


if __name__ == '__main__':

    config = ConfigLoader.get_config()

    # define data sources
    file_srcs = [config["datasets"]["bundestag"]["directory"],
                 config["datasets"]["abstracts"]["sustainability"],
                 config["datasets"]["abstracts"]["climate_literature"]]

    # define corpus output sources
    corpus_dests = [config["corpora"]["bundestag_corpus"],
                    config["corpora"]["sustainability_corpus"],
                    config["corpora"]["abstract_corpus"]]

    # read and parse data
    for i in range(3):
        parse_and_preprocess_src(data_source=file_srcs[i], corpus_destination=corpus_dests[i])
