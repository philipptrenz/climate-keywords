from typing import List, Set, Dict, NamedTuple
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


if __name__ == '__main__':

    config = ConfigLoader.get_config()

    # read and parse data
    bundestag_corpus = DataHandler.get_bundestag_speeches(dir=config["datasets"]["bundestag"]["directory"])
    Preprocessor.preprocess(bundestag_corpus)
    DataHandler.save_corpus(bundestag_corpus, "bundestag_corpus.json")
    del bundestag_corpus

    Document.save_corpus(bundestag_corpus, "bundestag_corpus.json")
    #Document.save_corpus(sustainability_corpus, "sustainability_corpus.json")
    #Document.save_corpus(abstract_corpus, "abstract_corpus.json")
