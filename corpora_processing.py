from collections import defaultdict, Counter
from typing import List, Set, Dict, NamedTuple
import pandas as pd
from rake_nltk import Rake
from nltk.corpus import stopwords
from tqdm import tqdm
import pke
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging

from utils import Document, ConfigLoader, DataHandler


class KeyPhraseExtractor:

    @staticmethod
    def term_frequency(documents: List[Document]) -> Dict[str, List[str]]:
        pass

    @staticmethod
    def tfidf_skl(documents: List[Document]) -> Dict[str, List[str]]:
        def top_tfidf_feats(row, features, top_n=25):
            ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
            topn_ids = np.argsort(row)[::-1][:top_n]
            top_feats = [(features[i], row[i]) for i in topn_ids]
            df = pd.DataFrame(top_feats)
            df.columns = ['feature', 'tfidf']
            print(df)
            return df

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([document.text for document in documents])
        doc_id_lookup = {i: document.doc_id for i, document in enumerate(documents)}

        k = 10
        results = {}
        features = tfidf_vectorizer.get_feature_names()

        for i, doc in tqdm(enumerate(tfidf_matrix)):
            df = pd.DataFrame(doc.T.todense(), index=features,
                              columns=["tfidf"])
            top_key_words = df.sort_values(by=["tfidf"], ascending=False)[:k]
            results[doc_id_lookup[i]] = list(top_key_words.index)

        return results

    @staticmethod
    def tfidf_pke(documents: List[Document]) -> Dict[str, List[str]]:
        # 1. create a TfIdf extractor.
        extractor = pke.unsupervised.TfIdf()
        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)
        # 3. select {1-3}-grams not containing punctuation marks as candidates.
        extractor.candidate_selection(n=3,
                                      stoplist=list(string.punctuation))
        # 4. weight the candidates using a `tf` x `idf`
        df = pke.load_document_frequency_file(input_file='path/to/df.tsv.gz')
        extractor.candidate_weighting(df=df)
        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    @staticmethod
    def rake(documents: List[Document] = None, document: Document = None) -> Dict[str, List[str]]:
        # additional parameters:
        # ranking_metric
        # punctuations

        results = {}
        if document:
            language = document.language.lower()
            logging.info(f"{language} is used")
            r = Rake(ranking_metric=None,
                     stopwords=stopwords.words(language),
                     language=language,
                     min_length=2,
                     max_length=4)
            r.extract_keywords_from_text(document.text)
            document.keywords = r.get_ranked_phrases()
            results[document.doc_id] = document.keywords
        else:
            language = documents[0].language.lower()
            logging.info(f"{language} is used")
            r = Rake(ranking_metric=None,
                     stopwords=stopwords.words(language),
                     language=language,
                     min_length=2,
                     max_length=4)
            for document in tqdm(documents):
                r.extract_keywords_from_text(document.text)
                document.keywords = r.get_ranked_phrases()
                results[document.doc_id] = document.keywords

        return results

    def key_word_count(keywords: Dict[str, List[str]], top_k=100):
        flattened_keywords = [word for document, document_keywords in keywords.items() for word in document_keywords]
        c = Counter(flattened_keywords)
        if top_k is None:
            return c
        return c.most_common(top_k)


# load configuration parameters from config file
def main():
    config = ConfigLoader.get_config()

    # corpus = DataHandler.load_corpus("bundestag_corpus.json")
    # corpus = DataHandler.load_corpus("sustainability_corpus.json")
    corpus = DataHandler.load_corpus("abstract_corpus.json")

    # extract keywords
    # rake_keywords = KeyPhraseExtractor.rake(document=bundestag_corpus[0])
    # tfidf_keywords = KeyPhraseExtractor.tfidf_skl(documents=corpus)
    # print(tfidf_keywords)

    # print(KeyPhraseExtractor.key_word_count(KeyPhraseExtractor.rake(documents=corpus)))

    # aggregate documents / keywords

    # translate and match key words

    # visualize matching


if __name__ == '__main__':
    main()
