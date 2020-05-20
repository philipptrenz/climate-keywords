from collections import defaultdict, Counter, OrderedDict
from typing import List, Set, Dict, NamedTuple, Callable
import pandas as pd
from rake_nltk import Rake
from nltk.corpus import stopwords
from tqdm import tqdm
import pke
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging

from utils import Document, ConfigLoader, DataHandler, Keyword, KeywordTranslator, KeyWordList, KeywordType


class KeyPhraseExtractor:
    min_nrgam = 1
    max_ngram = 4

    @classmethod
    def term_frequency(cls, documents: List[Document]) -> Dict[str, List[str]]:
        pass

    @classmethod
    def tfidf_skl(cls, documents: List[Document]) -> Dict[str, List[str]]:
        def top_tfidf_feats(row, features, top_n=25):
            ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
            topn_ids = np.argsort(row)[::-1][:top_n]
            top_feats = [(features[i], row[i]) for i in topn_ids]
            df = pd.DataFrame(top_feats)
            df.columns = ['feature', 'tfidf']
            print(df)
            return df

        language = documents[0].language.lower()
        tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words(language),
                                           ngram_range=(cls.min_nrgam, cls.max_ngram),
                                           min_df=2)
        tfidf_matrix = tfidf_vectorizer.fit_transform([document.text for document in documents])
        doc_id_lookup = {i: document.doc_id for i, document in enumerate(documents)}

        k = 10
        results = {}
        features = tfidf_vectorizer.get_feature_names()

        for i, doc in tqdm(enumerate(tfidf_matrix), desc="Calculating tf-idf", total=tfidf_matrix.shape[0]):
            df = pd.DataFrame(doc.T.todense(), index=features,
                              columns=["tfidf"])
            top_key_words = df.sort_values(by=["tfidf"], ascending=False)[:k]
            results[doc_id_lookup[i]] = list(top_key_words.index)

        return results

    @classmethod
    def tfidf_pke(cls, documents: List[Document]) -> Dict[str, List[str]]:
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

    @classmethod
    def rake(cls, documents: List[Document] = None, document: Document = None) -> Dict[str, List[str]]:
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
                     min_length=cls.min_nrgam,
                     max_length=cls.max_ngram)
            r.extract_keywords_from_text(document.text)
            document.keywords = r.get_ranked_phrases()
            results[document.doc_id] = document.keywords
        else:
            language = documents[0].language.lower()
            logging.info(f"{language} is used")
            r = Rake(ranking_metric=None,
                     stopwords=stopwords.words(language),
                     language=language,
                     min_length=cls.min_nrgam,
                     max_length=cls.max_ngram)
            for document in tqdm(documents, desc="Calculating RAKE"):
                r.extract_keywords_from_text(document.text)
                document.keywords = r.get_ranked_phrases()
                results[document.doc_id] = document.keywords

        return results

    @classmethod
    def key_word_count(cls, keywords: Dict[str, List[str]], top_k=100):
        flattened_keywords = [word for document, document_keywords in keywords.items() for word in document_keywords]
        c = Counter(flattened_keywords)
        if top_k is None:
            return c
        return c.most_common(top_k)

    @staticmethod
    def get_top_k_keywords(year_wise_keywords, top_k=100):
        return {year: keyword_list[:top_k] for year, keyword_list in year_wise_keywords.items()}




def main():
    # load configuration parameters from config file
    config = ConfigLoader.get_config()

    # corpus = DataHandler.load_corpus(config["corpora"]["abstract_corpus"])
    # corpus = DataHandler.load_corpus(config["corpora"]["bundestag_corpus"])
    corpus = DataHandler.load_corpus(config["corpora"]["sustainability_corpus"])
    corpus = corpus[:100]
    # build yearwise pseudo documents
    # corpus = corpus[:100]
    pseudo_docs = Document.year_wise_pseudo_documents(corpus)

    # extract keywords
    # tfidf_keywords = KeyPhraseExtractor.tfidf_skl(documents=pseudo_docs)
    # print(tfidf_keywords)
    rake_keywords = KeyPhraseExtractor.rake(documents=corpus)
    Document.assign_keywords(corpus, rake_keywords, keyword_type=KeywordType.RAKE)
    key_words_post_group = Document.group_keywords_year_wise(corpus)
    key_words_pre_group = Document.transform_pseudo_docs_keywords_to_dict(KeyPhraseExtractor.rake(documents=pseudo_docs))

    print(KeyPhraseExtractor.get_top_k_keywords(key_words_post_group, 10))
    print(KeyPhraseExtractor.get_top_k_keywords(key_words_pre_group, 10))
    # format: {year->list fo keywords}


    print('extracting keywords with rake ...')
    rake_keywords = KeyPhraseExtractor.rake(document=corpus[0])
    rake_keywords_keys = list(rake_keywords.keys())
    print('rake keywords dict keys:', rake_keywords_keys)

    kwt = KeywordTranslator()
    list_of_keywords = []

    for k in rake_keywords[rake_keywords_keys[0]]:
        kw = Keyword(german_translation=k, type=KeywordType.RAKE)
        kwt.translate(kw)
        list_of_keywords.append(kw)
        print('{} \t {} \t\t\t {}'.format(kw.source_language, kw.english_translation, kw.german_translation))

    # print('extracting keywords with tf-idf ...')
    # tfidf_keywords = KeyPhraseExtractor.tfidf_skl(documents=corpus)
    # print(tfidf_keywords, '\n')

    # print(KeyPhraseExtractor.key_word_count(KeyPhraseExtractor.rake(documents=corpus)))

    # aggregate documents / keywords

    # translate and match key words

    # visualize matching


if __name__ == '__main__':
    main()
