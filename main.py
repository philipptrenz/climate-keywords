from typing import List, Set, Dict, NamedTuple
from abc import ABC, abstractmethod
import pandas as pd
from rake_nltk import Rake
from tqdm import tqdm
# import pke
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_data(path="E:/mcc/all_docs.csv"):
    df = pd.read_csv(path)
    return [Document(text=row["content"], date=row["PY"], language="English", doc_id=i, tags=row["tags"]) for i, row in df.iterrows() if not pd.isna(row["content"])]


class Document:
    def __init__(self, text, date, language, doc_id: str, tags=None):
        self.text = text
        self.date = date
        self.language = language
        self.doc_id = doc_id
        self.tags = tags
        self.keywords = None

    @classmethod
    def from_sources_to_documents(cls, srcs):
        return [Document(src.text, src.date, src.language, src.doc_id) for src in srcs]

    @staticmethod
    def time_binning(documents: List["Document"], binning) -> Dict[float, List["Document"]]:
        pass

    def __str__(self):
        return f'{self.date}: {self.text[:30]}'


class Keyword():
    def __init__(self, english_translation, german_translation):
        self.english_translation = english_translation
        self.german_translation = german_translation

    def set_translation(self, translation_type, translation):
        if translation_type == "german":
            self.german_translation = translation
        else:
            self.english_translation = translation

class KeyWordList():
    def __init__(self, keywords: NamedTuple, time_spec):
        pass


def extract_tf_keywords(documents: List[Document]) -> Dict[str, List[str]]:
    pass


def extract_tfidf_keywords_skl(documents: List[Document]) -> Dict[str, List[str]]:
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


def extract_tfidf_keywords_pke(documents: List[Document]) -> Dict[str, List[str]]:
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

def extract_RAKE_keywords(documents: List[Document]) -> Dict[str, List[str]]:
    r = Rake()
    results = {}
    for document in tqdm(documents):
        r.extract_keywords_from_text(document.text)
        document.keywords = r.get_ranked_phrases()
        results[document.doc_id] = document.keywords

    return results



# read and parse data
abstract_corpus = get_data(path="E:/mcc/all_docs.csv")
print(extract_tfidf_keywords_skl(abstract_corpus[:1000]))
# print(extract_RAKE_keywords(abstract_corpus))

accademic_srcs = [] # add real data source
politic_srcs = [] # add real data source
documents = Document.from_sources_to_documents(accademic_srcs)
documents.extend(Document.from_sources_to_documents(politic_srcs))


# extract keywords
keywords = extract_tf_keywords(documents)


# aggregate documents / keywords


# translate and match key words

# visualize matching



