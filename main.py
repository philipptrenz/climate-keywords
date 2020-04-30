from typing import List, Set, Dict, NamedTuple
from abc import ABC, abstractmethod


class Document:
    def __init__(self, text, date, language, doc_id):
        self.text = text
        self.date = date
        self.language = language
        self.doc_id = doc_id

        self.keywords = None

    @classmethod
    def from_sources_to_documents(cls, srcs):
        return [Document(src.text, src.date, src.language, src.doc_id) for src in srcs]

    @staticmethod
    def time_binning(documents: List["Document"], binning) -> Dict[float, List["Document"]]:
        for d in documents:
        pass

class KeyWordList():
    def __init__(self, keywords: NamedTuple, time_spec, ):
        pass

def extract_tf_keywords(time_specific_documents: Dict[float, List[Document]]) -> Dict[float, List[str]]:
    pass


def extract_tfidf_keywords(time_specific_documents: Dict[float, List[Document]]) -> Dict[float, List[str]]:
    pass

# read and parse data
accademic_srcs = [] # add real data source
politic_srcs = [] # add real data source
documents = Document.from_sources_to_documents(accademic_srcs)
documents.extend(Document.from_sources_to_documents(politic_srcs))
time_binned_documents = Document.time_binning(documents, "year")

# extract key words
key_words = extract_tf_keywords(time_binned_documents)

# match key words

# visualize matching



