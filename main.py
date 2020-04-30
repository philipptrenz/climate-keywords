from typing import List, Set, Dict, NamedTuple
from abc import ABC, abstractmethod


class Document:
    def __init__(self, text, date, language, doc_id: str):
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
        pass


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


def extract_tfidf_keywords(documents: List[Document]) -> Dict[str, List[str]]:
    pass

# read and parse data
accademic_srcs = [] # add real data source
politic_srcs = [] # add real data source
documents = Document.from_sources_to_documents(accademic_srcs)
documents.extend(Document.from_sources_to_documents(politic_srcs))


# extract keywords
keywords = extract_tf_keywords(documents)

# aggregate documents / keywords


# translate and match key words

# visualize matching



