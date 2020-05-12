from collections import defaultdict
from typing import List, Set, Dict, NamedTuple
from abc import ABC, abstractmethod
import pandas as pd
from rake_nltk import Rake
from tqdm import tqdm
import pke
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import re
import json


class ConfigLoader:
    @staticmethod
    def get_config():
        if os.path.exists("config.json"):
            print('importing config from config.json ...')
            with open("config.json") as json_file:
                return json.load(json_file)

        elif os.path.exists("default.config.json"):
            print('importing config from default.config.json ...')
            with open("default.config.json") as json_file:
                return json.load(json_file)
        else:
            raise Exception("config file missing!")


class DataLoader:
    @staticmethod
    def get_sustainability_data(path):
        def nan_resolver(pandas_attribut):
            return None if pd.isna(pandas_attribut) else pandas_attribut

        df = pd.read_csv(path)
        return [Document(text=row["content"], date=nan_resolver(row["PY"]), language="English", doc_id=f'{i}',
                         tags=nan_resolver(row["tags"])) for i, row in
                df.iterrows() if not pd.isna(row["content"])]

    @staticmethod
    def get_abstracts(path):
        # https://images.webofknowledge.com/images/help/WOS/hs_wos_fieldtags.html
        # control sequences:
        # ER    end of record
        # EF    end of file
        # FN    file name
        # relevant:
        # PY    Year Published
        # DE    Author Keywords
        # AB    Abstract
        # unsure:
        # TI    Document Title
        # AU    Authors
        # AF    Authors Full Name
        with open(path, encoding="utf-8") as f:
            data = f.read()

        records = data.split("ER ")
        print(len(records))
        abstracts = []
        wos_categories = re.compile(
            '(FN|VR|PT|AU|AF|BA|BF|CA|GP|BE|TI|SO|SE|BS|LA|DT|CT|CY|CL|SP|HO|DE|ID|AB|C1|RP|EM|RI'
            '|OI|FU|FX|CR|NR|TC|Z9|U1|U2|PU|PI|PA|SN|EI|BN|J9|JI|PD|PY|VL|IS|SI|PN|SU|MA|BP|EP|AR'
            '|DI|D2|EA|EY|PG|P2|WC|SC|GA|PM|UT|OA|HP|HC|DA|ER|EF)\\s(.*)', re.MULTILINE)

        for i, record in tqdm(enumerate(records)):
            attributes = re.findall(wos_categories, record)

            wos_dict = defaultdict(lambda: None,
                                   {attribute[0]: ' '.join(list(attribute[1:])).strip() for attribute in attributes})

            if wos_dict["AB"]:
                abstracts.append(Document(text=wos_dict["AB"],
                                          date=wos_dict["PY"],
                                          language=wos_dict["LA"],
                                          doc_id=f'sc_{i}',
                                          author=wos_dict["AU"]))

        return abstracts

    @staticmethod
    def get_bundestag_speeches(dir):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(dir):
            files.extend(filenames)
            break

        for f in files:
            if f.startswith('.'):
                files.remove(f)

        speeches = []
        for file in tqdm(files):
            df = pd.read_csv(os.path.join(dir, file))
            speeches.extend([Document(text=row["Speech text"],
                                      date=row["Date"],
                                      language="German",
                                      doc_id=f'po_{i}',
                                      author=row["Speaker"],
                                      party="Speaker party",
                                      rating="Interjection count")
                             for i, row in df.iterrows() if not pd.isna(row["Speech text"])])

        return speeches


class Document:
    def __init__(self, text, date, language, doc_id: str, tags=None, author=None, party=None, rating=None,
                 keywords=None):
        self.text = text
        self.date = date
        self.language = language
        self.doc_id = doc_id
        self.tags = tags
        self.author = author
        self.party = party
        self.rating = rating
        self.keywords = keywords

    @staticmethod
    def time_binning(documents: List["Document"], binning) -> Dict[float, List["Document"]]:
        pass

    def __str__(self):
        return f'{self.date}: {self.text[:30]}'

    __repr__ = __str__

    @staticmethod
    def save_corpus(corpus: List["Document"], path: str = "data.json"):
        data = [doc.__dict__ for doc in corpus]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    @staticmethod
    def load_corpus(path: str = "data.json") -> List["Document"]:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())

        corpus = [Document(text=doc["text"],
                           date=doc["date"],
                           language=doc["language"],
                           doc_id=doc["doc_id"],
                           author=doc["author"],
                           tags=doc["tags"],
                           keywords=doc["keywords"],
                           party=doc["party"],
                           rating=doc["rating"]) for doc in data]

        return corpus


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
        r = Rake()
        results = {}
        if document:
            r.extract_keywords_from_text(document.text)
            document.keywords = r.get_ranked_phrases()
            results[document.doc_id] = document.keywords
        else:
            for document in tqdm(documents):
                r.extract_keywords_from_text(document.text)
                document.keywords = r.get_ranked_phrases()
                results[document.doc_id] = document.keywords

        return results


# load configuration parameters from config file
def main():
    config = ConfigLoader.get_config()

    # read and parse data
    # sustainability_corpus = DataLoader.get_sustainability_data(path=config["datasets"]["abstracts"]["sustainability"])
    # abstract_corpus = DataLoader.get_abstracts(path=config["datasets"]["abstracts"]["climate_literature"])
    bundestag_corpus = DataLoader.get_bundestag_speeches(dir=config["datasets"]["bundestag"]["directory"])
    # print(extract_tfidf_keywords_skl(abstract_corpus[:1000]))

    Document.save_corpus(bundestag_corpus)
    Document.load_corpus()

    # extract keywords
    rake_keywords = KeyPhraseExtractor.rake(document=bundestag_corpus[0])
    tfidf_keywords = KeyPhraseExtractor.tfidf_skl(documents=bundestag_corpus)
    print(rake_keywords)

    # aggregate documents / keywords

    # translate and match key words

    # visualize matching


if __name__ == '__main__':
    main()
