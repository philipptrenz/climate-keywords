from enum import Enum
from collections import defaultdict, Counter
from typing import List, Set, Dict, NamedTuple
import pandas as pd
from tqdm import tqdm
import os
import re
import json
import logging


class ConfigLoader:
    @staticmethod
    def get_config():
        if os.path.exists("config.json"):
            logging.info('importing config from config.json ...')
            with open("config.json") as json_file:
                return json.load(json_file)

        elif os.path.exists("default.config.json"):
            logging.info('importing config from default.config.json ...')
            with open("default.config.json") as json_file:
                return json.load(json_file)
        else:
            raise Exception("config file missing!")


class Document:
    def __init__(self, text: str, date, language: str, doc_id: str, tags=None, author=None, party=None, rating=None,
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


class KeywordType(Enum):

    UNKNOWN = 0
    TFIDF_SKL = "tfidf_skl"
    TFIDF_PKE = "tfidf_pke"
    RAKE = "rake"
    TEXTRANK = "textrank"


class KeywordSourceLanguage(Enum):

    UNKNOWN = 0
    DE = "german"
    EN = "english"


class Keyword:

    def __init__(self, english_translation: str = None, german_translation: str = None, type: KeywordType = KeywordType.UNKNOWN):

        if not english_translation and not german_translation:
            raise Exception("Keyword cannot be intialized without any translation")

        self.source_language = KeywordSourceLanguage.UNKNOWN

        if english_translation:
            self.english_translation = english_translation
            if not german_translation:
                self.german_translation = self.translate_english2german(english_translation)
                self.source_language = KeywordSourceLanguage.EN

        if german_translation:
            self.german_translation = german_translation
            if not english_translation:
                self.english_translation = self.translate_german2english(german_translation)
                self.source_language = KeywordSourceLanguage.DE

        self.type = type

    def set_translation(self, translation_type, translation):
        if translation_type == "german":
            self.german_translation = translation
        else:
            self.english_translation = translation

    @staticmethod
    def translate_english2german(self, english):
        logging.warning("Auto translation of keywords not yet implemented!")
        return None

    @staticmethod
    def translate_german2english(self, german):
        logging.warning("Auto translation of keywords not yet implemented!")
        return None


class KeyWordList():
    def __init__(self, keywords: NamedTuple, time_spec):
        pass


class DataHandler:
    @staticmethod
    def get_sustainability_data(path):
        def nan_resolver(pandas_attribut):
            return None if pd.isna(pandas_attribut) else pandas_attribut

        df = pd.read_csv(path)
        print(df.columns)
        return [Document(text=row["content"],
                         date=nan_resolver(row["PY"]),
                         language="English",
                         doc_id=f'su_{i}',
                         author=row["authors"],
                         tags=nan_resolver(row["tags"])) for i, row in
                tqdm(df.iterrows(), desc="Load sustainability abstracts", total=len(df)) if not pd.isna(row["content"])]

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

        for i, record in tqdm(enumerate(records), desc="Load abstracts", total=len(records)):
            attributes = re.findall(wos_categories, record)

            wos_dict = defaultdict(lambda: None,
                                   {attribute[0]: ' '.join(list(attribute[1:])).strip() for attribute in attributes})
            text = wos_dict["AB"]
            if text:
                abstracts.append(Document(text=text,
                                          date=wos_dict["PY"],
                                          language="English",
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
        for file in tqdm(files, desc="load Bundestag speeches", total=len(files)):
            df = pd.read_csv(os.path.join(dir, file))
            speeches.extend([Document(text=row["Speech text"],
                                      date=row["Date"][:4],
                                      language="German",
                                      doc_id=f'po_{i}',
                                      author=row["Speaker"],
                                      party=row["Speaker party"],
                                      rating=row["Interjection count"])
                             for i, row in df.iterrows() if not pd.isna(row["Speech text"])])

        return speeches

    @staticmethod
    def save_corpus(corpus: List[Document], path: str = "data.json"):
        data = [doc.__dict__ for doc in corpus]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        logging.info(f'saved {path}')

    @staticmethod
    def load_corpus(path: str = "data.json") -> List["Document"]:
        logging.info(f"load {path}")
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
        logging.info(f"{path} loaded")

        return corpus
