from enum import Enum
from collections import defaultdict, Counter
from typing import List, Set, Dict, NamedTuple
import pandas as pd
from tqdm import tqdm
import os
import re
import json
import logging
import googletrans


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
    def assign_keywords(documents: List["Document"], keywords):
        for document in tqdm(documents, desc="Assign keywords to documents"):
            document.keywords = keywords[document.doc_id]

    @staticmethod
    def year_wise_pseudo_documents(documents: List["Document"], language="English") -> List["Document"]:
        year_bins = defaultdict(list)

        for doc in documents:
            year_bins[doc.date].append(doc)

        pseudo_docs = [Document(doc_id=f'pseudo_{year}',
                                date=year,
                                text=" ".join([doc.text for doc in docs]),
                                language=language
                                )
                       for year, docs in tqdm(year_bins.items(), desc="Construct pseudo docs", total=len(year_bins))]

        return pseudo_docs

    @classmethod
    def group_keywords_year_wise(cls, documents_with_keywords: List["Document"], top_k_per_year=None) \
            -> Dict[int, List[str]]:

        year_bins = defaultdict(list)
        for doc in documents_with_keywords:
            year_bins[doc.date].extend(doc.keywords)

        grouped_keywords = defaultdict(list)
        for year, keyword_list in year_bins.items():
            keyword_counter = Counter(keyword_list)
            most_common_keywords = [keyword for keyword, freq in keyword_counter.most_common(top_k_per_year)]
            grouped_keywords[year] = most_common_keywords

        return grouped_keywords

    @staticmethod
    def transform_pseudo_docs_keywords_to_dict(keywords: Dict[str, List[str]]) -> Dict[int, List[str]]:
        return {int(id.replace("pseudo_", "")): keyword_list for id, keyword_list in keywords.items()}

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
    DE = "de"
    EN = "en"


class Keyword:

    def __init__(self, english_translation: str = None, german_translation: str = None, type: KeywordType = KeywordType.UNKNOWN):

        if not english_translation and not german_translation:
            raise Exception("Keyword cannot be intialized without any translation")

        self.english_translation = None
        self.german_translation = None
        self.source_language = KeywordSourceLanguage.UNKNOWN

        if english_translation:
            self.english_translation = english_translation
            self.source_language = KeywordSourceLanguage.EN

        if german_translation:
            self.german_translation = german_translation
            self.source_language = KeywordSourceLanguage.DE

        if english_translation and german_translation:
            self.source_language = KeywordSourceLanguage.UNKNOWN

        self.type = type


class KeywordTranslator:

    def __init__(self):
        self.translator = googletrans.Translator()

    def translate(self, keyword: Keyword):

        if keyword.source_language is KeywordSourceLanguage.UNKNOWN:
            logging.debug("KeywordTranslator: source is unknown, skipping")

        elif keyword.source_language is KeywordSourceLanguage.DE:
            self.translate_german2english(keyword)

        elif keyword.source_language is KeywordSourceLanguage.EN:
            self.translate_english2german(keyword)

    def translate_german2english(self, keyword):
        if not keyword.english_translation:
            logging.debug("KeywordTranslator: {} | source is DE, EN not set, translating ...".format(keyword.german_translation))

            translated = self.translator.translate(text=keyword.german_translation, src=str(KeywordSourceLanguage.DE.value), dest=str(KeywordSourceLanguage.EN.value))
            keyword.english_translation = translated.text

        else:
            logging.debug("KeywordTranslator: {}, {} | source is DE, but EN already set, skipping translation".format(keyword.german_translation, keyword.english_translation))

    def translate_english2german(self, keyword):
        if not keyword.german_translation:
            logging.debug("KeywordTranslator: {} | source is EN, DE not set, translating ...".format(keyword.english_translation))

            translated = self.translator.translate(text=keyword.english_translation, src=str(KeywordSourceLanguage.EN.value), dest=str(KeywordSourceLanguage.DE.value))
            keyword.german_translation = translated.text

        else:
            logging.debug("KeywordTranslator: {}, {}| source is EN, but DE already set, skipping translation".format(keyword.english_translation, keyword.german_translation))


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
    def save_corpus(corpus: List[Document], path: str):
        data = [doc.__dict__ for doc in corpus]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        logging.info(f'saved {path}')

    @staticmethod
    def load_corpus(path: str) -> List["Document"]:
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


if __name__ == '__main__':

    # key word translator example

    kwt = KeywordTranslator()

    def translate(keyword):
        print('before: {}\t| {}\t | {}'.format(keyword.english_translation, keyword.german_translation, keyword.source_language))
        kwt.translate(keyword)
        print('after:  {}\t| {}\t | {}\n'.format(keyword.english_translation, keyword.german_translation, keyword.source_language))

    translate(Keyword(english_translation="axis of evil"))

    translate(Keyword(german_translation="Achse des Bösen"))

    translate(Keyword(english_translation="axis of evil", german_translation="Achse des Bösen"))




