from enum import Enum
from collections import defaultdict, Counter
from typing import List, Set, Dict, NamedTuple, Tuple
import pandas as pd
from tqdm import tqdm
import os
import re
import json
import logging
import googletrans
import json


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

class KeywordType(str, Enum):

    UNKNOWN = "unknown"
    TFIDF_SKL = "tfidf_skl"
    TFIDF_PKE = "tfidf_pke"
    RAKE = "rake"
    TEXTRANK = "textrank"


class KeywordSourceLanguage(str, Enum):

    UNKNOWN = "unknown"
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

    def __eq__(self, other):
        if not isinstance(other, Keyword):
            return NotImplemented

        return self.german_translation == other.german_translation or self.english_translation == other.english_translation

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash((self.german_translation, self.english_translation))

    def __str__(self):
        if self.english_translation and self.german_translation:
            return f'({self.english_translation} | {self.german_translation})'
        if self.english_translation:
            return f'{self.english_translation}'
        if self.german_translation:
            return f'{self.german_translation})'

    __repr__ = __str__

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
    def assign_keywords(documents: List["Document"], keywords: Dict[int, List[str]] = None,
                        keyword_type: KeywordType = KeywordType.UNKNOWN,
                        translated_keywords: Dict[int, List[Keyword]] = None):

        for document in tqdm(documents, desc="Assign keywords to documents"):
            if keywords:
                if document.language == "German":
                    parsed_keywords = [Keyword(german_translation=keyword, type=keyword_type)
                                       for keyword in keywords[document.doc_id]]
                else:
                    parsed_keywords = [Keyword(english_translation=keyword, type=keyword_type)
                                       for keyword in keywords[document.doc_id]]
                document.keywords = parsed_keywords

            if translated_keywords:
                document.keywords = translated_keywords[document.doc_id]


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


class KeywordTranslator:

    def __init__(self, cache_file='translator_cache.json'):
        self.translator = googletrans.Translator()

        try:
            if not os.path.exists(cache_file):
                raise Exception("File does not exist")
            with open(cache_file) as json_file:
                self.cache = json.load(json_file)
        except Exception as e:
            logging.warning("Loading of file failed")
            logging.warning(e)
            self.cache = dict()

        self.cache_file = cache_file

    def __del__(self):
        self.save_cache()
        logging.info("saved cache file")

    def translate(self, keyword: Keyword):

        if keyword.source_language is KeywordSourceLanguage.UNKNOWN:
            logging.debug("KeywordTranslator: source is unknown, skipping")

        elif keyword.source_language is KeywordSourceLanguage.DE:
            self.translate_german2english(keyword)

        elif keyword.source_language is KeywordSourceLanguage.EN:
            self.translate_english2german(keyword)

    def translate_german2english(self, keyword):
        if keyword.german_translation in self.cache and self.cache[keyword.german_translation].english_translation:
            print('found keyword in cache, taking this one')
            keyword = self.cache[keyword.german_translation]
            return keyword
        else:
            if not keyword.english_translation:
                logging.debug("KeywordTranslator: {} | source is DE, EN not set, translating ...".format(keyword.german_translation))

                try:
                    translated = self.translator.translate(text=keyword.german_translation, src=str(KeywordSourceLanguage.DE.value), dest=str(KeywordSourceLanguage.EN.value))
                    keyword.english_translation = translated.text
                    self.add_to_cache(keyword)
                    return keyword
                except Exception as e:
                    logging.error("While trying to translate, an error occured:")
                    logging.error(e)

            else:
                logging.debug("KeywordTranslator: {}, {} | source is DE, but EN already set, skipping translation".format(keyword.german_translation, keyword.english_translation))
                return None

    def translate_english2german(self, keyword):

        if keyword.english_translation in self.cache and self.cache[keyword.english_translation].german_translation:
            print('found keyword in cache, taking this one')
            keyword = self.cache[keyword.english_translation]
            return keyword
        else:

            if not keyword.german_translation:
                logging.debug("KeywordTranslator: {} | source is EN, DE not set, translating ...".format(keyword.english_translation))
                try:
                    translated = self.translator.translate(text=keyword.english_translation, src=str(KeywordSourceLanguage.EN.value), dest=str(KeywordSourceLanguage.DE.value))
                    keyword.german_translation = translated.text
                    self.add_to_cache(keyword)

                    return keyword
                except Exception as e:
                    logging.error("While trying to translate, an error occured:")
                    logging.error(e)
            else:
                logging.debug("KeywordTranslator: {}, {}| source is EN, but DE already set, skipping translation".format(keyword.english_translation, keyword.german_translation))
                return None

    def add_to_cache(self, keyword: Keyword):
        self.cache[keyword.german_translation] = keyword
        self.cache[keyword.english_translation] = keyword

    def save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=1)


class KeyWordList():
    def __init__(self, keywords: NamedTuple, time_spec):
        pass


class KeywordMatcher:
    def __init__(self):
        pass

    @staticmethod
    def match_grouped_dicts(keywords_1: Dict[int, List[Keyword]],
                            keywords_2: Dict[int, List[Keyword]]) -> Dict[Keyword, Tuple[List[int], List[int]]]:
        reversed_keywords_1 = defaultdict[List]
        for year, keyword in keywords_1:
            reversed_keywords_1[keyword].append(year)

        reversed_keywords_2 = defaultdict[List]
        for year, keyword in keywords_2:
            reversed_keywords_2[keyword].append(year)

        # todo: implement real matching and check of both translation options
        for keyword in reversed_keywords_1:
            if keyword.german_translation in list(reversed_keywords_2):
                pass
        matched_keywords = ... #set(reversed_keywords_1).intersection(set(reversed_keywords_2))

        return {keyword: (reversed_keywords_1[keyword], reversed_keywords_2) for keyword in matched_keywords}

    @staticmethod
    def match_corpora(corpus_1: List[Document],
                      corpus_2: List[Document]) -> Dict[Keyword, Tuple[List[int], List[str]]]:
        reversed_keywords_1 = defaultdict[List]
        for document in corpus_1:
            for keyword in document.keywords:
                reversed_keywords_1[keyword].append(document.doc_id)

        reversed_keywords_2 = defaultdict[List]
        for document in corpus_2:
            for keyword in document.keywords:
                reversed_keywords_2[keyword].append(document.doc_id)

        # todo: implement real matching and check of both translation options
        matched_keywords = ... #set(reversed_keywords_1).intersection(set(reversed_keywords_2))

        return {keyword: (reversed_keywords_1[keyword], reversed_keywords_2) for keyword in matched_keywords}


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
    monkey = Keyword(german_translation="Affe", english_translation="Monkey")
    affe = Keyword(german_translation="Affe", english_translation="Ape")
    affe2 = Keyword(german_translation="Affe", english_translation="Ape")
    weird_monkey = Keyword(english_translation="Affe", german_translation="Monkey")
    patch = Keyword(german_translation="Patch", english_translation="patch")
    print(monkey == affe)
    print(affe == affe2)
    print(monkey == weird_monkey)
    print(affe == weird_monkey)
    print(monkey == patch)

    print(monkey in list({affe, affe2}))

    # key word translator example
    kwt = KeywordTranslator()

    def translate(keyword):
        print('before: {}\t| {}\t | {}'.format(keyword.english_translation, keyword.german_translation, keyword.source_language))
        kwt.translate(keyword)
        print('after:  {}\t| {}\t | {}\n'.format(keyword.english_translation, keyword.german_translation, keyword.source_language))

    translate(Keyword(english_translation="axis of evil"))

    translate(Keyword(german_translation="Achse des Bösen"))

    translate(Keyword(english_translation="axis of evil", german_translation="Achse des Bösen"))




