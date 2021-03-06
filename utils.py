import json
import logging
import math
import os
import random
import re
import time
from collections import defaultdict, Counter
from enum import Enum
from pathlib import Path
from typing import List, Set, Dict, Tuple, Union
import six

import pandas as pd
import spacy
from tqdm import tqdm

import googletrans

from google.cloud import translate_v2 as g_translate
from google_auth_oauthlib import flow
import bs4
import xml.etree.ElementTree as et


class ConfigLoader:
    @staticmethod
    def get_config(relative_path=""):

        path = os.path.join(relative_path, "config.json")
        if os.path.exists(path):
            logging.info('importing config from config.json ...')
            with open(path) as json_file:
                return json.load(json_file)

        path = os.path.join(relative_path, "default.config.json")
        if os.path.exists(path):
            path = os.path.join(relative_path, "default.config.json")
            logging.info('importing config from default.config.json ...')
            with open(path) as json_file:
                return json.load(json_file)

        raise Exception("config file missing!")


class KeywordType(str, Enum):
    UNKNOWN = "unknown"
    TF_SKL = "tf_skl"
    TFIDF_SKL = "tfidf_skl"
    TFIDF_PKE = "tfidf_pke"
    RAKE = "rake"
    TEXT_RANK = "text_rank"
    TEXT_RANK_PKE = "text_rank_pke"
    SINGLE_RANK_PKE = "single_rank_pke"
    YAKE_PKE = "yake_pke"
    TOPIC_RANK_PKE = "topic_rank_pke"
    TOPICAL_PAGE_RANK_PKE = "topical_page_rank_pke"
    POSITION_RANK_PKE = "position_rank_pke"
    MULTIPARTITE_RANK_PKE = "multipartite_rank_pke"


class Language(str, Enum):
    UNKNOWN = "unknown"
    DE = "de"
    EN = "en"

    @staticmethod
    def get_from_str(language: str) -> "Language":
        if language.lower() == "en" or language.lower() == "english" or \
                language.lower() == "englisch":
            return Language.EN
        if language.lower() == "de" or language.lower() == "deutsch" or \
                language.lower() == "ger" or language.lower() == "german":
            return Language.DE
        return Language.UNKNOWN


class Keyword:
    def __init__(self, english_translation: str = None, german_translation: str = None,
                 keyword_type: KeywordType = KeywordType.UNKNOWN, source_language: Language = Language.UNKNOWN):

        if not english_translation and not german_translation:
            raise Exception("Keyword cannot be intialized without any translation")

        self.english_translation = None
        self.german_translation = None

        if english_translation:
            self.english_translation = english_translation
            self.source_language = Language.EN

        if german_translation:
            self.german_translation = german_translation
            self.source_language = Language.DE

        if english_translation and german_translation:
            self.source_language = Language.UNKNOWN

        if source_language is not Language.UNKNOWN:
            self.source_language = source_language

        self.keyword_type = keyword_type

    def __getitem__(self, item):
        return self.__dict__[item]

    def __eq__(self, other):
        if not isinstance(other, Keyword):
            return NotImplemented
        r = self.german_translation == other.german_translation or self.english_translation == other.english_translation
        return r

    def lemmatize(self, german_nlp, english_nlp):
        german_lemmatized = []
        if self.german_translation is None and self.english_translation is None:
            print(self.english_translation, self.german_translation)
        if self.german_translation:
            for token in german_nlp(self.german_translation):
                german_lemmatized.append(token.lemma_)

        english_lemmatized = []
        if self.english_translation:
            for token in english_nlp(self.english_translation):
                english_lemmatized.append(token.lemma_)

        self.german_translation = " ".join(german_lemmatized)
        self.english_translation = " ".join(english_lemmatized)

    # def __hash__(self):
    #     # necessary for instances to behave sanely in dicts and sets.
    #     return hash((self.german_translation, self.english_translation))

    def __str__(self):
        if self.english_translation and self.german_translation:
            return f'{self.source_language}({self.english_translation} | {self.german_translation})'
        if self.english_translation:
            return f'{self.source_language}({self.english_translation} | -)'
        if self.german_translation:
            return f'{self.source_language}(- | {self.german_translation})'

    __repr__ = __str__

    @classmethod
    def from_json(cls, data: dict):
        return Keyword(
            english_translation=data["english_translation"],
            german_translation=data["german_translation"],
            keyword_type=KeywordType(data["keyword_type"]),
            source_language=Language(data["source_language"])
        )

    @staticmethod
    def parse_keywords(keywords_dict_list: List[Dict[str, str]]) -> Union[List["Keyword"], None]:
        if keywords_dict_list:
            parsed_keywords = [Keyword(german_translation=keyword_dict['german_translation'],
                                       english_translation=keyword_dict['english_translation'],
                                       keyword_type=KeywordType[keyword_dict['keyword_type'].upper()],
                                       source_language=Language[keyword_dict['source_language'].upper()])
                               for keyword_dict in keywords_dict_list]
            return parsed_keywords
        return None

    @staticmethod
    def parse_keywords_from_str_list(key_word_list: List[str], language: Language, keyword_type: KeywordType) \
            -> Union[List["Keyword"], None]:
        if key_word_list:
            if language == Language.DE:
                parsed_keywords = [Keyword(german_translation=keyword, keyword_type=keyword_type,
                                           source_language=language)
                                   for keyword in key_word_list]
            elif language == Language.EN:

                parsed_keywords = [Keyword(english_translation=keyword, keyword_type=keyword_type,
                                           source_language=language)
                                   for keyword in key_word_list]

                return parsed_keywords
            else:
                # parsed_keywords = [Keyword(english_translation=keyword, keyword_type=keyword_type,
                #                            source_language=Language.UNKNOWN)
                #                    for keyword in keywords[document.doc_id]]
                raise UserWarning(f"Document Language {language} unknown!")
            return parsed_keywords
        return None


class Document:
    def __init__(self, text: str, date, language: Language, doc_id: str, tags=None, author=None, party=None,
                 rating=None,
                 keywords: List[Keyword] = None):
        self.text = text
        self.date = date
        self.language = language
        self.doc_id = doc_id
        self.tags = tags
        self.author = author
        self.party = party
        self.rating = rating
        self.keywords = keywords

    def __str__(self):
        return f'{self.date}: {self.text[:30]}'

    __repr__ = __str__


class Corpus:
    def __init__(self, source: Union[Dict[Union[str, int], Document], List[Document], str],
                 name: str,
                 language: Language,
                 has_assigned_keywords=False,
                 has_translated_keywords=False):
        self.name = name
        self.language = language
        self.has_assigned_keywords = has_assigned_keywords
        self.has_translated_keywords = has_translated_keywords

        if isinstance(source, str):
            documents = self.load_corpus(path=source)
        else:
            documents = source
        if isinstance(documents, dict):
            self.documents = documents
        elif isinstance(documents, list):
            self.documents = {document.doc_id: document for document in documents}
        else:
            raise NotImplementedError("Not supported Document collection!")

    def get_documents(self, as_list=True) -> Union[List[Document], Dict[Union[str, int], Document]]:
        if as_list:
            return list(self.documents.values())
        else:
            return self.documents

    def assign_keywords(self, keywords: Dict[Union[int, str], List[Union[str, Keyword]]] = None,
                        keyword_type: KeywordType = KeywordType.UNKNOWN,
                        translated_keywords: Dict[int, List[Keyword]] = None,
                        keyword_type_is_str=True):
        for document in tqdm(self.get_documents(), desc="Assign keywords to documents"):
            if keywords or keywords is not None:
                if document.doc_id in keywords and keywords[document.doc_id]:
                    keyword_list = keywords[document.doc_id]
                    if keyword_type_is_str:
                        document.keywords = Keyword.parse_keywords_from_str_list(keyword_list,
                                                                                 document.language,
                                                                                 keyword_type)
                    else:
                        document.keywords = keyword_list
            else:
                document.keywords = []

            if translated_keywords:
                document.keywords = translated_keywords[document.doc_id]

        self.has_assigned_keywords = True

    def assign_corpus_with_keywords_from_file(self, keyword_path: str):
        with open(keyword_path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
            doc_id2keywords = {doc_id: Keyword.parse_keywords(keywords) for doc_id, keywords in data.items()}
            self.assign_keywords(doc_id2keywords, keyword_type_is_str=False)
            no_keywords_in = []
            for document in self.get_documents():
                if document.keywords is None or len(document.keywords) == 0:
                    no_keywords_in.append(document.doc_id)

        return doc_id2keywords

    def year_wise_pseudo_documents(self) -> "Corpus":
        year_bins = defaultdict(list)

        for doc in self.get_documents():
            year_bins[doc.date].append(doc)
        pattern = re.compile('[^a-zA-ZöäüÖÄÜß\s]+')

        pseudo_docs = [Document(doc_id=f'{self.name}_pseudo_{year}',
                                date=year,
                                text=" ".join([pattern.sub('', doc.text) for doc in docs]),
                                language=self.language
                                )
                       for year, docs in tqdm(year_bins.items(), desc="Construct pseudo docs", total=len(year_bins))]

        return Corpus(source=pseudo_docs,
                      name=self.name,
                      language=self.language,
                      has_assigned_keywords=self.has_assigned_keywords,
                      has_translated_keywords=self.has_translated_keywords)

    def group_keywords_year_wise(self, top_k_per_year=None) -> Dict[int, List[str]]:
        if self.has_assigned_keywords:
            year_bins = defaultdict(list)
            for doc in self.documents:
                year_bins[doc.date].extend(doc.keywords)

            grouped_keywords = defaultdict(list)
            for year, keyword_list in year_bins.items():
                keyword_counter = Counter(keyword_list)
                most_common_keywords = [keyword for keyword, freq in keyword_counter.most_common(top_k_per_year)]
                grouped_keywords[year] = most_common_keywords

            return grouped_keywords
        else:
            raise UserWarning("No keywords assigned for grouping!")

    def get_n_documents_as_corpus(self, n: int) -> "Corpus":
        documents = self.get_documents(as_list=True)
        documents = documents[:n]
        return Corpus(source=documents,
                      language=self.language,
                      has_assigned_keywords=self.has_assigned_keywords,
                      has_translated_keywords=self.has_translated_keywords,
                      name=f'{self.name}_top{n}')

    def save_corpus(self, path: str):
        data = [doc.__dict__ for doc in self.get_documents()]
        # data = {doc.doc_id: doc.__dict__ for doc in self.get_documents()}

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1, default=lambda o: o.__dict__)
        logging.info(f'saved {path}')

    def save_corpus_without_text(self, path: str):
        data = {}
        for doc in self.get_documents():
            doc.text = ""
            data[doc.doc_id] = doc.__dict__

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1, default=lambda o: o.__dict__)
        logging.info(f'saved {path}')

    def get_years(self) -> [str]:
        years = set()
        for d in self.get_documents(as_list=True):
            if d.date:
                years.add(d.date)
        return sorted(list(years))

    @staticmethod
    def load_corpus(path: str) -> List[Document]:
        logging.info(f"load {path}")
        with open(path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())

        corpus = [Document(text=doc["text"],
                           date=doc["date"],
                           language=Language.get_from_str(doc["language"]),
                           doc_id=doc["doc_id"],
                           author=doc["author"],
                           tags=doc["tags"],
                           keywords=Keyword.parse_keywords(doc["keywords"]),
                           party=doc["party"],
                           rating=doc["rating"]) for doc in data]
        logging.info(f"{path} loaded")

        return corpus

    def token_number(self):
        c = 0
        for d in self.get_documents():
            c += len(d.text.split())
        return c

    def year_wise(self, ids: bool = False) -> Dict[int, List[Union[str, int, Document]]]:
        year_bins = defaultdict(list)

        for doc in self.get_documents():
            if ids:
                year_bins[doc.date].append(doc.doc_id)
            else:
                year_bins[doc.date].append(doc)

        return year_bins

    def sample(self, number_documents=100, as_corpus=True, seed=None):
        if len(self) < number_documents:
            return self

        if seed:
            random.seed(seed)

        if as_corpus:
            result = Corpus(source=random.sample(self.get_documents(), k=number_documents),
                            language=self.language,
                            has_assigned_keywords=self.has_assigned_keywords,
                            has_translated_keywords=self.has_translated_keywords,
                            name=f'{self.name}_sample')
        else:
            result = random.sample(self.get_documents(), k=number_documents)
        return result

    def translate_keywords(self, keyword_translator: "KeywordTranslator", restrict_per_document=None) -> List[Keyword]:
        # if keyword_translator is None:
        #     keyword_translator = KeywordTranslator(cache_file=config["translator"]["cache_file"])
        list_of_keywords = []

        for document in self.get_documents():
            for i, kw in enumerate(document.keywords):
                if restrict_per_document and i >= restrict_per_document:
                    break
                keyword_translator.translate(kw)
                list_of_keywords.append(kw)
                print('{} \t {} \t\t\t {}'.format(kw.source_language, kw.english_translation, kw.german_translation))

        keyword_translator.save_cache()
        self.has_translated_keywords = True
        return list_of_keywords

    @staticmethod
    def transform_pseudo_docs_keywords_to_dict(keywords: Dict[str, List[str]]) -> Dict[int, List[str]]:
        return {int(i.replace("pseudo_", "")): keyword_list for i, keyword_list in keywords.items()}

    def __iter__(self):
        return self.documents.values().__iter__()

    def __len__(self):
        return len(self.documents)

    def __str__(self):
        return f'docs={len(self)}, lan={self.language}, keywords={self.has_assigned_keywords}, name={self.name}'

    __repr__ = __str__


class CorpusFilter:

    @staticmethod
    def filter(corpus: Corpus,
               text_contains_one_of: [str] = None,
               date_in_range: range = None,
               is_one_of_languages: [str] = None,
               is_one_of_doc_ids: [int] = None,
               has_authors: [str] = None,
               has_tags: [str] = None,
               is_one_of_parties: [str] = None,
               ratings_in_range: range = None,
               has_one_of_keywords: [str] = None,
               has_one_of_keywords_with_english_translation: [str] = None,
               has_one_of_keywords_with_german_translation: [str] = None) -> Corpus:

        filtered: [Document] = []

        for d in tqdm(corpus.get_documents(as_list=True), desc=f"Filtering \'{corpus.name}\' corpus "):
            # noinspection PyBroadException
            try:
                if text_contains_one_of:
                    does_contain = False
                    for t in text_contains_one_of:
                        if re.search(t, d.text, re.I):
                            does_contain = True
                            break
                    if not does_contain:
                        continue

                if date_in_range:
                    if not d.date or d.date not in date_in_range:
                        continue

                if is_one_of_languages:
                    if not d.language or d.language not in is_one_of_languages:
                        continue

                if is_one_of_doc_ids:
                    if not d.doc_id or d.doc_id not in is_one_of_doc_ids:
                        continue

                if has_authors:
                    if not d.author or d.author not in has_authors:
                        continue

                if has_tags:
                    if not d.tags or not set(d.tags).issubset(set(has_tags)):
                        continue

                if is_one_of_parties:
                    revised_parties = [].extend(is_one_of_parties)
                    for p in is_one_of_languages:
                        p = p.lower()
                        if p == 'english':
                            revised_parties.append('en')
                        if p == 'german':
                            revised_parties.append('de')
                    if not d.party or d.party.lower() not in revised_parties:
                        continue

                if ratings_in_range:
                    if not d.rating or d.rating not in ratings_in_range:
                        continue

                if has_one_of_keywords_with_english_translation:
                    if not d.keywords:
                        continue
                    keywords_en = [x['english_translation'] for x in d.keywords]
                    matched_keyword = False
                    for k in has_one_of_keywords_with_english_translation:
                        if k in keywords_en:
                            matched_keyword = True
                            break
                    if not matched_keyword:
                        continue

                if has_one_of_keywords_with_german_translation:
                    if not d.keywords:
                        continue
                    keywords_de = [x['german_translation'] for x in d.keywords]
                    matched_keyword = False
                    for k in has_one_of_keywords_with_german_translation:
                        if k in keywords_de:
                            matched_keyword = True
                            break
                    if not matched_keyword:
                        continue

                if has_one_of_keywords:
                    if not d.keywords:
                        continue
                    keywords_de = [x['german_translation'] for x in d.keywords]
                    keywords_en = [x['english_translation'] for x in d.keywords]
                    matched_keyword = False
                    for k in has_one_of_keywords:
                        if k in keywords_de or k in keywords_en:
                            matched_keyword = True
                            break
                    if not matched_keyword:
                        continue

                filtered.append(d)
            except:
                logging.exception("An exception occured while applying filters, skipping document")
                continue

        return Corpus(source=filtered,
                      name=corpus.name,
                      language=corpus.language,
                      has_assigned_keywords=corpus.has_assigned_keywords,
                      has_translated_keywords=corpus.has_translated_keywords)


class KeywordTranslator:
    def __init__(self, cache_file='translator_cache.json', timeout=1.0, google_client_secrets_file=None):
        if google_client_secrets_file:

            appflow = flow.InstalledAppFlow.from_client_secrets_file(
                google_client_secrets_file,
                scopes=[
                    'https://www.googleapis.com/auth/cloud-platform'
                ])
            appflow.run_local_server()  # launch browser
            # appflow.run_console()
            self.translator = g_translate.Client(credentials=appflow.credentials)

        else:  # fallback
            self.translator = googletrans.Translator()
        self.cache_file = cache_file
        self.timeout = timeout
        try:
            self.load_cache_from_file()
        except Exception as e:
            logging.warning("Loading of file failed")
            logging.warning(e)
            self.cache = dict()

    # def __del__(self):
    #     self.save_cache()
    #     logging.info("saved cache file")

    def translate(self, keyword: Keyword):
        if keyword.source_language is Language.UNKNOWN:
            logging.debug("KeywordTranslator: source is unknown, skipping")

        elif keyword.source_language is Language.DE:
            self.translate_german2english(keyword)

        elif keyword.source_language is Language.EN:
            self.translate_english2german(keyword)
        else:
            raise UserWarning("Wrong language term!")

    def translate_german2english(self, keyword):
        if keyword.german_translation in self.cache and self.cache[keyword.german_translation].english_translation:
            print(f"found keyword '{keyword.german_translation}' in cache, taking this one")
            # keyword = self.cache[keyword.german_translation]
            keyword.english_translation = self.cache[keyword.german_translation].english_translation
            return keyword
        else:
            if not keyword.english_translation:
                logging.debug("KeywordTranslator: {} | source is DE, EN not set, translating ...".format(
                    keyword.german_translation))
                try:
                    if isinstance(self.translator, g_translate.Client):

                        text_input = str(keyword.german_translation)
                        if isinstance(text_input, six.binary_type):
                            text_input = text_input.decode('utf-8')

                        result = self.translator.translate(text_input, target_language=keyword.source_language.value)
                        keyword.english_translation = result['translatedText']

                    else:
                        time.sleep(self.timeout)
                        translated = self.translator.translate(text=keyword.german_translation,
                                                               src=str(Language.DE.value),
                                                               dest=str(Language.EN.value))
                        keyword.english_translation = translated.text

                    self.add_to_cache(keyword)
                    return keyword
                except Exception as e:
                    logging.error("While trying to translate, an error occured:")
                    logging.error(e)

            else:
                logging.debug(
                    "KeywordTranslator: {}, {} | source is DE, but EN already set, skipping translation".format(
                        keyword.german_translation, keyword.english_translation))
                return None

    def translate_english2german(self, keyword):

        if keyword.english_translation in self.cache and self.cache[keyword.english_translation].german_translation:
            print(f"found keyword '{keyword.english_translation}' in cache, taking this one")
            # keyword = self.cache[keyword.english_translation]
            keyword.german_translation = self.cache[keyword.english_translation].german_translation
            return keyword
        else:

            if not keyword.german_translation:
                logging.debug("KeywordTranslator: {} | source is EN, DE not set, translating ...".format(
                    keyword.english_translation))
                try:
                    if isinstance(self.translator, g_translate.Client):

                        text_input = str(keyword.english_translation)
                        if isinstance(text_input, six.binary_type):
                            text_input = text_input.decode('utf-8')

                        result = self.translator.translate(text_input, target_language=keyword.source_language.value)
                        keyword.german_translation = result['translatedText']

                    else:
                        time.sleep(self.timeout)
                        translated = self.translator.translate(text=keyword.english_translation,
                                                               src=str(Language.EN.value),
                                                               dest=str(Language.DE.value))
                        keyword.german_translation = translated.text

                    self.add_to_cache(keyword)
                    return keyword
                except Exception as e:
                    logging.error("While trying to translate, an error occured:")
                    logging.error(e)
            else:
                logging.debug(
                    "KeywordTranslator: {}, {}| source is EN, but DE already set, skipping translation".format(
                        keyword.english_translation, keyword.german_translation))
                return None

    def translate_corpus(self, corpus: Corpus):
        list_of_keywords = []

        for document in corpus.get_documents():
            for kw in document.keywords:
                self.translate(kw)
                list_of_keywords.append(kw)
                print('{} \t {} \t\t\t {}'.format(kw.source_language, kw.english_translation, kw.german_translation))

        return list_of_keywords

    def load_cache_from_file(self):
        self.cache = dict()
        if not os.path.exists(self.cache_file):
            raise Exception("File does not exist")

        with open(self.cache_file, encoding='utf-8') as json_file:
            cache_data = json.load(json_file)
            for key in cache_data:
                self.cache[key] = Keyword.from_json(cache_data[key])

    def add_to_cache(self, keyword: Keyword):
        self.cache[keyword.german_translation] = keyword
        self.cache[keyword.english_translation] = keyword

    def save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=1, default=lambda o: o.__dict__)


class KeywordMatcher:
    def __init__(self, corpus_0, corpus_1):
        self.corpus_0 = corpus_0
        self.corpus_1 = corpus_1
        self.matches = None
        self.translations = None
        self.already_matched = False

    @staticmethod
    def lemmatize(keyword_collection: Union[Dict[int, List[Keyword]], List[Document], Corpus], german_model,
                  english_model):
        for instance in tqdm(keyword_collection, total=len(keyword_collection), desc="Lemmatize...", position=0,
                             leave=True):
            if isinstance(keyword_collection, dict):
                keywords = keyword_collection[instance]
            elif isinstance(keyword_collection, list):
                keywords = instance.keywords
            elif isinstance(keyword_collection, Corpus):
                keywords = instance.keywords
            else:
                raise NotImplementedError("Not supported type!")
            for keyword in keywords:
                keyword.lemmatize(german_model, english_model)

    @staticmethod
    def group_by_key(keyword_collection: Union[Dict[int, List[Keyword]], List[Document]],
                     ger_translations: Dict[str, Set[str]] = None,
                     en_translations: Dict[str, Set[str]] = None):
        reversed_keywords = defaultdict(set)
        if ger_translations is None:
            ger_translations = defaultdict(set)
        if en_translations is None:
            en_translations = defaultdict(set)

        for instance in tqdm(keyword_collection, total=len(keyword_collection), desc="Group keys...", position=0,
                             leave=True):

            if isinstance(keyword_collection, dict):
                keywords = keyword_collection[instance]
                identifier = instance
            elif isinstance(keyword_collection, list):
                keywords = instance.keywords
                identifier = instance.doc_id
            elif isinstance(keyword_collection, Corpus):
                keywords = instance.keywords
                identifier = instance.doc_id
            else:
                raise NotImplementedError("Not suppported type!")
            if keywords:
                for keyword in keywords:
                    # TODO: Here!
                    if keyword.german_translation is not None and keyword.german_translation != "":
                        reversed_keywords[keyword.german_translation].add(identifier)
                    if keyword.english_translation is not None and keyword.english_translation != "":
                        reversed_keywords[keyword.english_translation].add(identifier)

                    if keyword.german_translation is not None and keyword.english_translation is not None \
                            and keyword.german_translation != "" and keyword.english_translation != "":
                        ger_translations[keyword.german_translation].add(keyword.english_translation)
                        en_translations[keyword.english_translation].add(keyword.german_translation)

        return reversed_keywords, ger_translations, en_translations

    def match_corpora(self,
                      lemmatize: bool = True,
                      simplify_result: bool = True) \
            -> Tuple[Dict[Keyword, Tuple[List[int], List[int]]], Dict[str, str]]:
        print("Start matching...")
        if lemmatize:
            german_model = spacy.load("de_core_news_sm")
            english_model = spacy.load("en_core_web_sm")
            KeywordMatcher.lemmatize(self.corpus_0, german_model, english_model)
            KeywordMatcher.lemmatize(self.corpus_1, german_model, english_model)

        # groups ids or years by keywords
        reversed_keywords_1, ger_translations, en_translations = KeywordMatcher.group_by_key(self.corpus_0)
        reversed_keywords_2, ger_translations, en_translations = KeywordMatcher.group_by_key(self.corpus_1,
                                                                                             ger_translations,
                                                                                             en_translations)

        # matches years or ids by iterating through keywords in common
        matched_keywords = set()
        for keyword in reversed_keywords_1:
            if keyword in reversed_keywords_2:
                matched_keywords.add(keyword)

        for keyword in reversed_keywords_2:
            if keyword in reversed_keywords_1:
                matched_keywords.add(keyword)
        result_dict = {keyword: (reversed_keywords_1[keyword], reversed_keywords_2[keyword]) for keyword in
                       tqdm(matched_keywords, total=len(matched_keywords), desc="Match...", position=0, leave=True)}

        # uses only english keyword versions
        if simplify_result:
            filtered_result = {}
            for keyword, result in tqdm(result_dict.items(), total=len(result_dict), desc="Simplify..."):
                if keyword in ger_translations.keys():
                    for translation in ger_translations[keyword]:
                        if translation in result_dict.keys():
                            new_result = (
                                result[0].union(result_dict[translation][0]),
                                result[1].union(result_dict[translation][1]))
                            filtered_result[translation] = new_result
                        else:
                            filtered_result[translation] = result
                if keyword in en_translations.keys():
                    filtered_result[keyword] = result_dict[keyword]

            result_dict = filtered_result

        useful_translations = {keyword: translations for keyword, translations in en_translations.items()
                               if keyword in result_dict}
        self.matches = result_dict
        self.already_matched = True
        return result_dict, useful_translations

    @staticmethod
    def get_ngrams_of_matched(matched_corpus_ids,
                              keyword_collection: Union[Dict[int, List[Keyword]], List[Document], Corpus]):
        ngrams = set()

        # for instance in keyword_collection:
        #     if isinstance(keyword_collection, dict):
        #         keywords = keyword_collection[instance]
        #     elif isinstance(keyword_collection, list):
        #         keywords = instance.keywords

        for instance in keyword_collection:
            if isinstance(keyword_collection, dict):
                keywords = keyword_collection[instance]
                identifier = instance
            elif isinstance(keyword_collection, list):
                keywords = instance.keywords
                identifier = instance.doc_id
            elif isinstance(keyword_collection, Corpus):
                keywords = instance.keywords
                identifier = instance.doc_id

            else:
                raise NotImplementedError("Not supported type!")

            if identifier in matched_corpus_ids:
                # todo: add real ngrams?
                # ngrams.update(document.text.split())
                candidates = [keyword.german_translation for keyword in keywords] + \
                             [keyword.english_translation for keyword in keywords]
                ngrams.update(candidates)

        return ngrams

    def get_common_keyword_vocab(self):
        if not self.already_matched:
            raise UserWarning("Not macthed yet, call match_corpora first")
        matched_ids_src1 = set()
        matched_ids_src2 = set()
        for matched_keyword, documents in self.matches.items():
            src1, src2 = documents
            matched_ids_src1.update(src1)
            matched_ids_src2.update(src2)

        ngrams_1 = KeywordMatcher.get_ngrams_of_matched(matched_ids_src1, self.corpus_0)
        ngrams_2 = KeywordMatcher.get_ngrams_of_matched(matched_ids_src2, self.corpus_1)
        # print(ngrams_1.difference(ngrams_2))
        # print(ngrams_2.difference(ngrams_1))
        # print(ngrams_1.intersection(ngrams_2))
        return ngrams_1.intersection(ngrams_2)

    # if tf_mode it returns term frequency, document frequency otherwise
    def get_keyword_counts(self, common_keywords, as_dict=False, tf_mode=False):
        if as_dict:
            tuples = {}
        else:
            tuples = []
        for keyword in common_keywords:
            if keyword:
                corp_0_ids, corp_1_ids = self.matches[keyword]
                if tf_mode:
                    corp_0_counts = 0
                    corp_1_counts = 0
                    for doc_id in corp_0_ids:
                        corp_0_counts += self.corpus_0.documents[doc_id].text.count(keyword)
                    for doc_id in corp_1_ids:
                        corp_1_counts += self.corpus_1.documents[doc_id].text.count(keyword)

                else:
                    corp_0_counts = len(corp_0_ids)
                    corp_1_counts = len(corp_1_ids)

                if as_dict:
                    tuples[keyword] = (corp_0_counts, corp_1_counts)
                else:
                    tuples.append((corp_0_counts, corp_1_counts, keyword))

        if not as_dict:
            tuples = sorted(tuples, reverse=True)
        return tuples


    def get_first_mentions(self, common_keywords):
        mentions = {}
        for keyword in common_keywords:
            if keyword:
                corp_0_ids, corp_1_ids = self.matches[keyword]
                corp_0 = [self.corpus_0.documents[corp_id] for corp_id in corp_0_ids]
                corp_1 = [self.corpus_1.documents[corp_id] for corp_id in corp_1_ids]
                years_0 = [doc.date for doc in corp_0]
                years_1 = [doc.date for doc in corp_1]
                mentions[keyword] = (min(years_0), min(years_1))
        return mentions

    def liklihood_ratio_test(self, term: str, keyword_counts: Dict[str, Tuple[int, int]], n_1: int, n_2: int) -> float:
        # k = term freq
        # n = all freq
        def liklihood(prob: float, k: int, n: int) -> float:
            res = (prob ** k) * (1 - prob) ** (n - k)
            if res == 0:
                res = 0.000000000000001
            return res

        k_1 = keyword_counts[term][0]
        k_2 = keyword_counts[term][1]
        if n_1 == 0 or n_2 == 0:
            return 0
        p_1 = k_1 / n_1
        p_2 = k_2 / n_2
        p = (k_1 + k_2) / (n_1 + n_2)
        return 2 * (math.log(liklihood(p_1, k_1, n_1)) + math.log(liklihood(p_2, k_2, n_2))
                    - math.log(liklihood(p, k_1, n_1)) - math.log(liklihood(p, k_2, n_2)))

    def perform_liklihood_ratio_test(self, tf_mode=False) -> List[Tuple[str, float]]:
        common = self.get_common_keyword_vocab()
        keyword_counts = self.get_keyword_counts(common, as_dict=True, tf_mode=tf_mode)
        n_1 = 0
        n_2 = 0
        for keyword, counts in keyword_counts.items():
            n_1 += counts[0]
            n_2 += counts[1]

        ranked_terms = [(term, self.liklihood_ratio_test(term, keyword_counts, n_1, n_2)) for term in common if term]
        ranked_terms = sorted(ranked_terms, key=lambda x: x[1], reverse=True)
        return ranked_terms


class DataHandler:
    @staticmethod
    def get_sustainability_data(path):
        def nan_resolver(pandas_attribut):
            return None if pd.isna(pandas_attribut) else pandas_attribut

        df = pd.read_csv(path)
        print(df.columns)
        return [Document(text=row["content"],
                         date=nan_resolver(row["PY"]),
                         language=Language.EN,
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

        data_replaced = re.sub(
            r"\nER\n?\n(FN|VR|PT|AU|AF|BA|BF|CA|GP|BE|TI|SO|SE|BS|LA|DT|CT|CY|CL|SP|HO|DE|ID|AB|C1|RP|EM|RI"
            r"|OI|FU|FX|CR|NR|TC|Z9|U1|U2|PU|PI|PA|SN|EI|BN|J9|JI|PD|PY|VL|IS|SI|PN|SU|MA|BP|EP|AR"
            r"|DI|D2|EA|EY|PG|P2|WC|SC|GA|PM|UT|OA|HP|HC|DA|ER|EF)", r"###!!!#?#!!!###\g<1>", data)
        records = data_replaced.split("###!!!#?#!!!###")

        abstracts = []
        wos_categories = re.compile(
            '^(FN|VR|PT|AU|AF|BA|BF|CA|GP|BE|TI|SO|SE|BS|LA|DT|CT|CY|CL|SP|HO|DE|ID|AB|C1|RP|EM|RI'
            '|OI|FU|FX|CR|NR|TC|Z9|U1|U2|PU|PI|PA|SN|EI|BN|J9|JI|PD|PY|VL|IS|SI|PN|SU|MA|BP|EP|AR'
            '|DI|D2|EA|EY|PG|P2|WC|SC|GA|PM|UT|OA|HP|HC|DA|ER|EF)\\s(.*)', re.MULTILINE)
        no_text_counter = 0
        no_year_counter = 0
        no_year_and_no_text_counter = 0
        no_year_but_text_counter = 0
        no_text_but_year_counter = 0
        for i, record in tqdm(enumerate(records), desc="Load abstracts", total=len(records)):
            attributes = re.findall(wos_categories, record)

            wos_dict = defaultdict(lambda: None,
                                   {attribute[0]: ' '.join(list(attribute[1:])).strip() for attribute in attributes})
            text = wos_dict["AB"]
            year = wos_dict["PY"]

            if year is None:
                year = wos_dict["EY"]

            if text is None:
                if wos_dict["TI"]:
                    text = wos_dict["TI"]
                    if wos_dict["DE"]:
                        text += " " + str(wos_dict["DE"])
                else:
                    if wos_dict["DE"]:
                        text = wos_dict["DE"]

            if text is None and year is None:
                no_text_but_year_counter += 1
            if text and year is None:
                no_year_but_text_counter += 1
            if text is None and year:
                no_text_but_year_counter += 1
            if text is None:
                no_text_counter += 1
            if year is None:
                no_year_counter += 1

            if text and year:
                # print(text)
                if not year.isdecimal():
                    print(year)

                authors = wos_dict["AF"]
                if authors is None:
                    authors = wos_dict["BF"]
                    if authors is None:
                        authors = wos_dict["GP"]

                abstracts.append(Document(text=str(text),
                                          date=year,
                                          language=Language.EN,
                                          doc_id=f'sc_{i}',
                                          tags=wos_dict["DE"],
                                          author=authors))
            else:
                print(record)
                print("##########################################")

        print(no_year_but_text_counter, no_text_but_year_counter, no_year_and_no_text_counter, no_text_counter,
              no_year_counter, len(abstracts), no_text_counter + no_year_counter + len(abstracts))

        return abstracts

    @staticmethod
    def get_bundestag_speeches(directory):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(directory):
            files.extend(filenames)
            break

        for f in files:
            if f.startswith('.') or not f[-4:] == '.csv':
                files.remove(f)

        speeches = []
        index = 0
        for file in tqdm(files, desc="load Bundestag speeches", total=len(files)):
            df = pd.read_csv(os.path.join(directory, file))
            for i, row in df.iterrows():
                if not pd.isna(row["Speech text"]):
                    speeches.append(Document(text=row["Speech text"],
                                             date=row["Date"][:4],
                                             language=Language.DE,
                                             doc_id=f'po_{index}',
                                             author=row["Speaker"],
                                             party=row["Speaker party"],
                                             rating=row["Interjection count"]))
                    index += 1

        return speeches

    @staticmethod
    def get_un_texts(directory):
        files = []
        for path in Path(directory).rglob('*.xml'):
            files.append(path.absolute())

        texts = []
        index = 0
        for file in tqdm(files, desc="load UN texts", total=len(files)):
            try:
                root = et.parse(file).getroot()
            except et.ParseError:
                print("Parse Error:", file)
                continue

            text = []

            date = None
            for type_tag in root.findall('teiHeader/fileDesc/publicationStmt/date'):
                try:
                    date = int(type_tag.text[:4])
                except ValueError:
                    date = re.search(r'UNv1\.0-TEI[\\/]+en[\\/]+(\d\d\d\d)', str(file)).group(1)
            if date is None:
                raise UserWarning("No date")
            for type_tag in root.findall('text/body/p/s'):
                if type_tag.text:
                    text.append(type_tag.text)
            try:
                text = [t for t in text if t]
                text = " ".join(text)
            except TypeError:
                print("Text Type Error", file, text)
                continue
            texts.append(Document(text=text,
                                  date=date,
                                  language=Language.EN,
                                  doc_id=f'un_{index}'))
            index += 1

        return texts

    @staticmethod
    def get_state_of_the_union(directory: str):
        president_infos = pd.read_csv(os.path.join(directory, "presidents.csv"))
        president2infos= {row["President"].split()[-1]: {"party": row["Party"],
                                                         "full name": row["President"]}
                           for i, row in president_infos.iterrows()}

        speeches = os.path.join(directory, "speeches")

        files = []
        for path in Path(speeches).rglob('*.txt'):
            files.append(path.absolute())

        documents = []
        for index, file in enumerate(files):
            with open(str(file), mode="r", encoding="utf-8") as f:
                data = f.read()
                date = int(re.search(r'_(\d\d\d\d)\.txt', str(file)).group(1))
                president = re.search(r'(.*)_(\d\d\d\d)\.txt', str(file.name)).group(1)
                documents.append(Document(text=re.sub(r'\s+', " ",  data),
                                          date=date,
                                          author=president2infos[president]["full name"],
                                          party=president2infos[president]["party"],
                                          language=Language.EN,
                                          doc_id=f'sotu_{index}'))
        return documents


if __name__ == '__main__':
    # files = [
    #     "E:/mcc/UN/UNv1.0-TEI/en/2001/e/c_12/q/t_t/1.xml",
    #     # "E:/mcc/UN/UNv1.0-TEI/en/2013/ccpr/c/107/d/1835_1837/2008.xml"
    # ]
    # texts = []
    # index = 0
    # parser = et.XMLParser(encoding="utf-8")
    # for file in tqdm(files, desc="load UN texts", total=len(files)):
    #
    #     root = et.parse(file, parser).getroot()
    #     text = []
    #
    #     date = None
    #     for type_tag in root.findall('teiHeader/fileDesc/publicationStmt/date'):
    #         try:
    #             date = int(type_tag.text[:4])
    #         except ValueError:
    #             date = re.search(r'UNv1\.0-TEI[\\/]+en[\\/]+(\d\d\d\d)', str(file)).group(1)
    #     if date is None:
    #         raise UserWarning("No date")
    #     for type_tag in root.findall('text/body/p/s'):
    #         if type_tag.text:
    #             text.append(type_tag.text)
    #     try:
    #         text = [t for t in text if t]
    #         text = " ".join(text)
    #     except TypeError:
    #         print("Text Type Error", file, text)
    #         continue
    #     texts.append(Document(text=text,
    #                           date=date,
    #                           language=Language.EN,
    #                           doc_id=f'un_{index}'))
    #     index += 1

    monkey = Keyword(german_translation="Affe", english_translation="Monkey")
    affe = Keyword(german_translation="Affe", english_translation="Ape")
    affe2 = Keyword(german_translation="Affe", english_translation="Ape")
    weird_monkey = Keyword(english_translation="Affe", german_translation="Monkey")
    patch = Keyword(german_translation="Patch", english_translation="patch")

    print(patch.__dict__)
    print(monkey == affe)
    print(affe == affe2)
    print(monkey == weird_monkey)
    print(affe == weird_monkey)
    print(monkey == patch)

    # Example matching:
    # km = KeywordMatcher(sampled_corpora[1], sampled_corpora[4])
    # km.match_corpora(lemmatize=False, simplify_result=False)
    # common = km.get_common_keyword_vocab()
    # km.get_first_mentions(common)

    # print(monkey in list({affe, affe2}))

    # key word translator example
    # kwt = KeywordTranslator(cache_file=config["translator"]["cache_file"],
    #                         google_client_secrets_file=config["translator"]["google_client_secret_file"])


    # def translate(keyword):
    #     print('before: {}\t| {}\t | {}'.format(keyword.english_translation, keyword.german_translation,
    #                                            keyword.source_language))
    #     kwt.translate(keyword)
    #     print('after:  {}\t| {}\t | {}\n'.format(keyword.english_translation, keyword.german_translation,
    #                                              keyword.source_language))


    # print('translated')
    #
    # translate(Keyword(english_translation="axis of evil"))
    #
    # translate(Keyword(german_translation="Achse des Bösen"))
    #
    # translate(Keyword(english_translation="axis of evil", german_translation="Achse des Bösen"))
    #
    # kwt.save_cache()
