import os
import time
import json
import logging
import six

import googletrans
from google.cloud import translate_v2 as g_translate
from google_auth_oauthlib import flow

from utils import Language, Keyword, ConfigLoader


def load_cache_from_file(cache_file):
    cache = dict()
    if not os.path.exists(cache_file):
        raise Exception("File does not exist")

    with open(cache_file, encoding='utf-8') as json_file:
        cache_data = json.load(json_file)
        for key in cache_data:
            cache[key] = Keyword.from_json(cache_data[key])
    return cache


def add_to_cache(keyword: Keyword, cache):
    cache[keyword.german_translation] = keyword
    cache[keyword.english_translation] = keyword


def translate_german2english(keyword, cache, translator, timeout):
    if keyword.german_translation in cache and cache[keyword.german_translation].english_translation:
        print(f"found keyword '{keyword.german_translation}' in cache, taking this one")
        # keyword = self.cache[keyword.german_translation]
        keyword.english_translation = cache[keyword.german_translation].english_translation
        return keyword
    else:
        if not keyword.english_translation:
            logging.debug("KeywordTranslator: {} | source is DE, EN not set, translating ...".format(
                keyword.german_translation))
            try:
                if isinstance(translator, g_translate.Client):

                    text_input = str(keyword.german_translation)
                    if isinstance(text_input, six.binary_type):
                        text_input = text_input.decode('utf-8')

                    result = translator.translate(text_input, target_language=keyword.source_language.value)
                    keyword.english_translation = result['translatedText']

                else:
                    time.sleep(timeout)
                    translated = translator.translate(text=keyword.german_translation,
                                                           src=str(Language.DE.value),
                                                           dest=str(Language.EN.value))
                    keyword.english_translation = translated.text

                add_to_cache(keyword)
                return keyword
            except Exception as e:
                logging.error("While trying to translate, an error occured:")
                logging.error(e)

        else:
            logging.debug(
                "KeywordTranslator: {}, {} | source is DE, but EN already set, skipping translation".format(
                    keyword.german_translation, keyword.english_translation))
            return None


def translate_english2german(keyword, cache, translator, timeout):

    if keyword.english_translation in cache and cache[keyword.english_translation].german_translation:
        print(f"found keyword '{keyword.english_translation}' in cache, taking this one")
        # keyword = self.cache[keyword.english_translation]
        keyword.german_translation = cache[keyword.english_translation].german_translation
        return keyword
    else:

        if not keyword.german_translation:
            logging.debug("KeywordTranslator: {} | source is EN, DE not set, translating ...".format(
                keyword.english_translation))
            try:
                if isinstance(translator, g_translate.Client):

                    text_input = str(keyword.english_translation)
                    if isinstance(text_input, six.binary_type):
                        text_input = text_input.decode('utf-8')

                    result = translator.translate(text_input, target_language=keyword.source_language.value)
                    keyword.german_translation = result['translatedText']

                else:
                    time.sleep(timeout)
                    translated = translator.translate(text=keyword.english_translation,
                                                           src=str(Language.EN.value),
                                                           dest=str(Language.DE.value))
                    keyword.german_translation = translated.text

                add_to_cache(keyword)
                return keyword
            except Exception as e:
                logging.error("While trying to translate, an error occured:")
                logging.error(e)
        else:
            logging.debug(
                "KeywordTranslator: {}, {}| source is EN, but DE already set, skipping translation".format(
                    keyword.english_translation, keyword.german_translation))
            return None

def main():
    # parser = argparse.ArgumentParser(description='Extracts keywords for given algorithm on given corpora')
    # parser.add_argument('-a', '--algorithm', help='Algorithm to use like rake or tfidf', default="rake")
    # parser.add_argument('-c', '--corpora', help='Corpora to annotate as list', nargs='+',
    #                     default=['state_of_the_union'])
    # parser.add_argument('-t', '--translate', help='Translate keywords', action='store_true')
    # args = vars(parser.parse_args())
    #
    # config = ConfigLoader.get_config()
    #
    # #  remove and use actual args
    # chosen_corpora = [
    #     # 'state_of_the_union',
    #     'bundestag', 'abstract', 'sustainability'
    # ]  # args['corpora']
    #
    # PathMetaData = namedtuple('PathMetaData', 'path corpus_name language')
    # paths_and_meta_data = [
    #     PathMetaData(config["corpora"]["bundestag_corpus"], "bundestag", Language.DE),
    #     PathMetaData(config["corpora"]["abstract_corpus"], "abstract", Language.EN),
    #     PathMetaData(config["corpora"]["sustainability_corpus"], "sustainability", Language.EN),
    #     PathMetaData(config["corpora"]["state_of_the_union_corpus"], "state_of_the_union", Language.EN),
    #     PathMetaData(config["corpora"]["united_nations_corpus"], "united_nations", Language.EN)
    # ]
    # paths_and_meta_data = [path_meta for path_meta in paths_and_meta_data if path_meta.corpus_name in chosen_corpora]

    config = ConfigLoader.get_config()

    paths = ['data/state_of_the_union_corpus_rake_keywords.json']  # TODO

    cache_file = config["translator"]["cache_file"]
    google_client_secrets_file = config["translator"]["google_client_secret_file"]
    translator = None
    timeout = None

    if google_client_secrets_file and not google_client_secrets_file == "":

        appflow = flow.InstalledAppFlow.from_client_secrets_file(
            google_client_secrets_file,
            scopes=[
                'https://www.googleapis.com/auth/cloud-platform'
            ])
        appflow.run_local_server()  # launch browser
        # appflow.run_console()
        translator = g_translate.Client(credentials=appflow.credentials)

    else:  # fallback
        translator = googletrans.Translator()

    cache = None
    timeout = timeout

    try:
        cache = load_cache_from_file(cache_file)
    except Exception as e:
        logging.warning("Loading of file failed")
        logging.warning(e)
        cache = dict()

    for path in paths:
        with open(path) as f:
            data = json.load(f)
            for doc_id, keywords in data:
                en_translation = keywords["english_translation"]
                ger_translation = keywords["german_translation"]
                print(keywords)
                if en_translation is None:
                    translate_german2english(keyword, cache, translator, timeout)
                if ger_translation is None:
                    translate_english2german(keyword, cache, translator, timeout)


if __name__ == '__main__':
    main()
