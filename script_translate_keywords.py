import argparse
import os
import time
import json
import logging
import six

import googletrans
from google.cloud import translate_v2 as g_translate
from google_auth_oauthlib import flow
from tqdm import tqdm

from utils import Language, Keyword, ConfigLoader

logging.basicConfig(level=logging.INFO)


def load_cache_from_file(cache_file):
    if not os.path.exists(cache_file):
        raise Exception("File does not exist")
    with open(cache_file, encoding='utf-8') as json_file:
        return json.load(json_file)


def save_cache_to_file(cache, cache_file):
    with open(cache_file, "w", encoding='utf-8') as json_file:
        json.dump(cache, json_file)


def add_to_cache(keyword, translated_keyword, cache, translation_direction):
    cache[translation_direction][keyword] = translated_keyword
    cache[translation_direction][keyword] = translated_keyword


def translate(keyword, cache, translator, timeout, dest):

    if dest == "de":
        translation_direction = "en2de"
    elif dest == "en":
        translation_direction = "de2en"
    else:
        raise UserWarning('Translation direction defined incorrectly"')

    src = "de" if dest == "en" else "en"

    if keyword in cache[translation_direction] and cache[translation_direction][keyword] != "":
        logging.debug('already in cache file')
        return cache[translation_direction][keyword]
    else:
        try:
            if isinstance(translator, g_translate.Client):

                text_input = str(keyword)
                if isinstance(text_input, six.binary_type):
                    text_input = text_input.decode('utf-8')

                result = translator.translate(text_input, target_language=dest)
                translated_keyword = result['translatedText']

            else:

                translated = translator.translate(text=keyword, src=src, dest=dest)
                time.sleep(timeout)
                translated_keyword = translated.text

            logging.debug(f"\"{keyword}\" translated to \"{translated_keyword}\"")
            add_to_cache(keyword, translated_keyword, cache, translation_direction)
            return keyword
        except Exception as e:
            logging.error("While trying to translate, an error occured:")
            logging.error(e)


def main():
    parser = argparse.ArgumentParser(description='Translates keywords in keyword files of given paths')
    parser.add_argument('-p', '--paths', help='Paths of keyword files to translate', nargs='+',
                        default=['data/state_of_the_union_corpus_rake_keywords.json'])
    args = vars(parser.parse_args())

    config = ConfigLoader.get_config()
    paths = args['paths']

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
        cache = {
            "de2en": {},
            "en2de": {}
        }

    def iterate_keywords(data):
        tqdm_bar = tqdm(data.items(), total=len(data.keys()))
        for doc_id, keywords in tqdm_bar:
            for keyword in keywords:
                en_translation = keyword["english_translation"]
                ger_translation = keyword["german_translation"]
                if en_translation is None:
                    translated = translate(ger_translation, cache, translator, timeout, dest="en")
                    keyword["english_translation"] = translated
                if ger_translation is None:
                    translated = translate(en_translation, cache, translator, timeout, dest="de")
                    keyword["german_translation"] = translated

    try:
        for path in paths:
            logging.debug(f'loading keywords at \"{path}\"')
            with open(path, encoding='utf-8') as f:
                data = json.load(f)

            logging.debug('translating keywords ...')
            iterate_keywords(data)

            logging.debug(f'saving keywords with translations at \"{path}\"')
            with open(path, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=1)

    except KeyboardInterrupt:
        logging.debug('process was interrupted')
    finally:
        logging.debug('saving ...')
        save_cache_to_file(cache, cache_file)


if __name__ == '__main__':
    main()
