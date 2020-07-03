import json
import time
import logging
from typing import List, Union
from flask import Flask, render_template, request, Response

from utils import ConfigLoader, Corpus, Keyword, KeywordType, Language, KeywordTranslator, CorpusFilter
from simple_statistics import yearwise_documents

logging.basicConfig(level=logging.INFO)


def modify_path(path: str, algorithm: str):
    return path.replace('.json', f'_{algorithm}.json')


logging.info('importing corpora ...')
config = ConfigLoader.get_config()

corpus_data = {}
keyword_data = {}
min_year = 5000
max_year = 0

logging.info('importing corpora and keywords data ...')
start_time = time.time()

for corpus_name in config["corpora_for_viz"]:
    logging.info(corpus_name)
    with open(config["corpora_for_viz"][corpus_name]["corpus"]) as corpus_file:
        corpus_data[corpus_name] = json.load(corpus_file)
    with open(config["corpora_for_viz"][corpus_name]["keywords"]) as keyword_file:
        keyword_data[corpus_name] = json.load(keyword_file)
    logging.info('loading {} data took {}s'.format(corpus_name, int(time.time() - start_time)))

logging.info('finding min and max year over all corpora ...')
for corpus_name in corpus_data:
    for doc_id in corpus_data[corpus_name]:
        doc = corpus_data[corpus_name][doc_id]
        if doc['date']:
            date = int(doc['date'])
            if date > max_year:
                max_year = date
            if date < min_year:
                min_year = date
logging.info('min year: {}, max_year: {}'.format(min_year, max_year))

logging.info('calculating number of documents per year for all corpora ...')
number_of_documents_per_year = {}
for corpus_name in corpus_data:
    number_of_documents_per_year[corpus_name] = [0] * (max_year-min_year+1)
    for doc_id in corpus_data[corpus_name]:
        doc = corpus_data[corpus_name][doc_id]
        if not doc['date']:
            continue
        number_of_documents_per_year[corpus_name][int(doc['date'])-min_year] += 1

logging.info('importing translation cache ...')
with open(config["translator"]["cache_file"]) as f:
    translation_cache = json.load(f)

logging.info('starting flask ...')
app = Flask(__name__)
logging.info('server boot took about {}s'.format(int(time.time()-start_time)))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/data', methods=["POST"])
def data():
    data = request.get_json() # array of keywords
    result = get_documents_per_year_filtered_by(data)
    return Response(json.dumps(result),  mimetype='application/json')


def process_documents_from_inverse_keyword(keyword, c_name):
    df = [0] * (max_year-min_year+1)
    norm = [0] * (max_year-min_year+1)
    tf = [0] * (max_year-min_year+1)

    for d in keyword:
        for d_id in d:
            year = corpus_data[c_name][d_id]["date"]
            index = year - min_year  # min_year has index 0
            term_frequency = d[d_id]  # within this document
            df[index] += 1  # one document more
            norm[index] = df[index] / number_of_documents_per_year[c_name][index]
            tf[index] += term_frequency

    return df, norm, tf


def get_documents_per_year_filtered_by(keywords):

    result = {
        'corpora': {c_name: {} for c_name in corpus_data},
        'years': list(range(min_year, max_year+1))
    }

    for c_name in corpus_data:

        result['corpora'][c_name] = {
            'df': {},  # document frequency per year
            'norm': {},  # document frequency per year, normalized by total documents per year
            'tf': {}  # term frequency per year
        }

        corpus_language = "de" if "bundestag" in c_name else "en"

        for key in keywords:
            key = key.strip().lower()

            translated_key = None
            if key in translation_cache["de2en"]:
                translated_key = translation_cache["de2en"][key].lower()
                translated_dest_lang = "en"
            elif key in translation_cache["en2de"]:
                translated_key = translation_cache["en2de"][key].lower()
                translated_dest_lang = "de"

            is_translation_key_relevant = translated_key in keyword_data[c_name] \
                                          and translated_dest_lang == corpus_language \
                                          and translated_key != key

            if key in keyword_data[c_name] or is_translation_key_relevant:
                df, norm, tf = process_documents_from_inverse_keyword(keyword_data[c_name][key], c_name)

                if is_translation_key_relevant:
                    transl_df, transl_norm, transl_tf = process_documents_from_inverse_keyword(keyword_data[c_name][translated_key], c_name)

                    # for translation: aggregate values from initial findings and translated findings, as we do not know
                    # in which language keywords got entered
                    result['corpora'][c_name]['df'][key] = [(df[i] + transl_df[i]) for i in range(max_year-min_year+1)]
                    result['corpora'][c_name]['norm'][key] = [(norm[i] + transl_norm[i]) for i in range(max_year-min_year+1)]
                    result['corpora'][c_name]['tf'][key] = [(tf[i] + transl_tf[i]) for i in range(max_year-min_year+1)]
                else:
                    result['corpora'][c_name]['df'][key] = df
                    result['corpora'][c_name]['norm'][key] = norm
                    result['corpora'][c_name]['tf'][key] = tf
            else:
                result['corpora'][c_name]['df'][key] = [0] * (max_year-min_year+1)
                result['corpora'][c_name]['norm'][key] = [0] * (max_year-min_year+1)
                result['corpora'][c_name]['tf'][key] = [0] * (max_year-min_year+1)

    return result


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host='127.0.0.1', port=8080)
