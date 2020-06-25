import json
import logging
from typing import List, Union
from flask import Flask, render_template, request, Response

from utils import ConfigLoader, Corpus, Keyword, KeywordType, Language, KeywordTranslator, CorpusFilter
from simple_statistics import yearwise_documents

logging.info('importing corpora ...')
config = ConfigLoader.get_config()
corpora: List[Corpus] = [
#     Corpus(source=config["corpora"]["bundestag_corpus"], name="bundestag", language=Language.DE),
    Corpus(source=config["corpora"]["abstract_corpus"], name="abstract", language=Language.EN),
#     Corpus(source=config["corpora"]["sustainability_corpus"], name="sustainability", language=Language.EN),
    Corpus(source=config["corpora"]["state_of_the_union_corpus"], name="state_of_the_union", language=Language.EN),
#     Corpus(source=config["corpora"]["united_nations_corpus"], name="united_nations", language=Language.EN)
]

logging.info('starting flask ...')
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/data', methods=["POST"])
def data():
    data = request.get_json() # array of keywords
    result = get_documents_per_year_filtered_by(data)
    return Response(json.dumps(result),  mimetype='application/json')


def get_documents_per_year_filtered_by(keywords, do_normalize=True):

    def get_min_max_years():
        min_year = 5000
        max_year = 0
        for corpus in corpora:
            years, _ = yearwise_documents(corpus)
            if min(years) < min_year:
                min_year = min(years)
            if max(years) > max_year:
                max_year = max(years)
        return min_year, max_year

    min_year, max_year = get_min_max_years()

    result = {
        'corpora': {},
        'years': list(range(min_year, max_year+1))
    }
    for corpus in corpora:
        result['corpora'][corpus.name] = {}

        for key in keywords:
            if do_normalize:
                unfiltered_years = yearwise_documents(corpus, as_dict=True)
            filtered_corpus = CorpusFilter.filter(corpus=corpus, has_one_of_keywords=key)
            years_counts = yearwise_documents(filtered_corpus, as_dict=True)

            result['corpora'][corpus.name][key] = {
                'df': [],  # document frequency per year
                'norm': [],  # document frequency per year, normalized by total documents per year
                'tf': []  # term frequency per year
            }
            for i in range(min_year, max_year+1):
                if i in years_counts:
                    result['corpora'][corpus.name][key]['norm'].append(years_counts[i]/unfiltered_years[i])
                    result['corpora'][corpus.name][key]['df'].append(years_counts[i])
                    result['corpora'][corpus.name][key]['tf'].append(0) # TODO: add term frequency
                else:
                    result['corpora'][corpus.name][key]['norm'].append(0)
                    result['corpora'][corpus.name][key]['df'].append(0)
                    result['corpora'][corpus.name][key]['tf'].append(0)
    return result


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='127.0.0.1', port=8080)
