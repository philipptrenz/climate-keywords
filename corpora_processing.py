from collections import defaultdict, Counter, OrderedDict
from typing import List, Set, Dict, NamedTuple, Callable, Union
import pandas as pd
from rake_nltk import Rake
from nltk.corpus import stopwords
from tqdm import tqdm
import pke
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import logging

from utils import Corpus, Document, ConfigLoader, DataHandler, Keyword, KeywordTranslator, KeywordType, Language, DocumentsFilter


class KeyPhraseExtractor:
    min_nrgam = 1
    max_ngram = 4

    @classmethod
    def tf_skl(cls, corpus: Corpus, top_k: int = 10):

        if corpus.language == Language.EN:
            stop_words = stopwords.words("English")
        elif corpus.language == Language.DE:
            stop_words = stopwords.words("German")

        count_vectorizer = CountVectorizer(stop_words=stop_words,
                                           ngram_range=(cls.min_nrgam, cls.max_ngram),
                                           min_df=2)
        tf_matrix = count_vectorizer.fit_transform([document.text for document in corpus.get_documents(as_list=True)])
        doc_id_lookup = {i: document.doc_id for i, document in enumerate(corpus.get_documents(as_list=True))}

        features = count_vectorizer.get_feature_names()

        keywords = {}

        for i, doc in tqdm(enumerate(tf_matrix), desc="Calculating tf", total=tf_matrix.shape[0]):
            df = pd.DataFrame(doc.T.todense(), index=features,
                              columns=["tf"])
            top_key_words = df.sort_values(by=["tf"], ascending=False)[:top_k]
            keywords[doc_id_lookup[i]] = list(top_key_words.index)

        corpus.assign_keywords(keywords=keywords, keyword_type=KeywordType.TF_SKL)

    @classmethod
    def tfidf_skl(cls, corpus: Corpus, top_k: int = 10):
        if corpus.language == Language.EN:
            stop_words = stopwords.words("english")
        elif corpus.language == Language.DE:
            stop_words = stopwords.words("german")

        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words,
                                           ngram_range=(cls.min_nrgam, cls.max_ngram),
                                           min_df=2)
        tfidf_matrix = tfidf_vectorizer.fit_transform([document.text for document in corpus.get_documents(as_list=True)])
        doc_id_lookup = {i: document.doc_id for i, document in enumerate(corpus.get_documents(as_list=True))}

        features = tfidf_vectorizer.get_feature_names()

        keywords = {}

        for i, doc in tqdm(enumerate(tfidf_matrix), desc="Calculating tf-idf", total=tfidf_matrix.shape[0]):
            df = pd.DataFrame(doc.T.todense(), index=features,
                              columns=["tfidf"])
            top_key_words = df.sort_values(by=["tfidf"], ascending=False)[:top_k]
            keywords[doc_id_lookup[i]] = list(top_key_words.index)

        corpus.assign_keywords(keywords=keywords, keyword_type=KeywordType.TFIDF_SKL)

    @classmethod
    def tfidf_pke(cls, corpus: Corpus):
        key_words_of_documents = {}
        number_keywords = 10
        stop_list = list(string.punctuation)
        # 1. create a TfIdf extractor.
        extractor = pke.unsupervised.TfIdf()
        # 2. load the content of the document.

        if corpus.language == Language.DE:
            lan = "de"
        else:
            lan = "en"

        for document in corpus.get_documents(as_list=True):
            extractor.load_document(input=document.text[:10000],
                                    language=lan,
                                    normalization="lemmatization"
                                    )
            # 3. select {1-3}-grams not containing punctuation marks as candidates.
            # must link spacy languages to language code
            extractor.candidate_selection(n=3, stoplist=stop_list)

            # pke.compute_document_frequency(input_dir='/path/to/collection/of/documents/',
            #                                output_file='output.tsv.gz',
            #                                extension='xml',
            #                                language='en',
            #                                normalization="lemmatization",
            #                                stoplist=stop_list)
            #
            # # 4. weight the candidates using a `tf` x `idf`
            # df = pke.load_document_frequency_file(input_file='output.tsv.gz')
            #
            # extractor.candidate_weighting(df=df)
            extractor.candidate_weighting()
            # 5. get the 10-highest scored candidates as keyphrases
            keyphrases = extractor.get_n_best(n=number_keywords)

            corpus.assign_keywords(keywords={document.doc_id: keyphrases}, keyword_type=KeywordType.TFIDF_PKE)

    @classmethod
    def yake_pke(cls, corpus: Corpus):
        number_keywords = 10
        # 1. create a YAKE extractor.
        extractor = pke.unsupervised.YAKE()

        if corpus.language == Language.DE:
            lan = "de"
            stop_list = stopwords.words('german')
        else:
            lan = "en"
            stop_list = stopwords.words('english')

        # 2. load the content of the document.
        for document in corpus.get_documents(as_list=True):
            extractor.load_document(input=document.text[:10000],
                                    language=lan,
                                    normalization="lemmatization"
                                    )

            # 3. select {1-3}-grams not containing punctuation marks and not
            #    beginning/ending with a stopword as candidates.
            extractor.candidate_selection(n=3, stoplist=stop_list)

            # 4. weight the candidates using YAKE weighting scheme, a window (in
            #    words) for computing left/right contexts can be specified.
            window = 2
            extractor.candidate_weighting(window=window,
                                          stoplist=stop_list,
                                          use_stems=True)

            # 5. get the 10-highest scored candidates as keyphrases.
            #    redundant keyphrases are removed from the output using levenshtein
            #    distance and a threshold.
            threshold = 0.8
            keyphrases = extractor.get_n_best(n=number_keywords, threshold=threshold)

            corpus.assign_keywords(keywords={document.doc_id: keyphrases}, keyword_type=KeywordType.YAKE_PKE)

    @classmethod
    def text_rank_pke(cls, corpus: Corpus):
        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}
        number_keywords = 10
        # 1. create a TextRank extractor.
        extractor = pke.unsupervised.TextRank()

        if corpus.language == Language.DE:
            lan = "de"
        else:
            lan = "en"

        # 2. load the content of the document.
        for document in corpus.get_documents(as_list=True):

            extractor.load_document(input=document.text[:10000],
                                    language=lan,
                                    normalization="lemmatization"
                                    )

            # 3. build the graph representation of the document and rank the words.
            #    Keyphrase candidates are composed from the 33-percent
            #    highest-ranked words.
            extractor.candidate_weighting(window=2,
                                          pos=pos,
                                          top_percent=0.33)

            # 4. get the 10-highest scored candidates as keyphrases
            keyphrases = extractor.get_n_best(n=number_keywords)

            corpus.assign_keywords(keywords={document.doc_id: keyphrases}, keyword_type=KeywordType.TEXT_RANK_PKE)

    @classmethod
    def single_rank_pke(cls, corpus: Corpus):
        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}
        number_keywords = 10
        # 1. create a SingleRank extractor.
        extractor = pke.unsupervised.SingleRank()

        if corpus.language == Language.DE:
            lan = "de"
        else:
            lan = "en"

        # 2. load the content of the document.
        for document in corpus.get_documents(as_list=True):

            extractor.load_document(input=document.text[:10000],
                                    language=lan,
                                    normalization="lemmatization"
                                    )

            # 3. select the longest sequences of nouns and adjectives as candidates.
            extractor.candidate_selection(pos=pos)

            # 4. weight the candidates using the sum of their word's scores that are
            #    computed using random walk. In the graph, nodes are words of
            #    certain part-of-speech (nouns and adjectives) that are connected if
            #    they occur in a window of 10 words.
            extractor.candidate_weighting(window=10,
                                          pos=pos)

            # 5. get the 10-highest scored candidates as keyphrases
            # 5. get the 10-highest scored candidates as keyphrases
            keyphrases = extractor.get_n_best(n=number_keywords)
            corpus.assign_keywords(keywords={document.doc_id: keyphrases}, keyword_type=KeywordType.SINGLE_RANK_PKE)

    @classmethod
    def topic_rank_pke(cls, corpus: Corpus):
        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}

        number_keywords = 10

        # 1. create a TopicRank extractor.
        extractor = pke.unsupervised.TopicRank()

        if corpus.language == Language.DE:
            lan = "de"
            stop_list = stopwords.words('german')
        else:
            lan = "en"
            stop_list = stopwords.words('english')

        # 2. load the content of the document.
        for document in corpus.get_documents(as_list=True):

            stop_list += list(string.punctuation)
            stop_list += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

            extractor.load_document(input=document.text[:10000],
                                    language=lan,
                                    normalization="lemmatization"
                                    )

            extractor.candidate_selection(pos=pos, stoplist=stop_list)

            # 4. build topics by grouping candidates with HAC (average linkage,
            #    threshold of 1/4 of shared stems). Weight the topics using random
            #    walk, and select the first occuring candidate from each topic.
            extractor.candidate_weighting(threshold=0.74, method='average')

            # 5. get the 10-highest scored candidates as keyphrases
            keyphrases = extractor.get_n_best(n=number_keywords)
            corpus.assign_keywords(keywords={document.doc_id: keyphrases}, keyword_type=KeywordType.TOPIC_RANK_PKE)

    @classmethod
    def topical_page_rank_pke(cls, corpus: Corpus):
        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}
        # define the grammar for selecting the keyphrase candidates
        grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

        number_keywords = 10

        # 1. create a TopicalPageRank extractor.
        extractor = pke.unsupervised.TopicalPageRank()

        if corpus.language == Language.DE:
            lan = "de"
        else:
            lan = "en"

        # 2. load the content of the document.
        for document in corpus.get_documents(as_list=True):

            extractor.load_document(input=document.text[:10000],
                                    language=lan,
                                    normalization="lemmatization"
                                    )

            # 3. select the noun phrases as keyphrase candidates.
            extractor.candidate_selection(grammar=grammar)

            # 4. weight the keyphrase candidates using Single Topical PageRank.
            #    Builds a word-graph in which edges connecting two words occurring
            #    in a window are weighted by co-occurrence counts.
            extractor.candidate_weighting(window=10,
                                          pos=pos,
                                          lda_model='path/to/lda_model') # todo: find model

            # 5. get the 10-highest scored candidates as keyphrases
            # 5. get the 10-highest scored candidates as keyphrases
            keyphrases = extractor.get_n_best(n=number_keywords)
            corpus.assign_keywords(keywords={document.doc_id: keyphrases}, keyword_type=KeywordType.TOPICAL_PAGE_RANK_PKE)

    @classmethod
    def position_rank_pke(cls, corpus: Corpus) -> Dict[str, List[str]]:
        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}
        # define the grammar for selecting the keyphrase candidates
        grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

        number_keywords = 10

        # 1. create a PositionRank extractor.
        extractor = pke.unsupervised.PositionRank()

        if corpus.language == Language.DE:
            lan = "de"
        else:
            lan = "en"

        # 2. load the content of the document.
        for document in corpus.get_documents(as_list=True):

            extractor.load_document(input=document.text[:10000],
                                    language=lan,
                                    normalization="lemmatization"
                                    )

            # 3. select the noun phrases up to 3 words as keyphrase candidates.
            extractor.candidate_selection(grammar=grammar,
                                          maximum_word_number=3)

            # 4. weight the candidates using the sum of their word's scores that are
            #    computed using random walk biaised with the position of the words
            #    in the document. In the graph, nodes are words (nouns and
            #    adjectives only) that are connected if they occur in a window of
            #    10 words.
            extractor.candidate_weighting(window=10,
                                          pos=pos)

            # 5. get the 10-highest scored candidates as keyphrases
            # 5. get the 10-highest scored candidates as keyphrases
            keyphrases = extractor.get_n_best(n=number_keywords)
            corpus.assign_keywords(keywords={document.doc_id: keyphrases}, keyword_type=KeywordType.POSITION_RANK_PKE)

    @classmethod
    def multipartite_rank_pke(cls, corpus: Corpus):
        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}

        # 1. create a MultipartiteRank extractor.
        extractor = pke.unsupervised.MultipartiteRank()

        if corpus.language == "German":
            lan = "de"
            stop_list = stopwords.words('german')
        else:
            lan = "en"
            stop_list = stopwords.words('english')

        # 2. load the content of the document.
        key_words_of_documents = {}
        number_keywords = 10
        for document in corpus.get_documents(as_list=True):

            stop_list += list(string.punctuation)
            stop_list += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

            extractor.load_document(input=document.text[:10000],
                                    language=lan,
                                    normalization="lemmatization"
                                    )

            extractor.candidate_selection(pos=pos, stoplist=stop_list)

            # 4. build the Multipartite graph and rank candidates using random walk,
            #    alpha controls the weight adjustment mechanism, see TopicRank for
            #    threshold/method parameters.
            extractor.candidate_weighting(alpha=1.1,
                                          threshold=0.74,
                                          method='average')

            # 5. get the 10-highest scored candidates as keyphrases
            keyphrases = extractor.get_n_best(n=number_keywords)
            corpus.assign_keywords(keywords={document.doc_id: keyphrases}, keyword_type=KeywordType.MULTIPARTITE_RANK_PKE)

    @classmethod
    def rake(cls, corpus: Union[Corpus, Document]):
        # additional parameters:
        # ranking_metric
        # punctuations

        if isinstance(corpus, Corpus):
            corpus = corpus
            if corpus.language == Language.DE:
                language = 'german'
            else:
                language = 'english'
            logging.info(f"{language} is used")
            r = Rake(ranking_metric=None,
                     stopwords=stopwords.words(language),
                     language=language,
                     min_length=cls.min_nrgam,
                     max_length=cls.max_ngram)

            keywords = {}
            for document in tqdm(corpus.get_documents(as_list=True), desc="Calculating RAKE"):
                r.extract_keywords_from_text(document.text)
                keywords[document.doc_id] = r.get_ranked_phrases()
            corpus.assign_keywords(keywords=keywords, keyword_type=KeywordType.RAKE)

        else:
            results = {}

            document: Document = corpus
            if document.language == Language.DE:
                language = 'german'
            else:
                language = 'english'

            logging.info(f"{language} is used")
            r = Rake(ranking_metric=None,
                     stopwords=stopwords.words(language),
                     language=language,
                     min_length=cls.min_nrgam,
                     max_length=cls.max_ngram)
            r.extract_keywords_from_text(document.text)
            document.keywords = r.get_ranked_phrases()
            results[document.doc_id] = document.keywords

            return results

    @classmethod
    def key_word_count(cls, keywords: Dict[str, List[str]], top_k=100):
        flattened_keywords = [word for document, document_keywords in keywords.items() for word in document_keywords]
        c = Counter(flattened_keywords)
        if top_k is None:
            return c
        return c.most_common(top_k)

    @staticmethod
    def get_top_k_keywords(year_wise_keywords, top_k=100):
        return {year: keyword_list[:top_k] for year, keyword_list in year_wise_keywords.items()}


def main():
    # load configuration parameters from config file
    config = ConfigLoader.get_config()

    # corpus = Corpus(source=config["corpora"]["abstract_corpus"], language=Language.EN)
    # corpus = Corpus(source=config["corpora"]["bundestag_corpus"], language=Language.DE)
    corpus = Corpus(source=config["corpora"]["sustainability_corpus"], language=Language.EN)

    # print(len(corpus))
    # test = DocumentsFilter.filter(corpus, has_tags=['test'])
    # print(set([x.tags for x in test]))
    # print(len(test))
    #
    # exit(0)

    corpus = corpus.get_n_documents_as_corpus(n=100)

    # build yearwise pseudo documents

    pseudo_corpus = corpus.year_wise_pseudo_documents()
    # extract keywords
    KeyPhraseExtractor.tfidf_skl(corpus=pseudo_corpus)
    print([d.keywords for d in pseudo_corpus.get_documents()])

    KeyPhraseExtractor.rake(corpus=corpus)
    print([d.keywords for d in corpus.get_documents()])
    # key_words_post_group = Document.group_keywords_year_wise(corpus)
    # key_words_pre_group = Document.transform_pseudo_docs_keywords_to_dict(KeyPhraseExtractor.rake(documents=pseudo_corpus))

    # print(KeyPhraseExtractor.get_top_k_keywords(key_words_post_group, 10))
    # print(KeyPhraseExtractor.get_top_k_keywords(key_words_pre_group, 10))
    # format: {year->list fo keywords}

    kwt = KeywordTranslator(cache_file=config["translator"]["cache_file"])

    counter = 0
    for doc in corpus.get_documents():
        for keyword in doc.keywords:
            if counter > 100:
                break
            kwt.translate(keyword)
            print(keyword)
            counter += 1
        break



    print('extracting keywords with rake ...')
    rake_keywords = KeyPhraseExtractor.rake(corpus=corpus.get_documents()[0])
    rake_keywords_keys = list(rake_keywords.keys())
    print('rake keywords dict keys:', rake_keywords_keys)

    kwt = KeywordTranslator()
    list_of_keywords = []

    for k in rake_keywords[rake_keywords_keys[0]]:
        kw = Keyword(german_translation=k, type=KeywordType.RAKE)
        kwt.translate(kw)
        list_of_keywords.append(kw)
        print('{} \t {} \t\t\t {}'.format(kw.source_language, kw.english_translation, kw.german_translation))

    # print('extracting keywords with tf-idf ...')
    # tfidf_keywords = KeyPhraseExtractor.tfidf_skl(documents=corpus)
    # print(tfidf_keywords, '\n')

    # print(KeyPhraseExtractor.key_word_count(KeyPhraseExtractor.rake(documents=corpus)))

    # aggregate documents / keywords

    # translate and match key words

    # visualize matching


if __name__ == '__main__':
    main()
