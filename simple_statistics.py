from collections import defaultdict, OrderedDict
from typing import Callable
from utils import ConfigLoader, Language, Corpus


def count_non_years(corpus: Corpus):
    without_year = [d for d in corpus.get_documents() if d.date is None]
    print(len([d.date for d in corpus.get_documents() if d.date and len(str(d.date)) != 4]))
    with_year = [d for d in corpus.get_documents() if d.date]
    print(f'{len(without_year)} / {len(with_year)}')


def token_number(corpus: Corpus):
    return corpus.token_number()


def document_number(corpus: Corpus):
    return len(corpus.documents)


def yearwise_documents(corpus: Corpus, aggregation_func: Callable = len, printing: bool = False, as_dict: bool = False):
    year_bins = defaultdict(list)

    for doc in corpus.get_documents():
        year_bins[doc.date].append(doc)

    result = {year: aggregation_func(Corpus(source=docs, language=corpus.language, name=f'{corpus.name}_yearwise'))
              for year, docs in year_bins.items() if year is not None}
    result = OrderedDict(sorted(result.items()))

    if as_dict:
        return result

    years = []
    counts = []
    for year, count in result.items():
        years.append(year)
        counts.append(count)
        if printing:
            print(f'{year}: {count}')

    # print(years)
    # print(counts)
    return years, counts


def main():
    config = ConfigLoader.get_config()

    corpora = [Corpus(source=config["corpora"]["abstract_corpus"],
                      language=Language.EN, name="abstract_corpus"),
               Corpus(source=config["corpora"]["bundestag_corpus"],
                      language=Language.DE, name="bundestag_corpus"),
               Corpus(source=config["corpora"]["sustainability_corpus"],
                      language=Language.EN, name="sustainability_corpus"),
               Corpus(source=config["corpora"]["state_of_the_union_corpus"],
                      language=Language.EN, name="state_of_the_union_corpus")]

    # count_non_years(corpora[0])
    # count_non_years(corpora[1])
    # count_non_years(corpora[2])

    # Results: non date vs useable date
    # abstract: 54135 / 261215, 1387 don't have a year as date but a string
    # bundestag: 0 / 877973
    # sustainability 3 / 221034

    print(document_number(corpora[0]))
    print(document_number(corpora[1]))
    print(document_number(corpora[2]))
    print(document_number(corpora[3]))

    print(token_number(corpora[0]))
    print(token_number(corpora[1]))
    print(token_number(corpora[2]))
    print(token_number(corpora[3]))

    # Results: token number
    # abstract: 59314582
    # bundestag: 226300348
    # sustainability: 52878146

    yearwise_documents(corpora[0], aggregation_func=len)
    # [1900, 1904, 1951, 1961, 1965, 1972, 1974, 1975, 1976, 1978, 1979, 1980, 1981, 1983, 1984, 1985, 1986, 1987, 1988,
    #  1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
    #  2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    # [1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 3, 2, 1, 4, 14, 28, 47, 44, 124, 714, 962, 1080, 1143, 1513, 2104, 2341,
    # 2554, 2862, 2947, 3470, 3617, 4230, 4495, 4827, 5655, 6948, 8331, 10287, 11750, 14345, 16149, 19308, 20899,
    # 23429, 26201, 28937, 29835]
    yearwise_documents(corpora[0], aggregation_func=token_number)
    # [1900, 1904, 1951, 1961, 1965, 1972, 1974, 1975, 1976, 1978, 1979, 1980, 1981, 1983, 1984, 1985, 1986, 1987, 1988,
    #  1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
    #  2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    # [237, 289, 26, 196, 299, 4, 2, 302, 13, 35, 163, 2, 513, 13, 3, 354, 2763, 5930, 10297, 9573, 20802, 124895,
    # 172925, 202836, 227647, 303919, 435539, 496060, 558721, 628000, 653111, 770258, 822043, 937258, 1009178, 1078762,
    # 1283970, 1593002, 1880724, 2268271, 2621783, 3192629, 3664511, 4406424, 4775594, 5367972, 6024271,
    # 6682090, 7080373]

    yearwise_documents(corpora[1], aggregation_func=len)
    # [1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967,
    #  1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986,
    #  1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005,
    #  2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    # [1540, 7359, 7846, 7492, 6252, 5534, 5794, 7532, 6738, 4469, 4446, 7027, 5950, 7756, 8704, 12078, 13355, 14542,
    #  15855, 15673, 14876, 15917, 16901, 8760, 15082, 16343, 17110, 11914, 14095, 15597, 14811, 8937, 14207, 14647,
    #  9904, 16009, 19397, 16843, 10560, 16032, 16220, 11704, 14972, 14102, 17113, 11485, 16825, 17482, 13614, 9905,
    #  15310, 14208, 14124, 10926, 12884, 14305, 7757, 14210, 13508, 14408, 10609, 16643, 17751, 16497, 11335, 15374,
    #  14794, 13705, 5829, 17021, 9469]
    yearwise_documents(corpora[1], aggregation_func=token_number)
    # [1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967,
    #  1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986,
    #  1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005,
    #  2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    # [612509, 2854944, 3061777, 3034065, 2113852, 2406060, 2380625, 2660021, 2460495, 2114953, 1715064, 2049805,
    # 1614656, 1634229, 1867580, 2135204, 2055406, 2452521, 2553521, 2575640, 2464189, 2675640, 2836025, 1644761,
    # 2665313, 3244912, 3004963, 2657335, 2751084, 2919374, 3366152, 2159773, 2722208, 3171091, 2280604, 3443955,
    # 3855233, 3566063, 2569335, 3565324, 4173720, 3067311, 3987509, 3832524, 4291976, 3145478, 4291797, 4338335,
    # 3925125, 3094547, 4464993, 4373147, 4392056, 3738766, 3946187, 4129635, 2350304, 4330315, 3983980, 4532271,
    # 3752798, 5167090, 5442241, 5468729, 3942007, 4846052, 4613129, 4046021, 1607377, 4583019, 2525648]

    yearwise_documents(corpora[2], aggregation_func=len)
    # [1986, 1987, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
    #  2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    # [1, 1, 39, 297, 476, 572, 749, 1017, 1117, 1327, 1479, 1673, 1953, 2072, 2246, 2762, 2971, 3593, 4149, 5313, 6234,
    #  7880, 9095, 10858, 12484, 15035, 17163, 20084, 23485, 29233, 35676]
    yearwise_documents(corpora[2], aggregation_func=token_number)
    # [1986, 1987, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
    #  2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    # [96, 217, 8748, 59556, 98237, 119917, 153011, 212506, 237082, 291767, 329358, 375165, 444080, 476757, 529224,
    #  657520, 693466, 847938, 985064, 1255443, 1473326, 1856967, 2120475, 2548691, 2924106, 3559252, 4097080, 4829304,
    #  5716151, 7148684, 8828958]


if __name__ == '__main__':
    main()
