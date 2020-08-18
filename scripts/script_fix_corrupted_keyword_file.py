import json
from collections import defaultdict

path = "data/sustainability_corpus_yake_keywords_yearwise.json"
with open(path, encoding='utf-8') as f:
    data = json.load(f)

res = defaultdict(list)
for doc_id, keywords in data.items():
    for keyword in keywords:
        if isinstance(keyword["german_translation"], list):
            keyword["german_translation"] = keyword["german_translation"][0]

        if isinstance(keyword["english_translation"], list):
            keyword["english_translation"] = keyword["english_translation"][0]

        res[doc_id].append(keyword)


with open(path, "w", encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=True)
