import math
import json
import operator
from collections import Counter

def load(path):
    f = open(path)
    return json.load(f)

def init_vectors(sample):
    docs = set()
    lenght = {}
    scores = {}
    
    for itens in sample.items():
        for item in itens[1]:
            docs.add(item[0])
            if item[0] not in lenght:
                lenght[item[0]] = 0
            lenght[item[0]] += item[1]
            scores[item[0]] = 0
            
    return (docs, lenght, scores)

def rank(path, query, idf=False):
    sample = load(path)
    
    docs, lenght, scores = init_vectors(sample)

    query_tf = Counter(query.split())
    query_terms = set(query_tf.keys())
    vocabulary = set(sample.keys())
    terms = query_terms.intersection(vocabulary)

    for term in terms: # Term-at-a-time
        wq = query_tf[term]
        widf = math.log10(len(docs)/len(sample[term]))
        for doc in sample[term]:        
            if idf:
                scores[doc[0]] += doc[1] * wq * widf
            else:
                scores[doc[0]] += doc[1] * wq

    return sorted(scores.items(), key=operator.itemgetter(1), reverse=True)