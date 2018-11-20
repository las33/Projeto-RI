import math
import json
import operator
from collections import Counter

class QueryProcessing():    
    def load(self, path): # carregar o indice invertido no formato json
        f = open(path)
        return json.load(f)

    def init_vectors(self, spl): # inicia as tres estruturas base, recebe o indice invertido como paramentro (sample)
        docs = set() # lista de documentos reconhecidos pelo indice invertido
        length = {} # a quantidade de vezes que um documento foi referenciado
        scores = {} # estrutura base de score, cada chave é um documento que tem valor 0 (zero)

        for key in spl.keys(): # processa cada token do indice invertido
            dc = spl[key][1] # lista de documentos onde o token ocorre
            for item in dc.keys():
                docs.add(item) # adiciona o documento
                if item not in length:
                    length[item] = 0
                length[item] += dc[item] # contabiliza um para esse documento
                scores[item] = 0 # inicializa o valor 0 (zero) para esse documento

        return (docs, length, scores)
    
    def __init__(self, index_path):
        self.sample = self.load(index_path)
        self.docs, self.length, self.scores = self.init_vectors(self.sample)

    def rank(self, query, attr="", idf=False): # ranqueamento dos documentos para uma query
        query_terms = set([attr + "." + q if attr != "" else q for q in query.split()]) # processa de acordo com o attr informado (para consultas estruturadas)
        query_tf = Counter(query_terms) # tf da consulta, contagem de termos
        vocabulary = set(self.sample.keys()) # vocabulario do indice invertido (lista de tokens)
        terms = query_terms.intersection(vocabulary) # filtro de tokens que existem no vocabulario
        docs_score = dict(self.scores) # uma copia da estrutura de scores para os documentos que serao ranqueados

        for term in terms: # metodo Term-at-a-time
            term_docs = self.sample[term][1] # lista de documentos que o termo ocorre
            wq = query_tf[term] # peso da query (tf puro)
            if idf: # caso considere o idf
                widf = math.log10(len(self.docs)/len(term_docs))
            for doc in term_docs: 
                if idf:
                    docs_score[doc] += term_docs[doc] * wq * widf
                else:
                    docs_score[doc] += term_docs[doc] * wq

        for s in self.scores.keys(): # normalizacao
            docs_score[s] = docs_score[s]/self.length[s]

        return sorted(docs_score.items(), key=operator.itemgetter(1), reverse=True) # ordenar pelo maior ranking