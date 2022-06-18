import csv
import collections
import re
import math
from typing import Text
from nltk.stem import PorterStemmer
import string
import gensim
from gensim.corpora.dictionary import Dictionary 
from collections import defaultdict, Counter


ps = PorterStemmer()

with open('englishST.txt' , 'r') as stopfile:
    read1 = stopfile.read()
stopwords = read1.split('\n')
chars = str.maketrans('', '', string.punctuation)

processed = defaultdict(list)
word_doc_ctr = {}
def preprocess(line):
    data = re.findall('\w+' , line.lower())
    tokens = []
    for word in data:
        tokens.append(word.translate(chars))   
    tokens = [ps.stem(word) for word in tokens if word not in stopwords and word != '']
    return tokens

with open('train_and_dev.tsv', 'r') as task2:
    doc = csv.reader(task2 , delimiter="\t", quoting=csv.QUOTE_NONE)
    for line in doc:
        tokens = preprocess(line[1])
        processed[line[0]].append(tokens)
        if line[0] not in word_doc_ctr.keys():
            word_doc_ctr[line[0]] = defaultdict(int)
        for token in set(tokens):
            # if token not in word_doc_ctr[line[0]]:
            #     word_doc_ctr[line[0]][token] = 1
            # else:
            word_doc_ctr[line[0]][token] += 1




############################################################################################
n = sum(len(c) for c in processed.values())
def chi_square(list_of_corpus_counters):
    res = {}
    print(f'n = {n}')
    for corpus_idx, corpus in list_of_corpus_counters.items():
        res[corpus_idx] = {}
        for term, count in corpus.items():
            if count == 0:
                print(term,'zero')
                continue
            n11 = count
            n10 = sum([c[term] for i, c in list_of_corpus_counters.items() if i != corpus_idx])
            n01 = len(processed[corpus_idx]) - n11
            n00 = sum([len(c) for i, c in processed.items() if i != corpus_idx]) - n10

            # if term == 'lord':
            #     print(corpus_idx)
            #     print(n00, n01, n10, n11)

            na1 = n01 + n11
            n1a = n11 + n10
            na0 = n10 + n00
            n0a = n01 + n00

            # if term == 'lord':
            #     print(na1, n1a, na0, n0a)
            if n1a >= 10:
                if n10 == 0:
                    m1 = 0
                else:
                    m1 = (n10/n * math.log2((n*n10)/(n1a * na0)))
            mutual_info = (n11/n * math.log2((n*n11)/(n1a * na1))) + (n01/n * math.log2((n*n01)/(n0a * na1))) + m1 + (n00/n * math.log2((n*n00)/(n0a * na0)))
            if term == 'lord':
                print(corpus_idx)
                print(mutual_info)

            x_s = ((n11 + n10 + n01 + n00) * (n11*n00 - n10*n01)**2) / ((n11+n01) * (n11 + n10) * (n10 + n00) * (n01 + n00))

            # res[corpus_idx][term] = (mutual_info, x_s)
            res[corpus_idx][term] = x_s
    return res

p1 = chi_square(word_doc_ctr)
for id, corpus in p1.items():
    m = sorted(corpus.items(), key=lambda item: item[1], reverse=True)
    print(id)
    print(m[:10])
        

# print(len(p1))

############################################################################################

def findt(l):
    results = []
    for t in range(20):
        sum= 0
        for d in l:
            sum += d[t][1]
        results.append(sum / len(l))
    return results

processed_texts = [doc for corpus in processed.values() for doc in corpus ]
#processed_texts = sum([v for v in processed.values()], [])
print(len(processed_texts),'here')
lens = [len(v) for v in processed.values()]
print(lens)

common_dictionary = Dictionary(processed_texts)
c=[common_dictionary.doc2bow(text) for text in processed_texts]
print(len(c))
print(c[0])
print(c[1])
print(c[2])
lda=gensim.models.ldamodel.LdaModel(corpus=c,id2word=common_dictionary,num_topics=20 , random_state=1)
l1=[lda.get_document_topics(bow = i , minimum_probability=0) for i in c[:lens[0]]]
l2=[lda.get_document_topics(bow = i , minimum_probability=0) for i in c[lens[0]:lens[1]]]
l3=[lda.get_document_topics(bow = i , minimum_probability=0) for i in c[lens[1]:]]
print(len(l1))
print(l1[0])
print(findt(l1))
topics = lda.print_topics()
#print(topics)
