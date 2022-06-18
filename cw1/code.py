from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import xml.etree.ElementTree as ET
import functools, operator, pickle, re, sys, math

BOOLEAN_OPERATORS = (' AND NOT ', ' OR NOT ', ' AND ', ' OR ')

def tf(t, d, index) -> int:
    # term frequency, which is the frequency of a term in a specified document
    return len(index[t][d])

def df(t, index) -> int:
    # document frequency, which is the number of documents the term showed in
    return len(index[t])

def tokenise(text) -> list[str]:
    #lowercases the text and splits it into tokens
    tokens = re.findall(r"\w+", text.lower())
    #tokens = re.findall(r"\w+'\w+|\w+", text.lower())
    return tokens

def load_stopwords(stopwords_file) -> set:
    #loads stopwords file given its name
    with open(stopwords_file) as f:
        stopwords = set(line.strip() for line in f)
    return stopwords


def stem(word) -> str:
    #stems a given token using Porter stemmer
    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    return porter.stem(str(word))

def preprocess(text, stopwords) -> list[str]:
    #runs a chain of preproccessing commands (tokenising, stopwords filtering and stemming all together)
    tokens = tokenise(text)
    tokens = list(map(stem, filter(lambda x: x not in stopwords, tokens)))
    return tokens



def parse_xml(stopwords, filename) -> tuple[dict[str, dict[str,set]], int]:
    #produces an index out of xml file
    index :dict[str, dict] = {}
    tree = ET.parse(filename)
    root = tree.getroot()

    for doc in root:
        doc_id = doc.find('DOCNO').text
        text =  doc.find('HEADLINE').text + doc.find('TEXT').text
        tokens = tokenise(text)
        idx = 1
        for t in tokens:
            if t not in stopwords:
                t = stem(t)
                docs = index[t] if t in index else {}    
                if doc_id not in docs:
                    docs[doc_id] = {idx}
                else:
                    docs[doc_id].add(idx)
                index[t] = docs
                idx = idx + 1
    return index, len(root)          

def save_index_to_txt(index, filename):
    #saves index to txt file
    with open(filename, "w") as f:
        for word in index:
            f.write(f'{word}:{df(word, index)}\n')
            for doc_id in index[word]:
                f.write(f'\t{doc_id}: {",".join(str(x) for x in sorted(index[word][doc_id]))}\n')
            f.write('\n')

def save_index_to_pickle(index: dict, filename: str):
    #saves inde to pickle binary file
    with open(filename, "wb") as f:
        pickle.dump(index,f)

def load_index_from_pickle(filename) -> dict[str, dict[str,set]]:
    #loads index from pickle binary file
    with open(filename, "rb") as f:
        return pickle.load(f)

def bool_query_process(query: str, stopwords, index):
    #preprocesses a boolean seach query according to its type and sends to its appropriate function
    for op in BOOLEAN_OPERATORS:
        if query.find(op) != -1:
            queries = query.split(op, 1)
            result1 = bool_query_process(queries[0], stopwords, index)
            result2 = bool_query_process(queries[1], stopwords, index)
            if op ==' AND ': return result1 & result2
            if op ==' OR ': return result1 | result2
            if op ==' AND NOT ': return result1 - result2
            if op ==' OR NOT ': return result1

    if query.startswith('"') and query.endswith('"'):
        tokens = preprocess(query, stopwords)
        return phrase_search(tokens, index)

    elif query.startswith('#'):
        proximity = int(query[1:query.find('(')])
        tokens = preprocess(query[query.find('('): query.find(')')], stopwords)
        return proximity_search(tokens, proximity, index)

    else:
        token = tokenise(query)[0]
        if token in stopwords: return {}
        token = stem(token)
        return boolean_search(token, index)

def parse_queries(filename):
    #parses queries giving their file
    queries = []
    with open(filename) as f:
        for line in f:
            queries.append((line[0:line.find(" ")], line[line.find(" ")+1:].replace("\n", "")))
    return queries

def intersect_docs(tokens, index):
    docs = list(map(lambda x: set() if x not in index else index[x].keys(), tokens))
    return functools.reduce(operator.and_, docs, docs[0])

def phrase_search(tokens, index):
    #performs a phrase search given proccessed tokens
    search_results = set()

    intersection = intersect_docs(tokens, index)
    for i in intersection:
        for n,t in enumerate(tokens):
            if t != tokens[0]:
                next_set = set(map(lambda x: x-n, index[t][i]))
                if not(first_set & next_set):
                    break
                if t == tokens[-1]:
                    search_results.add(i)
            else:
                first_set  = index[t][i]        
    return search_results

def proximity_search(tokens, proximity, index):
    #performs a proximity search giving proximity and processed tokens
    search_results = set()
    intersection = intersect_docs(tokens, index)

    for doc_id in intersection:
        found = False
        for i in index[tokens[0]][doc_id]:
            if found: break
            for j in index[tokens[1]][doc_id]:
                if abs(j - i) <= proximity:
                    search_results.add(doc_id)
                    found = True
                    break
    return search_results

def boolean_search(query: str, index: dict):
    #performs a boolean search for a single word
    return index[query].keys() if query in index else set()

def ranked_search(query, stopwords, n, index):
    #performs a rank search on a given query, returns 
    results = {}
    tokens = preprocess(query, stopwords)
    w = lambda t, d: (1 + math.log10(tf(t, d, index))) * math.log10(n/df(t, index))
    for t in tokens:
        if t in index: docs = index[t]  
        else: continue
        for d in docs:
            score = w(t,d)
            results[d] = results[d] + score if d in results else score
    results = sorted({k: round(v, 4) for k, v in results.items()}.items(), reverse=True, key=lambda x:x[1])
    return results if len(results)<=150 else results[:150]


def results_to_txt(formated_results, filename, ranked: bool):
    #writes formatted search results to txt file
    
    with open(filename, "w") as f:
        if ranked:
            for r in formated_results:
                f.write(f'{r[0]},{r[1][0]},{r[1][1]}\n')
        else:
            for r in formated_results:
                f.write(f'{r[0]},{r[1]}\n')

    


if __name__ == "__main__":

    xml_file = "trec.5000.xml"
    stopwords_file = "englishST.txt"
    queries_file = "queries.boolean.txt"
    ranked_queries_file = "queries.ranked.txt"

    stopwords = load_stopwords(stopwords_file)

    index, n = parse_xml(stopwords, xml_file)    
    save_index_to_txt(index, 'index.txt')

    #save_index_to_pickle(index, "index.pkl")
    #index2 = load_index_from_pickle("index.pkl")

    #parse boolean queries
    queries = parse_queries(queries_file)
    formated_results = []
    #query
    for q in queries:
        results = bool_query_process(q[1], stopwords, index)
        for r in results:
            formated_results.append((q[0], r))
    results_to_txt(formated_results, 'results.boolean.txt', ranked=False)       

    #parse ranked queries
    r_queries = parse_queries(ranked_queries_file)
    formated_r_results = []
    #ranked query
    for rq in r_queries:
        ranked_results = ranked_search(rq[1], stopwords, n, index)
        for rr in ranked_results:
            formated_r_results.append((rq[0], rr))

    results_to_txt(formated_r_results, 'results.ranked.txt', ranked=True)

    