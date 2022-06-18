from collections import defaultdict
from os import system
from nltk.stem.porter import PorterStemmer
import csv, math, re, random
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from scipy import stats
from collections import defaultdict
import sklearn
import string
import scipy
from sklearn.metrics import classification_report


class PREPROCESS:
    def tokenise(text) -> list[str]:
    #lowercases the text and splits it into tokens
        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def load_stopwords(stopwords_file) -> set:
        #loads stopwords file given its name
        with open(stopwords_file) as f:
            stopwords = set(line.strip() for line in f)
        return stopwords


    def stem(word) -> str:
        #stems a given token using Porter stemmer
        porter = PorterStemmer()
        return porter.stem(str(word))

    def preprocess(text, stopwords) -> list[str]:
        #runs a chain of preproccessing commands (tokenising, stopwords filtering and stemming all together)
        tokens = PREPROCESS.tokenise(text)
        tokens = list(map(PREPROCESS.stem, filter(lambda x: x not in stopwords, tokens)))
        return tokens

    def process_tsv(tsv_file, stopwords):
        docs_tokens = defaultdict(list)
        corpuses_tokens_counts = defaultdict(lambda: defaultdict(int))
        with open(tsv_file, 'r') as data:
            csv_reader = csv.reader(data , delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in csv_reader:
                row_tokens = PREPROCESS.preprocess(row[1], stopwords)
                docs_tokens[row[0]].append(row_tokens)
                for t in set(row_tokens):
                    corpuses_tokens_counts[row[0]][t] += 1
        #number of all documents n
        n = sum(len(c) for c in docs_tokens.values())
        return docs_tokens, corpuses_tokens_counts, n

class ANALYSIS:
    def chi_squared(n11, n10, n01, n00, n_1, n1_, n_0, n0_):
        return ((n1_ + n0_)*(n11*n00 - n10*n01)**2) / (n_1*n1_*n_0*n0_)
    
    def mutual_info(n11, n10, n01, n00, n_1, n1_, n_0, n0_, n):    
        t1 = 0 if n11 == 0 else (n11/n * math.log2((n*n11)/(n1_ * n_1)))
        t2 = 0 if n01 == 0 else (n01/n * math.log2((n*n01)/(n0_ * n_1)))
        t3 = 0 if n10 == 0 else (n10/n * math.log2((n*n10)/(n1_ * n_0)))
        t4 = 0 if n00 == 0 else (n00/n * math.log2((n*n00)/(n0_ * n_0)))
        return t1 + t2 + t3 + t4

    def analyse_corpuses(docs_tokens, corpuses_tokens_counts, n):
        mutual_info, chi_squared = {}, {}
        for corpus, tokens in corpuses_tokens_counts.items():
            mutual_info[corpus] = {}
            chi_squared[corpus] = {}
            for token, count in tokens.items():
                #number of docs that contain token and in corpus
                n11 = count
                #number of docs that contain token but in not corpus
                n10 = sum([tokens[token] for c, tokens in corpuses_tokens_counts.items() if c != corpus])
                #number of docs that don't contain token but in corpus
                n01 = len(docs_tokens[corpus]) - n11
                #number of docs that doesn't contain token and not in corpus
                n00 = sum([len(docs) for c, docs in docs_tokens.items() if c != corpus]) - n10
                #number of docs in corpus
                n_1 = n01 + n11
                #number of docs that contains token
                n1_ = n11 + n10
                #number of docs that are not in corpus
                n_0 = n10 + n00
                #number of docs that don't contain token
                n0_ = n01 + n00
                
                mutual_info[corpus][token] = ANALYSIS.mutual_info(n11, n10, n01, n00, n_1, n1_, n_0, n0_, n)
                chi_squared[corpus][token] = ANALYSIS.chi_squared(n11, n10, n01, n00, n_1, n1_, n_0, n0_)
        return mutual_info, chi_squared 

    def avg_score_per_topic(doc_topic_probs):
        avg_scores = []
        for topic in range(20):
            sum= 0
            for doc in doc_topic_probs:
                sum += doc[topic][1]
            avg_scores.append(sum / len(doc_topic_probs))
        return avg_scores
    
    def get_docs_topic_probs(lda, corpus):
        return [lda.get_document_topics(bow = doc , minimum_probability=0) for doc in corpus]

    def lda_topic_model(docs_tokens):
        common_texts = [doc for corpus in docs_tokens.values() for doc in corpus]
        corpus_sizes = [len(c) for c in docs_tokens.values()]
        common_dictionary = Dictionary(common_texts)
        common_corpus=[common_dictionary.doc2bow(text) for text in common_texts]
        lda = LdaModel(corpus=common_corpus,id2word=common_dictionary,num_topics=20 , random_state=77)     

        OT_docs_topic_probs= ANALYSIS.get_docs_topic_probs(lda, common_corpus[:corpus_sizes[0]])
        NT_docs_topic_probs= ANALYSIS.get_docs_topic_probs(lda, common_corpus[corpus_sizes[0]:corpus_sizes[0]+corpus_sizes[1]])
        Q_docs_topic_probs= ANALYSIS.get_docs_topic_probs(lda, common_corpus[corpus_sizes[0]+corpus_sizes[1]:])

        OT_avg_topic_score = ANALYSIS.avg_score_per_topic(OT_docs_topic_probs)
        NT_avg_topic_score = ANALYSIS.avg_score_per_topic(NT_docs_topic_probs)
        Q_avg_topic_score = ANALYSIS.avg_score_per_topic(Q_docs_topic_probs)

        OT_sorted_topics = sorted(list(zip(range(20),OT_avg_topic_score)), key= lambda x: x[1], reverse=True)
        NT_sorted_topics = sorted(list(zip(range(20),NT_avg_topic_score)), key= lambda x: x[1], reverse=True)
        Q_sorted_topics = sorted(list(zip(range(20),Q_avg_topic_score)), key= lambda x: x[1], reverse=True)

        for t in OT_sorted_topics[:3]:
            print('OT')
            print(f'Topic: {t[0]} at {t[1]}')
        print(OT_avg_topic_score)
        for t in NT_sorted_topics[:3]:
            print('NT')
            print(f'Topic: {t[0]} at {t[1]}')
        print(NT_avg_topic_score)
        for t in Q_sorted_topics[:3]:
            print('Q')
            print(f'Topic: {t[0]} at {t[1]}')
        print(Q_avg_topic_score)
        topics = lda.print_topics()
        print(topics)

class EVAL:
    def parse_results(results_file):
        results = defaultdict(lambda: defaultdict(list))
        with open(results_file) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader)
            for row in reader:
                results[int(row[0])][int(row[1])].append(row[2:])
        return results
    
    def parse_qrels(qrels_file):
        qrels = defaultdict(dict)
        with open(qrels_file) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader)
            for row in reader:
                qrels[int(row[0])][row[1]]=int(row[2])
        return qrels

    def precision_at(k: int, results: list, qrels: dict):
        tp = len(qrels.keys() & [r[0] for r in results[:k]])
        return tp/k
    
    def recall_at(k: int, results: list, qrels: dict):
        tp = len(qrels.keys() & [r[0] for r in results[:k]])
        return round(tp/len(qrels), 3)
    
    def avg_precision(results: list, qrels: dict):
        precisions =  [EVAL.precision_at(int(r[1]), results, qrels) for r in results if r[0] in qrels]
        return round(sum(precisions)/len(qrels),3) if len(precisions)>0 else 0

    def DCG_at(k: int, results: list, qrels: dict):
        rel1 = qrels[results[0][0]] if results[0][0] in qrels else 0
        DG = lambda x: qrels[x[0]]/math.log2(int(x[1])) if x[0] in qrels else 0
        return rel1 + sum(map(DG, results[1:k]))

    def iDCG_at(k: int, qrels: dict):
        list_qrels = list(qrels.values())
        rel1 = list_qrels[0]
        iDG = lambda x: x[1]/math.log2(x[0]+2)
        return rel1 + sum(map(iDG, enumerate(list_qrels[1:k])))

    def nDCG_at(k: int, results: list, qrels: dict):
        return round(EVAL.DCG_at(k, results, qrels)/EVAL.iDCG_at(k, qrels),3)
    
    def significance_test(eval_results, means):
        #calculates significance test for top two systems for each score
        prints = ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']
        for s in range(6):
            sorted_by_score = sorted([(i+1,x[s]) for i,x in enumerate(means)], reverse=True, key=lambda x: x[1])
            a, b = sorted_by_score[0][0], sorted_by_score[1][0]
            a_val, b_val = sorted_by_score[0][1], sorted_by_score[1][1]
            print(f'top two systems by {prints[s]} are:')
            print(f'systems {a} {b}: {a_val} {b_val}')
            print(eval_results[a][s])
            print(eval_results[b][s])
            print(stats.ttest_ind(eval_results[a][s], eval_results[b][s]))


    def eval_systems(results, qrels):
        with open('ir_eval.csv', 'w') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',')
                    csv_writer.writerow(['system_number', 'query_number', 'P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20'])
                    means = []
                    eval_results = {}
                    for system in results:
                        Ps, Rs, RPs, APs, nDCGs10, nDCGs20 = ([] for i in range(6)) #lists of eval results per system
                        for query in results[system]:

                            p_at_10 = round(EVAL.precision_at(10, results[system][query], qrels[query]),3)
                            r_at_50 = EVAL.recall_at(50, results[system][query], qrels[query])
                            r_precision = round(EVAL.precision_at(len(qrels[query]), results[system][query], qrels[query]),3)
                            ap = EVAL.avg_precision(results[system][query], qrels[query])
                            nDCG10 = EVAL.nDCG_at(10, results[system][query], qrels[query])
                            nDCG20 = EVAL.nDCG_at(20, results[system][query], qrels[query])

                            Ps.append(p_at_10)
                            Rs.append(r_at_50)
                            RPs.append(r_precision)
                            APs.append(ap)
                            nDCGs10.append(nDCG10)
                            nDCGs20.append(nDCG20)

                            csv_writer.writerow([system, query, p_at_10,r_at_50,r_precision,ap,nDCG10,nDCG20])
                        n = len(results[system])
                        eval_results[system] = [Ps, Rs, RPs, APs, nDCGs10, nDCGs20]
                        means.append([round(sum(Ps)/n,3), round(sum(Rs)/n,3), round(sum(RPs)/n,3), round(sum(APs)/n,3), round(sum(nDCGs10)/n,3), round(sum(nDCGs20)/n,3)])
                        csv_writer.writerow([system, 'mean', means[-1][0], means[-1][1], means[-1][2],  means[-1][3], means[-1][4], means[-1][5]])
                    return eval_results, means

class CLASSIFY:
    def split_train_dev(verses, corpuses, train_split, dev_split):        
        X_train, X_dev, y_train, y_dev = sklearn.model_selection.train_test_split(verses, corpuses, train_size=train_split, test_size=dev_split, random_state=77)
        return X_train, X_dev, y_train, y_dev
    
    def word2id(vocab):
        word2id, corpus2id = {}, {}
        for word_id, word in enumerate(vocab):
            word2id[word] = word_id
        corpus2id = {'OT' : 0, 'NT' : 1, 'Quran' : 2}
        return word2id, corpus2id

    def convert_to_bow_matrix(preprocessed_data, word2id):    
        matrix_size = (len(preprocessed_data),len(word2id)+1)
        oov_index = len(word2id)
        X = scipy.sparse.dok_matrix(matrix_size)

        for doc_id,doc in enumerate(preprocessed_data):
            for word in doc:
                X[doc_id,word2id.get(word,oov_index)] += 1
        return X
    
    def convert_to_normalised_bow_matrix(preprocessed_data, word2id):    
        matrix_size = (len(preprocessed_data),len(word2id)+1)
        oov_index = len(word2id)
        X = scipy.sparse.dok_matrix(matrix_size)

        for doc_id,doc in enumerate(preprocessed_data):
            for word in doc:
                X[doc_id,word2id.get(word,oov_index)] += 1/len(doc) # normalised vectorisation
        return X

    def baseline_data_preprocess(data):    
        verses, corpuses, vocab = [], [], set()
        chars_to_remove = re.compile(f'[{string.punctuation}]')
        lines = data.split("\n") 
        for line in lines:
            line = line.strip()
            if line:
                corpus, verse= line.split('\t')
                words = chars_to_remove.sub('',verse).lower().split()#remove lower
                for word in words:
                    vocab.add(word)
                verses.append(words)
                corpuses.append(corpus)        
        return verses, corpuses, vocab

    def improved_data_preprocess(data):
        verses, corpuses, vocab = [], [], set()
        chars_to_remove = re.compile(f'[{string.punctuation}]')
        lines = data.split("\n") 
        for line in lines:
            line = line.strip()
            if line:
                corpus, verse= line.split('\t')
                words = chars_to_remove.sub('',verse).split() #remove lower
                for word in words:
                    vocab.add(word)
                verses.append(words)
                corpuses.append(corpus)        
        return verses, corpuses, vocab


    def prepare_baseline_preprocessed_data(verses, corpuses, vocab):
        X_train, nX_dev, y_train, y_dev = CLASSIFY.split_train_dev(verses, corpuses, 0.7, 0.3)
        word2id, corpuses2id = CLASSIFY.word2id(vocab)

        X_train = CLASSIFY.convert_to_bow_matrix(X_train, word2id)
        X_dev = CLASSIFY.convert_to_bow_matrix(nX_dev, word2id)
        y_train = [corpuses2id[c] for c in y_train]
        y_dev = [corpuses2id[c] for c in y_dev]
        return X_train, X_dev, y_train, y_dev, nX_dev

    def prepare_improved_preprocessed_data(verses, corpuses, vocab):
        X_train, X_dev, y_train, y_dev = CLASSIFY.split_train_dev(verses, corpuses, 0.9, 0.1)
        word2id, corpuses2id = CLASSIFY.word2id(vocab)

        X_train = CLASSIFY.convert_to_normalised_bow_matrix(X_train, word2id)
        X_dev = CLASSIFY.convert_to_normalised_bow_matrix(X_dev, word2id)
        y_train = [corpuses2id[c] for c in y_train]
        y_dev = [corpuses2id[c] for c in y_dev]
        return X_train, X_dev, y_train, y_dev

    def prepare_baseline_test_data(verses, corpuses, vocab):
        
        word2id, corpuses2id = CLASSIFY.word2id(vocab)
        X_test = CLASSIFY.convert_to_bow_matrix(verses, word2id)
        y_test = [corpuses2id[c] for c in corpuses]
        return X_test, y_test,
    
    def prepare_improved_test_data(verses, corpuses, vocab):
        
        word2id, corpuses2id = CLASSIFY.word2id(vocab)
        X_test = CLASSIFY.convert_to_normalised_bow_matrix(verses, word2id)
        y_test = [corpuses2id[c] for c in corpuses]
        return X_test, y_test

    
    def scores_to_csv(scores_file, bl_train, bl_dev, bl_test, imp_train, imp_dev, imp_test):
        with open(scores_file, 'w') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',')
                    csv_writer.writerow(['system', 'split', 'p-quran', 'r-quran', 'f-quran', 'p-ot', 'r-ot', 'f-ot', 'p-nt', 'r-nt', 'f-nt', 'p-macro', 'r-macro', 'f-macro'])
                    for i,x in enumerate([bl_train, bl_dev, bl_test, imp_train, imp_dev, imp_test]):
                        model = 'baseline' if i<3 else 'improved'
                        score = 'train' if i in [0,3] else 'dev' if i in [1,4] else 'test'
                        csv_writer.writerow([model, score,
                                            round(x['2']['precision'],3), round(x['2']['recall'],3), round(x['2']['f1-score'],3),
                                            round(x['0']['precision'],3), round(x['0']['recall'],3), round(x['0']['f1-score'],3),
                                            round(x['1']['precision'],3), round(x['1']['recall'],3), round(x['1']['f1-score'],3),
                                            round(x['macro avg']['precision'],3), round(x['macro avg']['recall'],3), round(x['macro avg']['f1-score'],3)])


    def compute_accuracy(predictions, true_values):
        num_correct = 0
        num_total = len(predictions)
        for predicted,true in zip(predictions,true_values):
            if predicted==true:
                num_correct += 1
        return num_correct / num_total
    
    def train_base_model(data, test_data):
        verses, corpuses, vocab = CLASSIFY.baseline_data_preprocess(data)
        X_train, X_dev, y_train, y_dev, nX_dev = CLASSIFY.prepare_baseline_preprocessed_data(verses, corpuses, vocab)
        model = sklearn.svm.LinearSVC(C=1000, dual=False, max_iter= 20000) #dual=False, max_iter= 20000
        model.fit(X_train,y_train)

        y_train_predict = model.predict(X_train)
        y_dev_predict = model.predict(X_dev)

        test_verses, test_corpuses, _ = CLASSIFY.baseline_data_preprocess(test_data)
        X_test, y_test = CLASSIFY.prepare_baseline_test_data(test_verses, test_corpuses, vocab)

        y_test_predict = model.predict(X_test)
        return y_train_predict, y_dev_predict, y_test_predict, y_train, y_dev, y_test, nX_dev

    def train_improved_model(data, test_data):
        verses, corpuses, vocab = CLASSIFY.improved_data_preprocess(data)
        X_train, X_dev, y_train, y_dev = CLASSIFY.prepare_improved_preprocessed_data(verses, corpuses, vocab)
        model = sklearn.svm.SVC(C=50, verbose=True)
        model.fit(X_train,y_train)

        y_train_predict = model.predict(X_train)
        y_dev_predict = model.predict(X_dev)

        test_verses, test_corpuses, _ = CLASSIFY.improved_data_preprocess(test_data)
        X_test, y_test = CLASSIFY.prepare_improved_test_data(test_verses, test_corpuses, vocab)

        y_test_predict = model.predict(X_test)
        return y_train_predict, y_dev_predict, y_test_predict, y_train, y_dev, y_test
    
    def catch_incorrect(pred, true):
        incorr = []
        for i in range(len(pred)):
            if pred[i] != true[i]:
                incorr.append([pred[i],true[i],i])
        return incorr





    



if __name__ == "__main__":
    results_file = "system_results.csv"
    qrels_file = "qrels.csv"
    tsv_file = 'train_and_dev.tsv'
    stopwords_file = 'englishST.txt'
    test_file = 'test.tsv'
    scores_file = 'classification.csv'
    #####################################
    #TASK 1

    res = EVAL.parse_results(results_file)
    qrels = EVAL.parse_qrels(qrels_file)
    eval_results, means = EVAL.eval_systems(res, qrels)
    EVAL.significance_test(eval_results, means)
    #####################################
    #TASK 2

    stopwords = PREPROCESS.load_stopwords(stopwords_file)
    docs_tokens, corpuses_tokens_counts, n = PREPROCESS.process_tsv(tsv_file, stopwords)

    mi, chi_sq = ANALYSIS.analyse_corpuses(docs_tokens, corpuses_tokens_counts, n)
    for id, corpus in chi_sq.items():
        m = sorted(corpus.items(), key=lambda item: item[1], reverse=True)
        print(id, 'chi')
        print(m[:10])
    for id, corpus in mi.items():
        m = sorted(corpus.items(), key=lambda item: item[1], reverse=True)
        print(id, 'mi')
        print(m[:10])
    ANALYSIS.lda_topic_model(docs_tokens)
    ######################################
    #TASK 3
    data = open(tsv_file).read()

    test_data = open(test_file).read()

    bl_y_train_pred, bl_y_dev_pred, bl_y_test_pred, bl_y_train, bl_y_dev, bl_y_test, nX_dev = CLASSIFY.train_base_model(data, test_data)

    accuracy = CLASSIFY.compute_accuracy(bl_y_train_pred,bl_y_train)
    print("Training Accuracy:",accuracy)

    accuracy = CLASSIFY.compute_accuracy(bl_y_dev_pred,bl_y_dev)
    print("Validation Accuracy:",accuracy)
    incorr = CLASSIFY.catch_incorrect(bl_y_dev_pred, bl_y_dev)
    for i in incorr[:3]:
        print(f'pred {i[0]} actual,{i[1]}, {nX_dev[i[2]]}')

    accuracy = CLASSIFY.compute_accuracy(bl_y_test_pred,bl_y_test)
    print("Test Accuracy:",accuracy)


    imp_y_train_pred, imp_y_dev_pred, imp_y_test_pred, imp_y_train, imp_y_dev, imp_y_test = CLASSIFY.train_improved_model(data, test_data)

    accuracy = CLASSIFY.compute_accuracy(imp_y_train_pred,imp_y_train)
    print("Training Accuracy:",accuracy)

    accuracy = CLASSIFY.compute_accuracy(imp_y_dev_pred,imp_y_dev)
    print("Validation Accuracy:",accuracy)

    accuracy = CLASSIFY.compute_accuracy(imp_y_test_pred,imp_y_test)
    print("Test Accuracy:",accuracy)


    bl_train_scores = classification_report(bl_y_train, bl_y_train_pred, output_dict=True)
    bl_dev_scores = classification_report(bl_y_dev, bl_y_dev_pred, output_dict=True)
    bl_test_scores = classification_report(bl_y_test, bl_y_test_pred, output_dict=True)

    imp_train_scores = classification_report(imp_y_train, imp_y_train_pred, output_dict=True)
    imp_dev_scores = classification_report(imp_y_dev, imp_y_dev_pred, output_dict=True)
    imp_test_scores = classification_report(imp_y_test, imp_y_test_pred, output_dict=True)
  

    CLASSIFY.scores_to_csv(scores_file, bl_train_scores, bl_dev_scores, bl_test_scores, imp_train_scores, imp_dev_scores, imp_test_scores)
    


