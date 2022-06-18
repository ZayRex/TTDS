import numpy as np
import pandas as pd
import csv
import sklearn
import scipy
import nltk
import re
import math
from gensim.models import ldamodel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
import gensim.corpora as corpora

class Eval():
    def __init__(self,sys_file,q_file):
        self.data_sys = self.read_file(sys_file)
        self.data_q = self.read_file(q_file)
        self.system_number = list(map(int,np.unique(self.data_sys[:,0])))
        self.query_number = list(map(int,np.unique(self.data_sys[:,1])))
        self.index = self.create_dict()
        self.write_index_to_csv()

    def read_file(self,file_path):
      data = []
      with open(file_path,newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        next(reader)
        for row in reader:
          row = list(map(float,row))
          data.append(np.array(row))
        data = np.array(data)
      return data
  
    def calc_DCG(self,retrieved_relevant_docs,relevant_query):
        DCG = 0
        for idx,doc_id in enumerate(retrieved_relevant_docs[:,2]):
            if idx==0:
                if len(np.where(relevant_query[:,1] == doc_id)[0]) != 0:
                    DCG = relevant_query[relevant_query[:,1] == doc_id,2][0] 
            else:
                if len(np.where(relevant_query[:,1] == doc_id)[0]) != 0:
                    DCG += relevant_query[relevant_query[:,1] == doc_id,2][0]/np.log2(idx+1)
        return DCG
                
    def calc_iDCG(self,k,relevant_query):
        iDCG = 0
        query = relevant_query[:k]
        for idx,row in enumerate(query):
            if idx==0:
                iDCG = row[2]
            else:
                iDCG += row[2]/np.log2(idx+1)
        return iDCG
    
    def write_index_to_csv(self):
        with open('ir_eval.csv','w') as f:
            f.write("system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20" + "\n")
            for system in self.index:
                metrics = {'Pa10':0,'Ra50':0,'r-precision':0,'AP':0,'nDCGa10':0,'nDCGa20':0}
                for query in self.index[system]:
                    metrics['Pa10'] += self.index[system][query]['Pa10']
                    metrics['Ra50'] += self.index[system][query]['Ra50']
                    metrics['r-precision'] += self.index[system][query]['r-precision']
                    metrics['AP'] += self.index[system][query]['AP']
                    metrics['nDCGa10'] += self.index[system][query]['nDCGa10']
                    metrics['nDCGa20'] += self.index[system][query]['nDCGa20']
                    f.write(str(system) + ',' + str(query) + ',' + str(self.index[system][query]['Pa10']) + ',' +
                           str(self.index[system][query]['Ra50']) + ',' + str(self.index[system][query]['r-precision'])
                           + ',' + str(self.index[system][query]['AP']) + ',' + str(self.index[system][query]['nDCGa10'])
                           + ',' + str(self.index[system][query]['nDCGa20']) + '\n')
                    
                f.write(str(system) + ',' + 'mean' + ',' + str(round(metrics['Pa10']/10,3)) + ',' + 
                        str(round(metrics['Ra50']/10,3)) + ',' + str(round(metrics['r-precision']/10,3)) + ',' + 
                        str(round(metrics['AP']/10,3))+ ',' + 
                            str(round(metrics['nDCGa10']/10,3))+ ',' + str(round(metrics['nDCGa20']/10,3)) + '\n')
            f.close()
                    
    def create_dict(self):
        index = {}
        for system in (self.system_number):
            index[system] = {}
            for query in (self.query_number):
                index[system][query] = {'Pa10':0,'Ra50':0,'r-precision':0,'AP':0,'nDCGa10':0,'nDCGa20':0} 
                
        for system in (self.system_number):
            for query in (self.query_number):
                relevant_query = self.data_q[self.data_q[:,0] == query]
                all_docs = self.data_sys[self.data_sys[:,0] == system]
                retrieved = all_docs[all_docs[:,1] == query]
                retrieved_10 = retrieved[:10]
                retrieved_20 = retrieved[:20]
                retrieved_50 = retrieved[:50]
                retrieved_R = retrieved[:len(relevant_query)]
                
                retrieved_relevant = np.intersect1d(retrieved[:,2],relevant_query[:,1])
                retrieved_relevant_10 = np.intersect1d(retrieved_10[:,2],relevant_query[:,1])
                retrieved_relevant_20 = np.intersect1d(retrieved_20[:,2],relevant_query[:,1])
                retrieved_relevant_50 = np.intersect1d(retrieved_50[:,2],relevant_query[:,1])
                retrieved_relevant_R = np.intersect1d(retrieved_R[:,2],relevant_query[:,1])
                
                average_precision = 0
                num_docs_retrieved = 0
                num_docs_relevant = 0
                precisions = []
                
                for retrieved_doc in retrieved[:,2]:
                    num_docs_retrieved += 1
                    if retrieved_doc in relevant_query[:,1]:
                        num_docs_relevant += 1
                        precisions.append(num_docs_relevant/num_docs_retrieved)
                        
                        
                average_precision = sum(precisions)/len(relevant_query)
                    
                DCG_10 = self.calc_DCG(retrieved_10,relevant_query)
                iDCG_10 = self.calc_iDCG(10,relevant_query)
                
                DCG_20 = self.calc_DCG(retrieved_20,relevant_query)
                iDCG_20 = self.calc_iDCG(20,relevant_query)
                    
                        
                index[system][query]['Pa10'] = len(retrieved_relevant_10)/10
                index[system][query]['Ra50'] = round(len(retrieved_relevant_50)/len(relevant_query),3)
                index[system][query]['r-precision'] = round(len(retrieved_relevant_R)/len(relevant_query),3)
                index[system][query]['AP'] = round(average_precision,3)
                index[system][query]['nDCGa10'] = round(DCG_10/iDCG_10,3)
                index[system][query]['nDCGa20'] = round(DCG_20/iDCG_20,3)
                
        return index

class TextClassifier():
  def __init__(self,train_dev_data_file,test_file):
    self.train_dev_dataframe = self.create_df(train_dev_data_file)
    self.test_dataframe = self.create_df(test_file)
    self.porterstemmer = nltk.stem.PorterStemmer()
    #self.stop_words = self.get_stop_words(stop_words_file)
    self.preprocessed_train_df = self.preprocess_data(self.train_dev_dataframe)
    self.vocab, self.categories = self.generate_vocab_and_categories(self.preprocessed_train_df)
    #print(self.categories)
    self.word2id, self.cat2id = self.word_and_cat_2_id(self.vocab, self.categories)
    #print(self.cat2id)
    self.Xtrain, self.Xval = self.split_train_val(self.preprocessed_train_df,split_ratio=0.2)
    self.bow_train = self.convert_to_bow_matrix(self.Xtrain,self.word2id)
    self.y_train = self.get_labels(self.Xtrain['Corpus'],self.cat2id)
    self.bow_val = self.convert_to_bow_matrix(self.Xval,self.word2id)
    self.y_val = self.get_labels(self.Xval['Corpus'],self.cat2id)
    self.bow_train_baseline = self.convert_to_bow_matrix_baseline(self.Xtrain,self.word2id)
    self.bow_val_baseline = self.convert_to_bow_matrix_baseline(self.Xval,self.word2id)
    self.processed_test_data = self.preprocess_data(self.test_dataframe)
    self.test_vocab, self.test_cats = self.generate_vocab_and_categories(self.processed_test_data)
    #self.test_word2id, self.test_cat2id = self.word_and_cat_2_id(self.test_vocab,self.test_cats)
    self.y_test_gt = self.get_labels(self.processed_test_data['Corpus'],self.cat2id)
    self.bow_test = self.convert_to_bow_matrix(self.processed_test_data,self.word2id)
    self.bow_test_baseline = self.convert_to_bow_matrix_baseline(self.processed_test_data,self.word2id)

    self.baseline_model = self.train_model(self.bow_train_baseline,self.y_train,linear=True,c=1000)
    self.model = self.train_model(self.bow_train,self.y_train)

    self.baseline_train_acc, self.baseline_train_preds = self.predict(self.baseline_model,self.bow_train_baseline,self.y_train)
    self.baseline_val_acc, self.baseline_val_preds = self.predict(self.baseline_model,self.bow_val_baseline, self.y_val)
    self.baseline_test_acc, self.baseline_test_preds = self.predict(self.baseline_model,self.bow_test_baseline, self.y_test_gt)


    self.train_acc, self.train_preds = self.predict(self.model,self.bow_train,self.y_train)
    self.val_acc, self.val_preds = self.predict(self.model,self.bow_val,self.y_val)
    self.test_acc, self.test_preds = self.predict(self.model,self.bow_test, self.y_test_gt)


    self.baseline_train_metrics = self.compute_metrics(self.baseline_train_preds,self.y_train)
    self.baseline_dev_metrics = self.compute_metrics(self.baseline_val_preds,self.y_val)
    self.baseline_test_metrics = self.compute_metrics(self.baseline_test_preds,self.y_test_gt)

    self.improved_train_metrics = self.compute_metrics(self.train_preds,self.y_train)
    self.improved_dev_metrics = self.compute_metrics(self.val_preds,self.y_val)
    self.improved_test_metrics = self.compute_metrics(self.test_preds,self.y_test_gt)

    self.Xval.index = list(i for i in range(self.Xval.shape[0]))

    self.misclassification_baseline_dev,self.baseline_wrong_pred_dev,self.true_labels_dev = self.misclassifications(self.Xval,self.baseline_val_preds,self.y_val)
    self.misclassification_improved_dev,self.improved_wrong_pred_dev,self.true_labels_dev = self.misclassifications(self.Xval,self.val_preds,self.y_val)

    self.write_to_csv()


  def write_to_csv(self):
    with open('classification.csv','w') as f:
      f.write('system,split,p-quran,r-quran,f-quran,p-ot,r-ot,f-ot,p-nt,r-nt,f-nt,p-macro,r-macro,f-macro' + '\n')
      f.write('baseline,train,' + str(self.baseline_train_metrics['Quran']['precision']) + ','
              + str(self.baseline_train_metrics['Quran']['recall']) + ',' + str(self.baseline_train_metrics['Quran']['f1-score'])
              + ',' + str(self.baseline_train_metrics['OT']['precision']) + ',' + str(self.baseline_train_metrics['OT']['recall'])
              + ',' + str(self.baseline_train_metrics['OT']['f1-score']) + ',' + str(self.baseline_train_metrics['NT']['precision'])
              + ',' + str(self.baseline_train_metrics['NT']['recall']) + ',' + str(self.baseline_train_metrics['NT']['f1-score'])
              + ',' + str(self.baseline_train_metrics['p-macro']) + ',' + str(self.baseline_train_metrics['r-macro'])
              + ',' + str(self.baseline_train_metrics['f-macro']) + '\n')
      f.write('baseline,dev,' + str(self.baseline_dev_metrics['Quran']['precision']) + ','
              + str(self.baseline_dev_metrics['Quran']['recall']) + ',' + str(self.baseline_dev_metrics['Quran']['f1-score'])
              + ',' + str(self.baseline_dev_metrics['OT']['precision']) + ',' + str(self.baseline_dev_metrics['OT']['recall'])
              + ',' + str(self.baseline_dev_metrics['OT']['f1-score']) + ',' + str(self.baseline_dev_metrics['NT']['precision'])
              + ',' + str(self.baseline_dev_metrics['NT']['recall']) + ',' + str(self.baseline_dev_metrics['NT']['f1-score'])
              + ',' + str(self.baseline_dev_metrics['p-macro']) + ',' + str(self.baseline_dev_metrics['r-macro'])
              + ',' + str(self.baseline_dev_metrics['f-macro']) + '\n')
      f.write('baseline,test,' + str(self.baseline_test_metrics['Quran']['precision']) + ','
              + str(self.baseline_test_metrics['Quran']['recall']) + ',' + str(self.baseline_test_metrics['Quran']['f1-score'])
              + ',' + str(self.baseline_test_metrics['OT']['precision']) + ',' + str(self.baseline_test_metrics['OT']['recall'])
              + ',' + str(self.baseline_test_metrics['OT']['f1-score']) + ',' + str(self.baseline_test_metrics['NT']['precision'])
              + ',' + str(self.baseline_test_metrics['NT']['recall']) + ',' + str(self.baseline_test_metrics['NT']['f1-score'])
              + ',' + str(self.baseline_test_metrics['p-macro']) + ',' + str(self.baseline_test_metrics['r-macro'])
              + ',' + str(self.baseline_test_metrics['f-macro']) + '\n')

      f.write('improved,train,' + str(self.improved_train_metrics['Quran']['precision']) + ','
              + str(self.improved_train_metrics['Quran']['recall']) + ',' +str(self.improved_train_metrics['Quran']['f1-score'])
              + ',' + str(self.improved_train_metrics['OT']['precision']) + ',' + str(self.improved_train_metrics['OT']['recall'])
              + ',' + str(self.improved_train_metrics['OT']['f1-score']) + ',' +str(self.improved_train_metrics['NT']['precision'])
              + ',' + str(self.improved_train_metrics['NT']['recall']) + ',' +str(self.improved_train_metrics['NT']['f1-score'])
              + ',' + str(self.improved_train_metrics['p-macro']) + ',' +str(self.improved_train_metrics['r-macro'])
              + ',' + str(self.improved_train_metrics['f-macro']) + '\n')

      f.write('improved,dev,' + str(self.improved_dev_metrics['Quran']['precision'])
              + ',' + str(self.improved_dev_metrics['Quran']['recall']) + ',' + str(self.improved_dev_metrics['Quran']['f1-score'])
              + ','  +str(self.improved_dev_metrics['OT']['precision']) + ',' + str(self.improved_dev_metrics['OT']['recall'])
              + ',' +str(self.improved_dev_metrics['OT']['f1-score']) + ',' + str(self.improved_dev_metrics['NT']['precision'])
              + ',' +str(self.improved_dev_metrics['NT']['recall']) + ',' + str(self.improved_dev_metrics['NT']['f1-score'])
              + ',' + str(self.improved_dev_metrics['p-macro']) + ',' +str(self.improved_dev_metrics['r-macro'])
              + ',' +str(self.improved_dev_metrics['f-macro']) + '\n')
      
      f.write('improved,test,' + str(self.improved_test_metrics['Quran']['precision'])
              + ',' + str(self.improved_test_metrics['Quran']['recall']) + ',' + str(self.improved_test_metrics['Quran']['f1-score'])
              + ',' + str(self.improved_test_metrics['OT']['precision']) + ',' + str(self.improved_test_metrics['OT']['recall'])
              + ',' + str(self.improved_test_metrics['OT']['f1-score']) + ',' + str(self.improved_test_metrics['NT']['precision'])
              + ',' + str(self.improved_test_metrics['NT']['recall']) + ',' + str(self.improved_test_metrics['NT']['f1-score'])
              + ',' + str(self.improved_test_metrics['p-macro']) + ',' + str(self.improved_test_metrics['r-macro'])
              + ',' + str(self.improved_test_metrics['f-macro']))
      f.close()


  def create_df(self,tsv_file):
    df = pd.read_csv(tsv_file,sep='\t',quoting=csv.QUOTE_NONE, header=None)
    df.columns = ['Corpus','Verse']
    return df

  def get_stop_words(self,stop_words_file):
    with open(stop_words_file,'r') as f:
      stop_words = set(f.readlines())
      stop_words_final = []
      for line in stop_words:
        stop_words_final.append(line.strip())
      return stop_words_final

  def preprocess_data(self,df):
    for row in range(df.shape[0]):
      df.loc[row]['Verse'] = re.findall(r"\w+",df.loc[row]['Verse']) #Tokeniser
      #df.loc[row]['Verse'] = sklearn.feature_extraction.text.CountVectorizer().fit_transform(df.loc[row]['Verse'])
      #df.loc[row]['Verse'] = [word for word in df.loc[row]['Verse'] if word not in self.stop_words] #Removing stop words
      #df.loc[row]['Verse'] = [self.porterstemmer.stem(word) for word in df.loc[row]['Verse']] #Porter Stemmer

    return df

  def generate_vocab_and_categories(self,df):
    vocab = []
    categories = []

    for verse in df['Verse']:
      vocab.extend(verse)
    
    categories.extend(df['Corpus'])

    return (set(vocab)), (set(categories))

  def split_train_val(self,df, split_ratio):
    shuffled_df = df.sample(frac=1)

    train, val = sklearn.model_selection.train_test_split(shuffled_df,test_size=split_ratio,random_state=2021)

    return train,val

  def word_and_cat_2_id(self,words,categories):
    word2id = {}
    cat2id = {}

    for word_idx,word in enumerate(words):
      word2id[word] = word_idx

    for cat_idx,cat in enumerate(categories):
      cat2id[cat] = cat_idx

    return word2id,cat2id

  def get_labels(self,train_categories,cat2id):
    return [cat2id[cat] for cat in train_categories]

  def convert_to_bow_matrix(self,df,word2id):
    matrix_size = (df.shape[0], len(word2id) + 1)
    oov_index = len(word2id)
    X = scipy.sparse.dok_matrix(matrix_size)

    for doc_id,doc in enumerate(df['Verse']):
      for word in doc:
        X[doc_id,word2id.get(word,oov_index)] += 1/len(doc)

    return X

  def convert_to_bow_matrix_baseline(self,df,word2id):
    matrix_size = (df.shape[0], len(word2id) + 1)
    oov_index = len(word2id)
    X = scipy.sparse.dok_matrix(matrix_size)

    for doc_id,doc in enumerate(df['Verse']):
      for word in doc:
        X[doc_id,word2id.get(word,oov_index)] += 1

    return X

  def train_model(self,X_train,y_train,linear=False,c=100):
    if linear:
      model = sklearn.svm.LinearSVC(C=c)
    model = sklearn.svm.SVC(C=c,kernel='rbf')
    model.fit(X_train,y_train)
    return model

  def predict(self,model,X_test,y_test_gt):
    predictions = model.predict(X_test)
    num_correct = 0
    num_total = len(predictions)

    for pred,gt in zip(predictions,y_test_gt):
      if pred == gt:
        num_correct+=1

    accuracy = num_correct/num_total
    return accuracy,predictions

  def misclassifications(self,dev_set,preds,labels):
    misclassified = []
    misclassified_labels = []
    misclassified_pred = []
    for row_idx,pred in enumerate(preds):
      if pred != labels[row_idx]:
        misclassified.append(dev_set.loc[row_idx]['Verse'])
        misclassified_labels.append(labels[row_idx])
        misclassified_pred.append(pred)
    return misclassified, misclassified_labels, misclassified_pred

  def compute_metrics(self,predictions,labels):
    metrics = {'OT':{},'NT':{},'Quran':{}}
    metrics_dict = sklearn.metrics.classification_report(labels,predictions,output_dict=True)

    for cat in self.categories:
      metrics[cat]['precision'] = round(metrics_dict[str(self.cat2id[cat])]['precision'],3)
      metrics[cat]['recall'] = round(metrics_dict[str(self.cat2id[cat])]['recall'],3)
      metrics[cat]['f1-score'] = round(metrics_dict[str(self.cat2id[cat])]['f1-score'],3)

    metrics['p-macro'] = round(metrics_dict['macro avg']['precision'],3)
    metrics['r-macro'] = round(metrics_dict['macro avg']['recall'],3)
    metrics['f-macro'] = round(metrics_dict['macro avg']['f1-score'],3)

    return metrics

class TextEval():
  def __init__(self,file_path,stop_words_path):
    self.df = self.create_df(file_path)
    self.porterstemmer = nltk.stem.PorterStemmer()
    if stop_words_path is not None:
      self.stop_words = self.get_stop_words(stop_words_path)
    
    self.preprocessed_df = self.preprocess_df(self.df)
    self.tokens = []
    for token in self.preprocessed_df['Verse']:
      self.tokens += token

    self.tokens = list(set(self.tokens))
    self.corpuses = list(set(self.preprocessed_df['Corpus']))
    self.michisquared = self.mi_chi_squared()

    self.texts = []
    self.corpus = []
    for x in self.preprocessed_df['Verse']:
      self.texts.append(list(x))

    self.id2word = corpora.Dictionary(self.texts)
    for x in self.preprocessed_df['Verse']:
      self.corpus.append(self.id2word.doc2bow(list(x)))

    self.quran_docs = []
    self.ot_docs = []
    self.nt_docs = []
    for row in range(self.preprocessed_df.shape[0]):
      if (self.preprocessed_df.loc[row]['Corpus']=='Quran'):
        self.quran_docs.append(self.id2word.doc2bow(list(self.preprocessed_df.loc[row]['Verse'])))
      elif (self.preprocessed_df.loc[row]['Corpus']=='OT'):
        self.ot_docs.append(self.id2word.doc2bow(list(self.preprocessed_df.loc[row]['Verse'])))
      elif (self.preprocessed_df.loc[row]['Corpus']=='NT'):
        self.nt_docs.append(self.id2word.doc2bow(list(self.preprocessed_df.loc[row]['Verse']))) 

    self.lda_model = ldamodel.LdaModel(corpus=self.corpus,
                                  id2word=self.id2word,
                                  num_topics=20,
                                  update_every=1,
                                  chunksize=100,
                                  passes=20,
                                  alpha='auto',
                                  random_state=2021)
    
    
    self.quran_most_likely_topic_idx, self.ot_most_likely_topic_idx, self.nt_most_likely_topic_idx = self.lda_topic_analysis(self.quran_docs,self.ot_docs,self.nt_docs)
    self.write_michi_to_csv()
    self.write_to_txt()

  def create_df(self,tsv_file):
    df = pd.read_csv(tsv_file,sep='\t',quoting=csv.QUOTE_NONE, header=None)
    df.columns = ['Corpus','Verse']
    return df

  def get_stop_words(self,stop_words_file):
    with open(stop_words_file,'r') as f:
      stop_words = set(f.readlines())
      stop_words_final = []
      for line in stop_words:
        stop_words_final.append(line.strip())
      return stop_words_final

  def preprocess_df(self,df):
    for row in range(df.shape[0]):
      df.loc[row]['Verse'] = re.findall(r"\w+",df.loc[row]['Verse'].lower()) #Tokeniser
      df.loc[row]['Verse'] = [word for word in df.loc[row]['Verse'] if word not in self.stop_words] #Removing stop words
      df.loc[row]['Verse'] = [self.porterstemmer.stem(word) for word in df.loc[row]['Verse']] #Porter Stemmer
    return df

  def calc_mi(self,N,N_00,N_01,N_10,N_11):
    try:
      first_term = (N_11/N)*math.log2(float(N*N_11)/float((N_10+N_11)*(N_01+N_11))) 
    except:
      first_term = 0
    try:
      second_term = (N_01/N)*math.log2(float(N*N_01)/float((N_00+N_01)*(N_01+N_11)))
    except:
      second_term = 0
    try:
      third_term = (N_10/N)*math.log2(float(N*N_10)/float((N_10+N_11)*(N_10+N_00)))
    except:
      third_term = 0
    try:
      fourth_term = (N_00/N)*math.log2(float(N*N_00)/float((N_00+N_01)*(N_00+N_10)))
    except:
      fourth_term = 0

    mi = first_term + second_term + third_term + fourth_term
    return mi

  def calc_chi_squared(self,N_00,N_01,N_10,N_11):
    numerator = (N_11+N_10+N_01+N_00)*(N_11*N_00 - N_10*N_01)**2
    denominator = (N_11+N_01)*(N_11+N_10)*(N_10+N_00)*(N_01+N_00)
    chi_squared = numerator/denominator
    return chi_squared

  def mi_chi_squared(self):
    michisquare = {}
    for idx,token in enumerate(self.tokens):
      michisquare[token] = {}
      for corpus in self.corpuses:
        michisquare[token][corpus] = {}
        verses_belonging_to_corpus = self.preprocessed_df[self.preprocessed_df['Corpus'] == corpus]['Verse']
        verses_rest = self.preprocessed_df[self.preprocessed_df['Corpus'] != corpus]['Verse']
        N = self.preprocessed_df.shape[0]
        N_00 = len([0 for x in verses_rest if token not in x])
        N_01 = len([0 for x in verses_belonging_to_corpus if token not in x])
        N_10 = len(verses_rest) - N_00
        N_11 = len(verses_belonging_to_corpus) - N_01
        
        mi = self.calc_mi(N,N_00,N_01,N_10,N_11)
        chi_squared = self.calc_chi_squared(N_00,N_01,N_10,N_11)
        michisquare[token][corpus]['MI'] = round(mi,3)
        michisquare[token][corpus]['CHI_SQ'] = round(chi_squared,3)
    return michisquare

  def write_michi_to_csv(self):
    mi_dict = {'Quran':{},'OT':{},'NT':{}}
    chi_squared_dict = {'Quran':{},'OT':{},'NT':{}}

    for token in self.michisquared:
      for corpus in self.michisquared[token]:
        mi_dict[corpus][token] = self.michisquared[token][corpus]['MI']
        chi_squared_dict[corpus][token] = self.michisquared[token][corpus]['CHI_SQ']
    
    for corpus in self.corpuses:
      sorted(mi_dict[corpus].items(), key=lambda x:x[1], reverse=True)
      sorted(chi_squared_dict[corpus].items(), key=lambda x:x[1], reverse=True)

    with open('mi_quran.csv','w') as f:
      for token in mi_dict['Quran'].keys():
        f.write(str(token) + ',' + str(mi_dict['Quran'][token]) + '\n')
      f.close()
    
    with open('mi_ot.csv','w') as f:
      for token in mi_dict['OT'].keys():
        f.write(str(token) + ',' + str(mi_dict['OT'][token]) + '\n')
      f.close()

    with open('mi_nt.csv','w') as f:
      for token in mi_dict['NT'].keys():
        f.write(str(token) + ',' + str(mi_dict['NT'][token]) + '\n')
      f.close()

    with open('chi_squared_quran.csv','w') as f:
      #for corpus in self.corpuses:
      for token in chi_squared_dict['Quran'].keys():
        f.write(str(token) + ',' + str(chi_squared_dict['Quran'][token]) + '\n')
      f.close()

    with open('chi_squared_ot.csv','w') as f:
      for token in chi_squared_dict['OT'].keys():
        f.write(str(token) + ',' + str(chi_squared_dict['OT'][token]) + '\n')
      f.close()

    with open('chi_squared_nt.csv','w') as f:
      for token in chi_squared_dict['NT'].keys():
        f.write(str(token) + ',' + str(chi_squared_dict['NT'][token]) + '\n')
      f.close()
  
  def write_to_txt(self):
    with open('lda_results.txt','w') as f:
      for idx,topic in self.lda_model.print_topics(-1):
        parsed_topics = re.findall(r'\w+',topic)
        parsed_string = ""
        for parsed in parsed_topics:
          parsed_string += parsed + ' '
        if idx == self.quran_most_likely_topic_idx:
          f.write('Most likely topic and terms in the Quran ' + parsed_string + '\n')
        if idx == self.ot_most_likely_topic_idx:
          f.write('Most likely topic and terms in the OT ' + parsed_string + '\n')
        if idx == self.nt_most_likely_topic_idx:
          f.write('Most likely topic and terms in the NT ' + parsed_string + '\n')
      f.close()


  def get_docs(self,lda,corpus):
    return [lda.get_document_topics(bow=doc,minimum_probability=0) for doc in corpus]

  def avg_score(self,doc_prob):
      avg_scores = []
      for topic in range(20):
          sum_per_topic=0
          for document in doc_prob:
              sum_per_topic += document[topic][1]
              avg_scores.append(sum_per_topic/len(doc_prob))
      return avg_scores


  def lda_topic_analysis(self,quran_docs,ot_docs,nt_docs):
    quran_docs_probs = self.get_docs(self.lda_model,self.quran_docs)
    ot_docs_probs = self.get_docs(self.lda_model, self.ot_docs)
    nt_docs_probs = self.get_docs(self.lda_model, self.nt_docs)

    quran_avg_topic_score = self.avg_score(quran_docs_probs)
    ot_avg_topic_score = self.avg_score(ot_docs_probs)
    nt_avg_topic_score = self.avg_score(nt_docs_probs)

    quran_topics = sorted(list(zip(range(20),quran_avg_topic_score)), key=lambda x:x[1], reverse=True)
    ot_topics = sorted(list(zip(range(20),ot_avg_topic_score)), key=lambda x:x[1], reverse=True)
    nt_topics = sorted(list(zip(range(20),nt_avg_topic_score)), key=lambda x:x[1], reverse=True)


    quran_idx = max(quran_topics)
    ot_idx = max(ot_topics)
    nt_idx =  max(nt_topics)
    
    return quran_idx, ot_idx ,nt_idx

if __name__ == "__main__":
    #evaluator = Eval('system_results.csv','qrels.csv')
    #text_classifier = TextClassifier('/content/drive/MyDrive/train_and_dev.tsv',
                                # '/content/drive/MyDrive/test.tsv')
    text_eval = TextEval('train_and_dev.tsv','englishST.txt')
    