from collections import defaultdict
import sklearn
import random    

import re
import string
import csv

import scipy
from scipy import sparse

from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import classification_report

import numpy as np

with open('train_and_dev.tsv', 'r') as file:
    doc = csv.reader(file , delimiter="\t", quoting=csv.QUOTE_NONE)
    corpuses = defaultdict(list)
    for line in doc:
        data = re.findall('\w+' , (str(line)).lower())    
        k = data[0]
        corpuses[k].append(data)

print('splitting data...')
training_data = []
test_data = []
for i, k in corpuses.items():
    x,y = sklearn.model_selection.train_test_split(k,test_size= 0.2 ,train_size = 0.8, shuffle=True)
    training_data += x
    test_data += y
    
random.shuffle(training_data) 
random.shuffle(test_data)

word2id = {}
train_vocab = set() ## unique terms to create numerical id for each
for t in training_data:
    train_vocab = train_vocab.union(set(t))

for word_id, word in enumerate(train_vocab):
    word2id[word] = word_id

## remove the corpus id from each doc
train_data_no_id = [k[1:] for k in training_data]
test_data_no_id = [k[1:] for k in test_data]

def convert_to_bow_matrix(preprocessed_data, word2id):    
    # matrix size is number of docs x vocab size + 1 (for OOV)
    matrix_size = (len(preprocessed_data),len(word2id)+1)
    oov_index = len(word2id)
    X = scipy.sparse.dok_matrix(matrix_size)

    for doc_id,doc in enumerate(preprocessed_data):
        for word in doc:
            # default is 0, so just add to the count for this word in this doc
            # if the word is oov, increment the oov_index
            X[doc_id,word2id.get(word,oov_index)] += 1
    return X

print('making BOW...')
X_train = convert_to_bow_matrix(train_data_no_id, word2id)
X_test = convert_to_bow_matrix(test_data_no_id, word2id)

# these are the labels to predict
label = {'ot' : 0, 'nt' : 1, 'quran' : 2}

y_train = [label[k[0]] for k in training_data]
y_test = [label[k[0]] for k in test_data]

print('training svm...')
model = sklearn.svm.LinearSVC(C=1000, max_iter = 15000, dual=False) ## need the max_iter and dual or it wont converge
model.fit(X_train,y_train)

y_train_predictions = model.predict(X_train)

def compute_accuracy(predictions, true_values):
    num_correct = 0
    num_total = len(predictions)
    for predicted,true in zip(predictions,true_values):
        if predicted==true:
            num_correct += 1
    return num_correct / num_total

accuracy = compute_accuracy(y_train_predictions,y_train)
print("Training Accuracy:",accuracy)

y_test_predictions = model.predict(X_test)
accuracy = compute_accuracy(y_test_predictions,y_test)
print("Validation Accuracy:",accuracy)







