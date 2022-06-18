class TextClassifier():
  def __init__(self,train_dev_data_file,stop_words_file,test_file):
    self.train_dev_dataframe = self.create_df(train_dev_data_file)
    self.test_dataframe = self.create_df(test_file)
    self.porterstemmer = nltk.stem.PorterStemmer
    self.stop_words = self.get_stop_words(stop_words_file)
    self.preprocessed_train_df = self.preprocess_data(self.train_dev_dataframe)
    self.vocab, self.categories = self.generate_vocab_and_categories(self.preprocessed_train_df)
    self.word2id, self.cat2id = self.word_and_cat_2_id(self.vocab, self.categories)
    self.train, self.val = self.split_train_val(self.preprocessed_train_df,0.2)
    self.bow_train = self.convert_to_bow_matrix(self.train,self.word2id)
    self.y_train = self.get_labels(self.train['Corpus'],self.cat2id)
    self.bow_val = self.convert_to_bow_matrix(self.val,self.word2id)
    self.y_val = self.get_labels(self.val['Corpus'],self.cat2id)
    
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
      df.loc[row]['Verse'] = re.findall(r"\w+",df.loc[row]['Verse'].lower()) #Tokeniser
      df.loc[row]['Verse'] = [word for word in df.loc[row]['Verse'] if word not in self.stop_words] #Removing stop words
      #df.loc[row]['Verse'] = [self.porterstemmer.stem(word) for word in df.loc[row]['Verse']] #Porter Stemmer

    return df

  def generate_vocab_and_categories(self,df):
    vocab = []
    categories = []

    for verse in df['Verse']:
      vocab.extend(verse)

    for corpus in df['Corpus']:
      categories.extend(corpus)

    return list(set(vocab)), list(set(categories))

  def split_train_val(self,df, split_ratio):
    shuffled_df = df.sample(frac=1)

    train, val = sklearn.model_selection.train_test_split(shuffled_df,split_ratio)

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
    matrix_size = (df.shape[0], len(word2id))
    oov_index = len(word2id)
    X = scipy.sparse.dok_matrix(matrix_size)

    for doc_id,doc in enumerate(df['Verse']):
      for word in doc:
        X[doc_id,word2id.get(word,oov_index)] += 1

    return X

  def train_model(self,X_train,y_train,svc='linear',c=0.1):
    model = sklearn.svm.LinearSVC(C=c)
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
    return accuracy