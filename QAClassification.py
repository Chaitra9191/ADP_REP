#Import statements
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
nltk.download('all')
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn import svm
import time

#Class defination
class Model:
    def __init__(self, datafile="train.csv"):
        # Constructor
        self.df = pd.read_csv(datafile)
        self.special_character_remover = re.compile('[/(){}\[\]\|@,;]')
        self.extra_symbol_remover = re.compile('[^0-9a-z #+_]')
        self.STOPWORDS = set(stopwords.words('english'))
        self.df = self.basic_clean( self.df)
    def clean_text(self, text):
        # Method to do advanced cleaning
        text = text.lower()
        text = self.special_character_remover.sub(' ', text)
        text = self.extra_symbol_remover.sub('', text)
        text = ' '.join(word for word in text.split() if word not in self.STOPWORDS)
        return text
    def calculatet_ex_time(self,start_time):
        # Method to calculate execution time
        recorded_time = time.time() - start_time
        return recorded_time
    def basic_clean(self,df):
        # Method to do basic cleaning
        df.fillna('', inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop('Id', axis=1, inplace=True)
        df['conversation'] = df['Question'] + df['Answer']
        df.drop('Question', axis=1, inplace=True)
        df.drop('Answer', axis=1, inplace=True)
        df['conversation'] = df['conversation'].apply(self.clean_text)
        return df
    def split(self, test_size):
        #Code to split train and test data
        X = pd.DataFrame(self.df['conversation'], columns=['conversation'])
        Y = pd.DataFrame(self.df['category'], columns=['category'])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=test_size,
                                                                                random_state=42)

    def naiveBayes(self):
        #Naive bayes for multiclass classification
        start_time = time.time()
        #pipeline set for countvectorizer and Tf-Idf
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()), ])
        text_clf = text_clf.fit(list(self.x_train['conversation']), list(self.y_train['category']))
        predicted = text_clf.predict(list(self.x_test['conversation']))
        print('time taken to run naive bayes in seconds : ', self.calculatet_ex_time(start_time))
        print('accuracy of naive bayes: ', accuracy_score(self.y_test, predicted))
    def logisticRegression(self):
        # logisticRegression for multiclass classification
        start_time = time.time()
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(penalty = 'l1',solver='liblinear')) ])
        text_clf = text_clf.fit(list(self.x_train['conversation']), list(self.y_train['category']))
        predicted = text_clf.predict(list(self.x_test['conversation']))
        print('time taken to run logistic regression in seconds  : ', self.calculatet_ex_time(start_time))
        print('accuracy of logistic regression : ', accuracy_score(self.y_test, predicted))
    def xgBoost(self):
        # xgBoost for multiclass classification
        start_time = time.time()
        label_encoder = preprocessing.LabelEncoder()
        self.y_train['category'] = label_encoder.fit_transform(self.y_train['category'])
        self.y_test['category'] = label_encoder.fit_transform(self.y_test['category'])
        text_clf_xg = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-xgb', xgb.XGBClassifier(n_estimators = 150 ,max_depth=15))])
        text_clf_xg = text_clf_xg.fit(self.x_train['conversation'], self.y_train['category'])
        predicted = text_clf_xg.predict(list(self.x_test['conversation']))
        print('time taken to run xgBoost in seconds : ', self.calculatet_ex_time(start_time))
        print('accuracy of xgBoost: ', accuracy_score(self.y_test, predicted))
    def svm(self):
        # svm for multiclass classification
        start_time = time.time()
        self.text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-svm', svm.SVC(kernel='rbf',gamma=1, C=1, decision_function_shape='ovo'))])
        self.text_clf_svm = self.text_clf_svm.fit(list(self.x_train['conversation']), list(self.y_train['category']))
        predicted = self.text_clf_svm.predict(list(self.x_test['conversation']))
        print('time taken to run svm in seconds : ', self.calculatet_ex_time(start_time))
        print('accuracy of svm: ', accuracy_score(self.y_test, predicted))
    def testing(self):
        #Code to predict new data
        df_test_ = pd.read_csv('test.csv')
        df_test = self.basic_clean(df_test_)
        predicted = self.text_clf_svm.predict(list(df_test['conversation']))
        result  = pd.concat([pd.read_csv('test.csv'),pd.DataFrame(predicted)],axis = 1)
        result.rename(columns={0:'Category'},inplace=True)
        result.to_csv (r'submission.csv', index = False, header=True)
    def generate_pickle(self):
        #For deployment purposes
        pickle.dump( self.text_clf_svm, open('svm.sav','wb'))
if __name__ == '__main__':
    model_instance = Model()
    model_instance.split(0.2)
    model_instance.naiveBayes()
    model_instance.logisticRegression()
    model_instance.xgBoost()
    model_instance.svm()
    model_instance.testing()
    model_instance.generate_pickle()