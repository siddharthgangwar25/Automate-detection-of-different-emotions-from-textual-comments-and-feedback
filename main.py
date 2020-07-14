import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag,word_tokenize
import re
import numpas np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing

stop = stopwords.words('english')
new_inp = []
new_out = []
fin = open('data/train.txt','r')
for line in fin:
    values = line.split(' ')
    new_out.append(values[-1])
    x = values[:-1]
    x = [i for i in x if i not in stop]
    new_inp.append(' '.join(x))
fin.close()

x_train,y_train = new_inp,new_out


acc=[]
tfidf = TfidfVectorizer()
x_train_tfidf = tfidf.fit_transform(x_train)
tfidf_trans = TfidfTransformer()
x_train_trans = tfidf_trans.fit_transform(x_train_tfidf)


lr=LogisticRegression().fit(x_train_trans,y_train)
acc.append(lr.score(x_train_trans,y_train))

training_acc=acc[0]

stop = stopwords.words('english')
new_test_inp = []
new_test_out = []
test_fin = open('data/val.txt','r')
for line in test_fin:
    new_values = line.split(' ')
    new_test_out.append(new_values[-1].strip())
    z = new_values[:-1]
    z = [i for i in z if i not in stop]
    new_test_inp.append(' '.join(z))
test_fin.close()

x_test,y_test=new_test_inp,new_test_out

test_acc=[]
x_test_tfidf = tfidf.transform(x_test)
Y=lr.predict(x_test_tfidf)
Y = [i.strip() for i in Y]
y_test=[j.strip() for j in y_test]

from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(y_test,Y)
