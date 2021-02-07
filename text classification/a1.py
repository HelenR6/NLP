import numpy as np
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from itertools import chain 
import statistics;
from statistics import *
from sklearn.model_selection import cross_val_score
from sklearn import model_selection, naive_bayes, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import wordnet
from nltk import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#unprocessed data
ps = PorterStemmer()
enc='latin-1'
positive_reviews = [line.rstrip('\n') for line in open('/Users/apple/550/rt-polaritydata/rt-polaritydata/rt-polarity.pos', encoding=enc)]
negative_reviews = [line.rstrip('\n') for line in open('/Users/apple/550/rt-polaritydata/rt-polaritydata/rt-polarity.neg', encoding=enc)]
data = positive_reviews+negative_reviews
#Lemmatize data
lemmatizer = WordNetLemmatizer()
def nltk_to_wntag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:                    
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))    
    wn_tagged = map(lambda x: (x[0], nltk_to_wntag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:                        
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)


lemmatize_data = []

lemm_y = (["POSITIVE"] * len(positive_reviews))+(["NEGATIVE"] * len(negative_reviews))
for sentence in data:
    lemmatize_data.append(lemmatize_sentence(sentence))
#stemmer on the input data. 
stemm_data=[]
for sentence in data:
    stemm_data.append(" ".join([ps.stem(i) for i in sentence.split()]))

#remove stop words and split
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
tfidf1 = TfidfVectorizer()
X = tfidf1.fit_transform(data)
y = (["POSITIVE"] * len(positive_reviews))+(["NEGATIVE"] * len(negative_reviews))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split( X, y, test_size=0.15, random_state=42)
#selected model
lsvc=LinearSVC()
mnb = MultinomialNB()
lr=LogisticRegression()
dt = DecisionTreeClassifier()
# after removed stop words, the mean accuarcy rate of each selected model. 
print("when we don't remove stop words, the accuarcy rate is  ")
x=cross_val_score(lsvc,X_train,y_train,cv=5)
print("linear svm: ",x,"mean = ",mean(x))
x=cross_val_score(mnb,X_train,y_train,cv=5)
print("multinomial naive bayes: ",x,"mean = ",mean(x))
x=cross_val_score(lr,X_train,y_train,cv=5)
print("logistic regression: ",x,"mean = ",mean(x))
x=cross_val_score(dt,X_train,y_train,cv=5)
print("decision tree ",x,"mean = ",mean(x))
print()

X = tfidf.fit_transform(data)
X_train, X_test, y_train, y_test  = train_test_split( X, y, test_size=0.15, random_state=42)
y = (["POSITIVE"] * len(positive_reviews))+(["NEGATIVE"] * len(negative_reviews))
print("if we remove stop words, the accuarcy rate is  ")
x=cross_val_score(lsvc,X_train,y_train,cv=5)
print("linear svm: ",x,"mean = ",mean(x))
x=cross_val_score(mnb,X_train,y_train,cv=5)
print("multinomial naive bayes: ",x,"mean = ",mean(x))
x=cross_val_score(lr,X_train,y_train,cv=5)
print("logistic regression: ",x,"mean = ",mean(x))
x=cross_val_score(dt,X_train,y_train,cv=5)
print("decision tree ",x,"mean = ",mean(x))
print()

tfidf = TfidfVectorizer(analyzer='word',stop_words= 'english')
X = tfidf.fit_transform(data)
vectors_lemm = tfidf.fit_transform(lemmatize_data)
from sklearn.model_selection import train_test_split
X_lemm_train, X_lemm_test, y_lemm_train, y_lemm_test  = train_test_split( vectors_lemm, lemm_y, test_size=0.15, random_state=42)
print("when we only lemmatize the sentence, the accuarcy rate is  ")
x=cross_val_score(lsvc,X_lemm_train,y_lemm_train,cv=5)
print("linear svm: ",x,"mean = ",mean(x))
x=cross_val_score(mnb,X_lemm_train,y_lemm_train,cv=5)
print("multinomial naive bayes: ",x,"mean = ",mean(x))
x=cross_val_score(lr,X_lemm_train,y_lemm_train,cv=5)
print("logistic regression: ",x,"mean = ",mean(x))
x=cross_val_score(dt,X_lemm_train,y_lemm_train,cv=5)
print("decision tree ",x,"mean = ",mean(x))
print()

tfidf = TfidfVectorizer(analyzer='word',min_df= 2)
X = tfidf.fit_transform(data)
vectors_lemm = tfidf.fit_transform(lemmatize_data)
from sklearn.model_selection import train_test_split
X_lemm_train, X_lemm_test, y_lemm_train, y_lemm_test  = train_test_split( vectors_lemm, lemm_y, test_size=0.15, random_state=42)
print("when we lemmatize the sentence and remove rare words, the accuarcy rate is  ")
x=cross_val_score(lsvc,X_lemm_train,y_lemm_train,cv=5)
print("linear svm: ",x,"mean = ",mean(x))
x=cross_val_score(mnb,X_lemm_train,y_lemm_train,cv=5)
print("multinomial naive bayes: ",x,"mean = ",mean(x))
x=cross_val_score(lr,X_lemm_train,y_lemm_train,cv=5)
print("logistic regression: ",x,"mean = ",mean(x))
x=cross_val_score(dt,X_lemm_train,y_lemm_train,cv=5)
print("decision tree ",x,"mean = ",mean(x))
print()


stem_y = (["POSITIVE"] * len(positive_reviews))+(["NEGATIVE"] * len(negative_reviews))
tfidf = TfidfVectorizer(analyzer='word')
vectors_stem = tfidf.fit_transform(stemm_data)
from sklearn.model_selection import train_test_split
X_stem_train, X_stem_test, y_stem_train, y_stem_test  = train_test_split( vectors_stem, stem_y, test_size=0.3, random_state=42)
print("when we stemmer the sentence, the accuarcy rate is  ")
x=cross_val_score(lsvc,X_stem_train,y_stem_train,cv=5)
print("linear svm: ",x,"mean = ",mean(x))
x=cross_val_score(mnb,X_stem_train,y_stem_train,cv=5)
print("multinomial naive bayes: ",x,"mean = ",mean(x))
x=cross_val_score(lr,X_stem_train,y_stem_train,cv=5)
print("logistic regression: ",x,"mean = ",mean(x))
x=cross_val_score(dt,X_stem_train,y_stem_train,cv=5)
print("decision tree ",x,"mean = ",mean(x))
print()

tfidf = TfidfVectorizer(analyzer='word',stop_words= 'english',min_df=2)
vectors_stem = tfidf.fit_transform(stemm_data)
X_stem_train, X_stem_test, y_stem_train, y_stem_test  = train_test_split( vectors_stem, stem_y, test_size=0.3, random_state=42)
print("when we stemmer the sentence, and remove the stop words and rare words the accuarcy rate is  ")
x=cross_val_score(lsvc,X_stem_train,y_stem_train,cv=5)
print("linear svm: ",x,"mean = ",mean(x))
x=cross_val_score(mnb,X_stem_train,y_stem_train,cv=5)
print("multinomial naive bayes: ",x,"mean = ",mean(x))
x=cross_val_score(lr,X_stem_train,y_stem_train,cv=5)
print("logistic regression: ",x,"mean = ",mean(x))
x=cross_val_score(dt,X_stem_train,y_stem_train,cv=5)
print("decision tree ",x,"mean = ",mean(x))
print()

tfidf = TfidfVectorizer(analyzer='word',stop_words= 'english')
vectors_stem = tfidf.fit_transform(stemm_data)
X_stem_train, X_stem_test, y_stem_train, y_stem_test  = train_test_split( vectors_stem, stem_y, test_size=0.3, random_state=42)
print("when we stemmer the sentence, and remove the stop words the accuarcy rate is  ")
x=cross_val_score(lsvc,X_stem_train,y_stem_train,cv=5)
print("linear svm: ",x,"mean = ",mean(x))
x=cross_val_score(mnb,X_stem_train,y_stem_train,cv=5)
print("multinomial naive bayes: ",x,"mean = ",mean(x))
x=cross_val_score(lr,X_stem_train,y_stem_train,cv=5)
print("logistic regression: ",x,"mean = ",mean(x))
x=cross_val_score(dt,X_stem_train,y_stem_train,cv=5)
print("decision tree ",x,"mean = ",mean(x))
print()



lsvc=LinearSVC(C=0.6)
mnb = MultinomialNB(alpha=6)
x=cross_val_score(lsvc,X_train,y_train,cv=5)
print("tune regularization parameter of linearSVC to 0.6")
print("linear svm: ",x,"mean = ",mean(x))
x=cross_val_score(mnb,X_train,y_train,cv=5)
print("tune the smoothing parameter of multinomial naive bayes to 0.6")
print("multinomial naive bayes: ",x,"mean = " ,mean(x))
print()

print("confusion matrix when using naive bayes without extracting feature")
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data)
X_train, X_test, y_train, y_test  = train_test_split( X, y, test_size=0.15, random_state=42)
prediction=mnb.fit(X_train, y_train).predict(X_test)
print(confusion_matrix(y_test,prediction))
print()

print("final validation of multinomial bayes with smoothing parameter be 0.6")
Naive=naive_bayes.MultinomialNB(alpha=6)
Naive.fit(X_train,y_train)
predictions=Naive.predict(X_test)
print("Accuracy Score -> ",accuracy_score(predictions,y_test))