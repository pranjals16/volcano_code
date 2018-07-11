import numpy as np
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

BASE_PATH = "C:/Users/pranjal/Desktop/kettle/model_data/"
SCIENCE_PATH = BASE_PATH + "science/*.txt"
TECH_PATH = BASE_PATH + "technology/*.txt"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

clf_nb = MultinomialNB().fit(X_train_tfidf, y_train)
clf_rf = RandomForestClassifier().fit(X_train_tfidf, y_train)


predicted_nb = clf_nb.predict(X_test_tfidf)
predicted_rf = clf_rf.predict(X_test_tfidf)
print "Naive Bayes: ", np.mean(predicted_nb == y_test)
print "Random Forest: ", np.mean(predicted_rf == y_test)
