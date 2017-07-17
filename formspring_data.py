from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing, svm, metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
import string
import csv
import os

def make_dataset():
        #dictionary contains the answers from the formspring data as keys and the combination of bullying occurances in the messages
        dct={}
        ques = 2
        ans = 3
        ans1 = 5
        ans2 = 8
        ans3 = 11
        bully1 = 7
        bully2 =10
        bully3 = 13
        with open('formspring.csv') as csvfile:
                reader = csv.reader(csvfile)
                i=0
                for rows in reader:
                        if i!=0:
                                for part in [ques, ans]:
                                        if rows[bully1] in rows[part]:
                                                dct[rows[part]]= True
                                        elif rows[bully2] in rows[part]:
                                                dct[rows[part]] = True  
                                        elif rows[bully3] in rows[part]:
                                                dct[rows[part]] = True  
                                        else:
                                                dct[rows[part]] = False                         
                        i+=1
        return dct # return dataset as dictionary with keys as text and values as True/False

def make_bag_of_words(dct):
        punctuation = set(string.punctuation)
        texts = []
        print('Appending texts')
        for tweet in dct:
                tweet = ''.join(ch if ch not in punctuation else ' ' for ch in tweet) # strip punctuation
                tweet = ' '.join(tweet.split())  # Remove whitespace
                tweet = tweet.lower()  # Convert to lowercase
                texts.append(tweet)
        print('Vectorirzer')
        vectorizer = CountVectorizer() # count word frequency
        bag_of_words = vectorizer.fit_transform(texts) # create bag of words model on text
        svm_bow = preprocessing.normalize(bag_of_words) # normalize for later use by SVM classifier
        X = bag_of_words.toarray() # change bag of words to numpy array
        y = [1 if dct[f] else 0 for f in dct] 
        return X, y, svm_bow

def train_test(X, y, svm_bow):
        print("Training and testing")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split data for train and test
        clf ={'RandomForest Classifier': RandomForestClassifier(),'SVM Classifier':svm.SVC()} # declare our classifiers
        print("\nClassifiers")
        for clas in clf:
                if clas == "SVM Classifier":
                        bag = svm_bow.toarray() # use normalized data for SVM classifier
                        bag_train, bag_test, y_train, y_test = train_test_split(bag, y, test_size=0.2)
                        clf[clas].fit(bag_train, y_train)
                        y_pred = clf[clas].predict(bag_test)
                else:
                        clf[clas].fit(X_train, y_train)
                        y_pred = clf[clas].predict(X_test)
        # print results and scores of all classifiers
        print("/n"+clas+" Report")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix")
        cmtx = confusion_matrix(y_test, y_pred)
        print(cmtx)
        acc = (cmtx[0][0] + cmtx[1][1])/(cmtx[0][0] + cmtx[1][1] + cmtx[0][1] + cmtx[1][0])
        print('Accuracy: %.2f' % acc)
        print("\nKappa Score:")
        print('%.2f' % cohen_kappa_score(y_test, y_pred))
        print('\nt-test: ')
        print(stats.ttest_ind(y_test, y_pred))

def k_fold(X, y):
        skf = StratifiedKFold(n_splits=2)
        for train_index, test_index in skf.split(X, y):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

if __name__ == "__main__":
        dataset = make_dataset()
        X, y, svm_bow = make_bag_of_words(dataset)
        #print(train_test(X, y, svm_bow))
        print(k_fold(X, y))
