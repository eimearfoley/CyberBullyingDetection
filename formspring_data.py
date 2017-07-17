from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import tree
from scipy import stats
import numpy as np
import string
import csv
import os

def make_dataset():
        # dict contains the answers from the formspring data as keys and the combination of bullying occurances in the messages
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

def make_samples(dct, n):
        sample = {}
        i = 0
        for true in dct:
                if dct[true]:
                        sample[true] = True
                if len(sample) == n//2:
                        break

        for false in dct:
                if not dct[false]:
                        sample[false] = False
                if len(sample) == n:
                        break
        return sample

def make_bag_of_words(dct):
        punctuation = set(string.punctuation)
        texts = []
        print('Appending texts')
        for tweet in dct:
                tweet = ''.join(ch if ch not in punctuation else ' ' for ch in tweet) # strip punctuation
                tweet = ' '.join(tweet.split())  # Remove whitespace
                tweet = tweet.lower()  # Convert to lowercase
                texts.append(tweet)
        print('Count Vectorirzer')
        vectorizer = CountVectorizer() # count word frequency
        bag_of_words = vectorizer.fit_transform(texts) # create bag of words model on text
        svm_bow = preprocessing.normalize(bag_of_words) # normalize for later use by SVM classifier
        X = bag_of_words.toarray() # change bag of words to numpy array
        y = [1 if dct[f] else 0 for f in dct] 
        return X, y, svm_bow

def make_tfidf(dct):
        punctuation = set(string.punctuation)
        texts = []
        print('Appending texts')
        for tweet in dct:
                tweet = ''.join(ch if ch not in punctuation else ' ' for ch in tweet) # strip punctuation
                tweet = ' '.join(tweet.split())  # Remove whitespace
                tweet = tweet.lower()  # Convert to lowercase
                texts.append(tweet)
        print('Tfidf Vectorirzer')
        vec = TfidfVectorizer()
        tfidf = vec.fit_transform(texts)
        svm_bow = preprocessing.normalize(tfidf)
        X = tfidf.toarray() # change tfidf to numpy array
        y = [1 if dct[f] else 0 for f in dct]
        return X, y, svm_bow

def make_crossfold(X, y):
        classifiers = {'GaussianNB': GaussianNB(), 'Decision Tree': tree.DecisionTreeClassifier(), 'RandomForest Classifier': RandomForestClassifier(),'Linear SVM Classifier': LinearSVC()}
        for clf in classifiers:
                scores = cross_val_score(classifiers[clf], X, y)
                print(clf, scores.mean()) 

def train_test(X, y, svm_bow):
        print("Training and testing")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split data for train and test
        clf ={'GaussianNB': GaussianNB(), 'Decision Tree': tree.DecisionTreeClassifier(), 'RandomForest Classifier': RandomForestClassifier(),'Linear SVM Classifier': LinearSVC()} # declare our classifiers
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
                print("\n"+clas+" Report")
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

def leave_one_out(X, y):
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X, y):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                print(X_train, X_test, y_train, y_test)

def k_fold(X, y):
        skf = StratifiedKFold(n_splits=2)
        for train_index, test_index in skf.split(X, y):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

def boost(X, y):
        clf = AdaBoostClassifier(n_estimators=100)
        scores = cross_val_score(clf, X, y)
        print('AdaBoost', scores.mean())
        

if __name__ == "__main__":
        dataset = make_dataset()
        dct = make_samples(dataset, 1200)
        print("\n****BAG OF WORDS****\n")
        X, y, svm_bow = make_bag_of_words(dct)
        print(train_test(X, y, svm_bow))
        print("\n****TFIDF****\n")
        X2, y2, svm_tfidf = make_tfidf(dct)
        print(train_test(X2, y2, svm_tfidf))
        print("\n****ADA BOOST CLASSIFIER BAG****\n")
        boost(X,y)
        print("\n****ADA BOOST CLASSIFIER TFIDF****\n")
        boost(X2,y2)
        print("\n****CROSS FOLD BAG****\n")
        make_crossfold(X, y)
        print("\n****CROSS FOLD TFIDF****\n")
        make_crossfold(X2, y2)
        print("\n****LEAVE ONE OUT BAG****\n")
        print(leave_one_out(X,y))
        print("\n****LEAVE ONE OUT TFIDF****\n")
        print(leave_one_out(X2,y2))
        #print(k_fold(X, y))
