from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LeaveOneOut, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,AdaBoostClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
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
    wrong = ['None', 'n/a', 'o', 'n/a0', "0`", "`0", 'N/a', 'N/A', 'n/A']
    with open('formspring.csv') as csvfile:
        reader = csv.reader(csvfile)
        i=0
        for rows in reader:
            if i!=0:
                for part in [ques, ans]:
                    if (rows[bully1] and rows[bully2]) in rows[part]:
                        dct[rows[part]]= True
                    elif (rows[bully2] and rows[bully3]) in rows[part]:
                        dct[rows[part]] = True  
                    elif (rows[bully3] and rows[bully1]) in rows[part]:
                        dct[rows[part]] = True
                    elif (rows[bully3] and rows[bully1] and rows[bully2]) in rows[part]:
                        dct[rows[part]] = True
                    elif  rows[6] not in wrong and int(rows[6]) >= 5:
                        if rows[bully1] in rows[part]:
                            dct[rows[part]]= True
                    elif  rows[9] not in wrong and int(rows[9]) >= 5:
                        if rows[bully2] in rows[part]:
                            dct[rows[part]]= True
                    elif  rows[12] not in wrong and int(rows[12]) >= 5:
                        if rows[bully3] in rows[part]:
                            dct[rows[part]]= True
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
    for txt in dct:
        txt = ''.join(ch if ch not in punctuation else ' ' for ch in txt) # strip punctuation
        txt = ' '.join(txt.split())  # Remove whitespace
        txt = txt.lower()  # Convert to lowercase
        texts.append(txt)
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
    for txt in dct:
        txt = ''.join(ch if ch not in punctuation else ' ' for ch in txt) # strip punctuation
        txt = ' '.join(txt.split())  # Remove whitespace
        txt = txt.lower()  # Convert to lowercase
        texts.append(txt)
    print('Tfidf Vectorirzer')
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(texts)
    svm_bow = preprocessing.normalize(tfidf)
    X = tfidf.toarray() # change tfidf to numpy array
    y = [1 if dct[f] else 0 for f in dct]
    return X, y, svm_bow

def train_test(X, y, svm_bow):
    print("Training and testing")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split data for train and test
    clf ={'Linear SVM Classifier': LinearSVC()} # declare our classifiers
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
    return results(y_test, y_pred, clas)

def results(y_test, y_pred, clas):
    # print results and scores of all classifiers
    print("\n"+clas+" Report")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix")
    cmtx = confusion_matrix(y_test, y_pred)
    print(cmtx)
    acc = (cmtx[0][0] + cmtx[1][1])/(cmtx[0][0] + cmtx[1][1] + cmtx[0][1] + cmtx[1][0])
    print('Accuracy: %.2f' % acc)
    print("\nKappa Score:")
    kappa = cohen_kappa_score(y_test, y_pred)
    print('%.2f' % kappa)
    print('\nt-test: ')
    print(stats.ttest_ind(y_test, y_pred))
    return kappa, acc

if __name__ == "__main__":
    data = make_dataset()
    k_scores = []
    a_scores = []
    n_range = []
    for n in range(100, 3001, 100):
        print("****NUMBER: %i ****" % n)
        dct = make_samples(data, n)    
        print("\n****BAG OF WORDS****\n")
        X, y, svm_bow = make_bag_of_words(dct)
        print("\n**** TRAIN & TEST BAG ****\n")
        k, a = train_test(X,y, svm_bow)
        k_scores += [k]
        a_scores += [a]
        n_range += [n]
    plt.plot(n_range, k_scores, 'r--', n_range, a_scores, 'b--')
    plt.show()
             
