from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LeaveOneOut
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

def make_stronger_dataset():
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

def train_test(X, y):
        print("Training and testing")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split data for train and test
        return X_train, X_test, y_train, y_test

def k_fold(X, y):
        y = np.array(y)
        skf = StratifiedKFold(n_splits=10)
        for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test

def classify(X_train, X_test, y_train, y_test):
        clf ={'GaussianNB': GaussianNB(), 'Decision Tree': tree.DecisionTreeClassifier(), 'RandomForest Classifier': RandomForestClassifier(),'Linear SVM Classifier': LinearSVC()} # declare our classifiers
        print("\nClassifiers")
        for clas in clf:
                clf[clas].fit(X_train, y_train)
                y_pred = clf[clas].predict(X_test)
                # print results and scores of all classifiers
                print("\n"+clas+" Report")
                print(classification_report(y_test, y_pred))
                print("\nConfusion Matrix")
                cmtx = confusion_matrix(y_test, y_pred)
                print(cmtx)
                #acc = (cmtx[0][0] + cmtx[1][1])/(cmtx[0][0] + cmtx[1][1] + cmtx[0][1] + cmtx[1][0])
                #print('Accuracy: %.2f' % acc)
                print("\nKappa Score:")
                print('%.2f' % cohen_kappa_score(y_test, y_pred))
                print('\nt-test: ')
                print(stats.ttest_ind(y_test, y_pred))

def boost(X, y):
        clf = AdaBoostClassifier(n_estimators=100)
        scores = cross_val_score(clf, X, y)
        print('AdaBoost', scores.mean())

def vote(X, y):
        clf1 = GaussianNB()
        clf2 = RandomForestClassifier()
        clf3 = LinearSVC()
        voter = VotingClassifier(estimators = [('gb', clf1),('rf', clf2),('svm', clf3)], voting = 'hard')
        voter = voter.fit(X, y)
        print(voter.predict(X))
        print(voter.score(X, y))

if __name__ == "__main__":
        data = make_stronger_dataset()
        for n in range(100, 3001, 100):
                print("****NUMBER: %i ****" % n)
                dct = make_samples(data, n)
                print("\n****BAG OF WORDS****\n")
                X, y, svm_bow = make_bag_of_words(dct)
                print("\n****TFIDF****\n")
                X2, y2, svm_tfidf = make_tfidf(dct)
                print("\n****ADA BOOST CLASSIFIER BAG****\n")
                boost(X,y)
                print("\n****ADA BOOST CLASSIFIER TFIDF****\n")
                boost(X2,y2)
                print("\n****CROSS FOLD BAG****\n")
                make_crossfold(X, y)
                print("\n****CROSS FOLD TFIDF****\n")
                make_crossfold(X2, y2)
                print("\n****VOTE BAG*****\n")
                vote(X,y)
                print("\n****VOTE TFIDF*****\n")
                vote(X2,y2)
                print("\n**** TRAIN & TEST BAG ****\n")
                a,b,c,d = train_test(X,y)
                classify(a,b,c,d)
                print("\n**** TRAIN & TEST TFIDF ****\n")
                a,b,c,d = train_test(X2,y2)
                classify(a,b,c,d)
                print("\n****K FOLD BAG****\n")
                a,b,c,d = k_fold(X, y)
                classify(a,b,c,d)
                print("\n****K FOLD TFIDF****\n")
                a,b,c,d = k_fold(X, y)
                classify(a,b,c,d)
