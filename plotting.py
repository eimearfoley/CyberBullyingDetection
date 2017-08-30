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

random_seed = 41

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
    with open('formspring1.csv') as csvfile:
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

def train_test(X, y, svm_bow, clas, classifier):
    print("Training and testing")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split data for train and test
    print("\nClassifiers")
    if clas == "SVM Classifier":
        bag = svm_bow.toarray() # use normalized data for SVM classifier
        bag_train, bag_test, y_train, y_test = train_test_split(bag, y, test_size=0.2)
        classifier.fit(bag_train, y_train)
        y_pred = clf[clas].predict(bag_test)
    else:
        classifier.fit(X_train, y_train)
        y_pred = clf[clas].predict(X_test)
    return results(y_test, y_pred, clas)

def k_fold(X, y, svm_bow, clas, classifier):
    y = np.array(y)
    skf = StratifiedKFold(n_splits=10)
    if clas == "SVM Classifier":
        bag = svm_bow.toarray() # use normalized data for SVM classifier
        for train_index, test_index in skf.split(bag, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    else:
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    clf[clas].fit(X_train, y_train)
    y_pred = clf[clas].predict(X_test)
    return results(y_test, y_pred, clas)

def cross_fold(X, y, svm_bow, clas, classifier):
    if clas == "SVM Classifier":
        X = svm_bow.toarray() # use normalized data for SVM classifier
    scores = cross_val_score(classifier, X, y, cv=10, scoring='f1')
    print("Average: %.2f" % (sum(scores) / float(len(scores))))

def results(y_test, y_pred, clas):
    # print results and scores of all classifiers
    print("\n"+clas+" Report")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix")
    cmtx = confusion_matrix(y_test, y_pred)
    print(cmtx)
    acc = (cmtx[0][0] + cmtx[1][1])/(cmtx[0][0] + cmtx[1][1] + cmtx[0][1] + cmtx[1][0])
    print('Accuracy: %.2f' % acc)
    fn = cmtx[0][1]/(cmtx[0][1] + cmtx[1][0])
    print('False Negative: %.2f' % fn)
    print("\nKappa Score:")
    kappa = cohen_kappa_score(y_test, y_pred)
    print('%.2f' % kappa)
    print('\nt-test: ')
    print(stats.ttest_ind(y_test, y_pred))
    return kappa, acc, fn

if __name__ == "__main__":
    data = make_dataset()
    clf ={'Linear SVM Classifier': LinearSVC(random_state=random_seed), 'RandomForest Classifier': RandomForestClassifier(random_state=random_seed), 'GaussianNB': GaussianNB(), 'Decision Tree': tree.DecisionTreeClassifier(random_state=random_seed)}
    for clas in clf:
        print("CLASSIFIER: %s" % clas)
        false_tt_b = []
        false_tt_t = []
        false_kf_b = []
        false_kf_t = []
        k_scores = []
        a_scores = []
        n_range = []
        kt_s = []
        at_s = []
        kfb_ks = []
        kfb_as = []
        kft_ks = []
        kft_as = []
        for n in range(100, 3001, 100):
            print("****NUMBER: %i ****" % n)
            dct = make_samples(data, n)    
            print("\n****BAG OF WORDS****\n")
            X, y, svm_bow = make_bag_of_words(dct)
            print("\n****TFIDF****\n")
            X2, y2, svm_tfidf = make_tfidf(dct)
            print("\n**** TRAIN & TEST BAG ****\n")
            classifier = clf[clas]
            k, a, tbfn = train_test(X,y, svm_bow, clas, classifier)
            print("\n**** TRAIN & TEST TFIDF ****\n")
            kt, at, ttfn = train_test(X2,y2, svm_tfidf, clas, classifier)
            print("\n*** CROSS FOLD VALIDATION ***\n")
            print(cross_fold(X, y, svm_bow, clas, classifier))
            print("\n**** K-FOLD BAG ****\n")
            kfbk, kfba, kfbfn = k_fold(X, y, svm_bow, clas, classifier)
            print("\n**** K-FOLD TFIDF ****\n")
            kftk, kfta, kftfn = k_fold(X2, y2, svm_tfidf, clas, classifier)
            false_tt_b += [tbfn]
            false_tt_t += [ttfn]
            false_kf_b += [kfbfn]
            false_kf_t += [kftfn]
            kfb_ks += [kfbk]
            kfb_as += [kfba]
            kft_ks += [kftk]
            kft_as += [kfta]
            k_scores += [k]
            a_scores += [a]
            kt_s += [kt]
            at_s += [at]
            n_range += [n]
        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax = fig.add_subplot(311)
        plt.title('%s' % clas)
        plt.xlabel('Data Subsets')
        plt.ylabel('Kappa & Accuracy Scores')
        plt.plot(n_range, k_scores, color='red', linestyle='solid', label='Bag Kappa')
        plt.plot(n_range, a_scores, color='blue', linestyle='solid', label='Bag Accuracy')
        plt.plot(n_range, kt_s, color='green', linestyle='solid', label='TFIDF Kappa')
        plt.plot(n_range, at_s, color='yellow', linestyle='solid', label='TFIDF Accuracy')
        plt.legend()
        ax2 = fig.add_subplot(312)
        plt.title('%s' % clas)
        plt.xlabel('Data Subsets')
        plt.ylabel('Kappa & Accuracy Scores')
        plt.plot(n_range, kfb_ks, 'ro-', label='K-Fold Bag Kappa')
        plt.plot(n_range, kfb_as, 'bo-',  label='K-Fold Bag Accuracy')
        plt.plot(n_range, kft_ks, 'go-', label='K-Fold TFIDF Kappa')
        plt.plot(n_range, kft_as, 'yo-',  label='K-Fold TFIDF Accuracy')
        plt.legend()
        ax3 = fig.add_subplot(313)
        plt.title('%s' % clas)
        plt.xlabel('Data Subsets')
        plt.ylabel('False Negative Accuracy')
        plt.plot(n_range, false_tt_b, 'ro-', label='TT Bag Fn')
        plt.plot(n_range, false_kf_b, 'co-', label='K-Fold Bag Fn')
        plt.plot(n_range, false_tt_t, 'mo-', label='TT TFIDF Fn')
        plt.plot(n_range, false_kf_t, 'o-', label='K-Fold TFIDF Fn')
        plt.legend()
        for xy in zip(n_range, k_scores):
            ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        for xy in zip(n_range, a_scores):
            ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        for xy in zip(n_range, kt_s):
            ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        for xy in zip(n_range, at_s):
            ax.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        for xy in zip(n_range, kfb_ks):
            ax2.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        for xy in zip(n_range, kfb_as):
            ax2.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        for xy in zip(n_range, kft_ks):
            ax2.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        for xy in zip(n_range, kft_as):
            ax2.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        for xy in zip(n_range, false_tt_b):
            ax3.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        for xy in zip(n_range, false_kf_b):
            ax3.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        for xy in zip(n_range, false_tt_t):
            ax3.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        for xy in zip(n_range, false_kf_t):
            ax3.annotate('%.2f' % xy[1], xy=xy, textcoords='data')
        plt.grid()
    plt.show()
