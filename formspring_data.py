import os
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import csv

#dictionary contains the answets from the formspring data as 
#keys and the combination of bullying occurances in the messages"""
dct={}
ques = 2
ans = 3
ans1 = 5
ans2 = 8
ans3 = 11
bully1 = 7
bully2 =10
bully3 = 13

filehandle = open("result.txt", "w")
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

punctuation = set(string.punctuation)
texts = []

print('Appending texts')
for tweet in dct:
	# Strip punctuation
	tweet = ''.join(ch if ch not in punctuation else ' ' for ch in tweet)  
	tweet = ' '.join(tweet.split())  # Remove whitespace
	tweet = tweet.lower()  # Convert to lowercase
	texts.append(tweet)
print('Vectorirzer')
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(texts)

X = bag_of_words.toarray()
y = [1 if dct[f] else 0 for f in dct]

print("Training and testing")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#clf ={'RandomForest Classifier':AdaBoostClassifier(RandomForestClassifier()),'SVM Classifier':AdaBoostClassifier(svm.SVC())}
clf ={"AdaBoostClassifier with Bagging":AdaBoostClassifier(BaggingClassifier())}
#clf={"Bagging classifier":BaggingClassifier()}
print()
print("Classifiers")
for clas in clf:
	clf[clas].fit(X_train, y_train)
	y_pred = clf[clas].predict(X_test)
	print("***"+clas+" Report***")
	print(classification_report(y_test, y_pred))
	print("***Confusion Matrix***")
	print(confusion_matrix(y_test, y_pred))
	print(cohen_kappa_score(y_test, y_pred))
	print()

