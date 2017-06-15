import os
import re
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer

folder = [f for f in listdir('../Insight/2017/textpacket1')]
vectorizer = CountVectorizer()
output = open('results.txt', 'w')

for file in folder:
    print(file)
    print('here')
    file = open('../Insight/2017/textpacket1/' + file, 'r')
    wordslst = []
    for word in file:
        #re.sub('  ', '', word)
        word = word.strip('\n')
        word = word.strip('\t')
        word = word.strip('        ')
        wordslst += [word]
    print(wordslst)
    bag_of_words = vectorizer.fit(wordslst)
    bag_of_words = vectorizer.transform(wordslst)
    print(bag_of_words)
    output.write(str(bag_of_words) + "\n")
    file.close()
    print(vectorizer.vocabulary_.get('Owen'))
    exit()
output.close()
