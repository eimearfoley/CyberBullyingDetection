import os
import re
import string
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer

def extract_words(s):
    return [re.sub('^[{0}]+|[{0}]+$'.format(string.punctuation), '', w).lower() for w in s.split()]

folder = [f for f in listdir('../Insight/2017/textpacket1')]
vectorizer = CountVectorizer()
output = open('results.txt', 'w')

for file in folder:
    print(file)
    file = open('../Insight/2017/textpacket1/' + file, 'r')
    wordslst = []
    for line in file:
        words = extract_words(line)
        wordslst += words
    print(wordslst)
    bag_of_words = vectorizer.fit(wordslst)
    bag_of_words = vectorizer.transform(wordslst)
    print(bag_of_words)
    output.write(str(bag_of_words) + "\n")
    file.close()
output.close()
