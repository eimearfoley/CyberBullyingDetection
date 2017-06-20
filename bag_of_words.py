from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import string
import numpy as np
import os

dataset_loc = './textpacket1'
records = [os.path.join(dataset_loc, f) for f in os.listdir(dataset_loc)] # dataset 
punctuation = set(string.punctuation)
outfile = open('outfile', 'w') # output file
min_max_scaler = preprocessing.MinMaxScaler() 

for file in records:
    text_list = []
    name = file
    to_write = {name: True, 'words':{}} # dict representation of file
    with open(file, 'r') as the_file:
        file_contents = the_file.read()  # Read file
        file_contents = ''.join(ch if ch not in punctuation else ' ' for ch in file_contents)  # Strip punctuation
        file_contents = ' '.join(file_contents.split())  # Remove whitespace
        file_contents = file_contents.lower()  # Convert to lowercase
        text_list.append(file_contents)
        vectorizer = CountVectorizer()
        bag_of_words = vectorizer.fit_transform(text_list) # counts word frequency and applies weights to words
        bag_of_words = bag_of_words.toarray() # converts from sparse matrix to 2D array
        #bag_of_words = min_max_scaler.fit_transform(bag_of_words) # normalization
        words_lst = vectorizer.get_feature_names() # all feature names in list
 
        # writes feature name and frequency to dict
        dist = np.sum(bag_of_words, axis=0)
        for tag, count in zip(words_lst, dist):
            to_write['words'][tag] = count
            
        outfile.write(str(to_write) + '\n')
outfile.close()
