from sklearn.feature_extraction.text import CountVectorizer
import os
import string
import csv

vectorizer = CountVectorizer()
output = open('results.csv', 'w')
writer = csv.writer(output, delimiter=';', quotechar='"')

if __name__ == '__main__':
    dataset_loc = '../Insight/2017/textpacket1'
    folder = [os.path.join(dataset_loc, f) for f in os.listdir(dataset_loc)]
    punctuation = set(string.punctuation)
    punctuation.remove("'")
    texts = []
    for file in folder:
        name = file
        #print(name)
        with open(file, 'r') as the_file:
                file_contents = the_file.read()  # Read file
                file_contents = ''.join(ch if ch not in punctuation else ' ' for ch in file_contents)  # Strip punctuation
                file_contents = ' '.join(file_contents.split())  # Remove whitespace
                file_contents = file_contents.lower()  # Convert to lowercase
                texts.append(file_contents)
        vectorizer = CountVectorizer()
        bag_of_words = vectorizer.fit_transform(texts)
        bag_of_words = bag_of_words.toarray()
        data = [[name, bag_of_words, 'Y']]
        writer.writerows(data)
        i_loc = vectorizer.vocabulary_.get('and')
        #print("and", i_loc)
        owen_loc = vectorizer.vocabulary_.get('owen')
        #print("owen", owen_loc)
    #print(vectorizer.get_feature_names())
    print(bag_of_words.shape)
    for i, row in enumerate(bag_of_words):
        print(i, row)
        #print(row.get_feature_names())
           #if row[owen_loc] != 0:
                #print(file[i])"""
output.close()
