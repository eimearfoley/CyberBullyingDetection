from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import string
import pickle

vectorizer = CountVectorizer()
#output = open('results.csv', 'w')
#writer = csv.writer(output, delimiter=';', quotechar='"')
min_max_scaler = preprocessing.MinMaxScaler()

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
        bag_of_words = bag_of_words.toarray() # from scarse to numpy array
        bag_of_words = min_max_scaler.fit_transform(bag_of_words) #feature scaling

        # pickling of 2D array
        pickle.dump(bag_of_words, open('results.pkl', 'wb'))
        Z = pickle.load(open('results.pkl', 'rb'))

        # splitting 2D array into test and train data
        Z_train, Z_test = train_test_split(Z, test_size=0.25, random_state=42)
        print(Z_train,'split' ,Z_test)

        #print(bag_of_words)
        #np.savetxt('npresults.csv', bag_of_words, delimiter=' ')
    #words = vectorizer.get_feature_names()
    #data = [['filename', [word for word in words], 'class']]
    #writer.writerows(data)
    # Sum up the counts of each vocabulary word
    #dist = np.sum(bag_of_words, axis=0)

    # For each, print the vocabulary word and the number of times it 
    # appears in the training set
    """for tag, count in zip(words, dist):
        print(count, tag)
    for i, row in enumerate(bag_of_words):
        print(i, row)
        #print(row.get_feature_names())
           #if row[owen_loc] != 0:
                #print(file[i])"""
#output.close()




