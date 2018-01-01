#For NLP

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# For Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random



def data_preprocessing(intents):
    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    
    classes = sorted(list(set(classes)))
    return (documents, classes, words)
    
def creating_training_data(intents):
    training = []
    output = []
    documents, classes, words = data_preprocessing(intents)
    # create an empty array for our output
    output_empty = [0] * len(classes)
    
    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
    
        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
    
        training.append([bag, output_row])
    
    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)
    
    # create train and test lists
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    return (train_x, train_y)

def trained_model(intents):
    train_x, train_y = creating_training_data(intents)
    # reset underlying graph data
    tf.reset_default_graph()
    # Build neural network
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)
    
    # Define model and setup tensorboard
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    # Start training (apply gradient descent algorithm)
    print("Thinking...")
    model.fit(train_x, train_y, n_epoch=1000, batch_size=8)
#    output = model.predict(input_sentence)
    return model
 
    










