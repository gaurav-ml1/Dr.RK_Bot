from tflearn_model import data_preprocessing, trained_model
from nltk.stem.lancaster import LancasterStemmer
import nltk
import numpy as np
import json
import enchant
import random
#from enchant.checker import SpellChecker
import re
stemmer = LancasterStemmer()

def cleaning_input_data(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def data_encode(sentence, intents, show_details=False):
    documents, classes, words = data_preprocessing(intents)
    # tokenize the pattern
    sentence_words = cleaning_input_data(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))



with open('doctor_intent.json') as json_data:
    intents = json.load(json_data)

ERROR_THRESHOLD = 0.25

def classify(intents, sentence, model):
    
    _, classes, _= data_preprocessing(intents)
    # generate probabilities from the model
    
    results = model.predict([data_encode(sentence, intents)])[0]
    print (results[0])
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    print(classes)
    for r in results:
        print(r)
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def output_message(intents, sentence, model, userID='123', show_details=False):
    results = classify(intents, sentence, model)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return print("Dr. RK -> ",random.choice(i['responses']))

            results.pop(0)
            
#d = enchant.Dict("en_GB")
d = enchant.request_pwl_dict("english_dict.txt")
model = trained_model(intents)

while True:
    sentence = input("")
    if sentence.lower().strip() == "bye":
        break
    else:
        output = output_message(intents, sentence, model)


