'''
Tutorial from https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077
on creating chatbots with TensorFlow
'''



import nltk #Natural language processing in Python 

#Stemming a word chops it to its simplest form: i.e. cats, catty, and catnip should all stem to cat
from nltk.stem.lancaster import LancasterStemmer 
stemmer = LancasterStemmer()

# Stanard set of TensorFlow imports
import numpy as np
import tflearn #high-level API on top of tensor flow
import tensorflow as tf
import random #random number generator
import json #reads json

'''
Opens a JSON file and reads data into correct format
'''
with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# stem and lower each word and remove duplicates
# stemming is important to match intent;
# i.e. If you ask "How much does it cost," or "How much" answer should be same
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

# Tensorts are essentially nested lists of numbers
# Words need to be converted to have numerical values
# in order for tensorflow to work

# create our training data
training = []
output = []


# create an empty array for our output
# we want an array that includes 9 classes 
# ['goodbye', 'greeting', 'hours', 'mopeds', 'opentoday', 'payments', 'rental', 'thanks', 'today']
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
	print(doc)
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

