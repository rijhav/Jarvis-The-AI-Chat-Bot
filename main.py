import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
from tensorflow.python.framework import ops
import random
import json
import pickle
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
import tensorflow as tf


from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)
#try:
    #with open("data.pickle","rb") as f:
        #words, labels , training, output = pickle.load(f)
#except:
    words = []
    labels =[]
    docs_x = []
    docs_y = []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)  #breaking the patten into the root words
            words.extend(wrds) #adding to the list n number of times
            docs_x.append(wrds) #adding one element to the end of the list at a time
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1) # we are adding 0||1 to indicate whether the word exists or not
            else:
                bag.append(0)

        output_row = list(out_empty)
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = np.array(output)

    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)

ops.reset_default_graph()

#constructing the neural network
net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net, 8)# number of nodes being used
net = tflearn.fully_connected(net, 8)# another structure where the 8 nodes meet another set of 8 nodes
net = tflearn.fully_connected(net, len(output[0]),activation="softmax")# the last structure of the nerual network (last level)
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training,output,n_epoch=500, batch_size=8, show_metric= True)
    model.save("model.tflearn")



def bagOfWords(s, words):
    bag = [0 for _ in range(len(words))] #a vector of 0 & 1 where 1 indicates if the word exists and 0 if not
    s_words = nltk.word_tokenize(s)# breaking the words up
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag) # sets the vector into a neat array

def menu(Veg, Special,Emenu):
    if (Veg and Special and Emenu != True):
        menu =["Veg Burger R15" , "Veg Burger with larger chips for R25"]
    elif (Veg and Special != True and Emenu != True):
        menu = ["Veg Breyani R20", "Chips medium R15", "Chips large R25", "Beans and puri R20", "2 Dozen roti R50"]
    elif (Veg != True and Special and Emenu != True):
        menu = ["Chicken burgers R20", "Mutton burgers R30", "Mutton burger + medium Chips R40", "Chicken and mayo toasted sandwhich R15"]
    elif (Veg != True and Special != True and Emenu != True):
        menu = ["Mutton pasta large R25 ", "Chicken Tikka pizza medium R30"]
    elif (Emenu and Special != True):
        menu = ["Breyani R20", "Chips medium R15", "Chips large R25", "Beans and puri R20", "2 Dozen roti R50",
                "Mutton pasta R25 large", "Chicken Tikka pizza medium R30"]
    elif (Emenu and Special):
        menu = ["Veg Burger R15" , "Veg Burger with larger chips for R25",
                "Chicken burgers R20", "Mutton burgers R30", "Mutton burger + medium Chips R40", "Chicken and mayo toasted sandwhich R15"]

    return menu



def Specials(name,Veg,Emenu,Special):
    if (Special == False):
        specials = ['yes', 'yes i would like to know', 'what are the specials']
        print("Jarvis: Would you like to know the specials?")
        special = input(name)
        if (special in specials):
            Special = True
    print("When you are ready to order just type what you want and type done to confirm your order")
    print(menu(Veg, Special,Emenu))  # used to display the menu on request

def DetermineCost(choices):
    return int(choices)

def chat():
    #this is like the main()
    #code that runs the program
    name = input('What is your name? ') + ": "
    print("Start talking with Dr. Jarvis The AI bot! (Type quit to stop)")

    count = 0

    total = 0

    selection = []

    Special = False

    while True:
        Veg = False

        Order = False
        inp = input(name) #inp takes in the users message
        if (inp.lower() == "quit"):
            break #terminate the program midway if we meet any bugs

        results = model.predict([bagOfWords(inp.lower(),words)])[0]# from here we do the pridictions of the words and the responses
        results_index = numpy.argmax(results)# result_index gets the largest value from the prediction in probability form
        tag = labels[results_index]
        print(results[results_index]) #prints the accuracy of the results
        if (results[results_index] > 0.80): #0.85 is the TH that indicates how accurate we want our responses to be
            for tg in data['intents']:
                if (tg['tag'] == tag):
                    if (tag == 'food'):
                        Special = True

                        tg['tag'] = 'menu'
                        responses = [" are you veg or non-veg?"]
                    else:
                        responses = tg['responses']

            if (tag == 'goodbye'): # i used the goodbye so that the bot knows to shut itself off after giving a response
                print("Jarvis: ", random.choice(responses))
                exit()
            if (tag == 'preferanceVeg'):
                Veg = True
                Specials(name,Veg,False,Special)
                Special = False
            if (tag == 'preferanceNonVeg'):
                Veg = False
                Specials(name,Veg,False,Special)
                Special = False
            if (tag == 'Entire_Menu'):
                Emenu = True
                Specials(name,Veg,Emenu,Special)
                Special = False
            try:
                total = total + DetermineCost(random.choice(responses))
                selection.append(inp)

            except:
                print("Jarvis: ", random.choice(responses))

        else:
            if (inp.lower() == "done"):
                print("Your order is: ")
                print(selection)
                print("Your total is: ")
                print("R",total)
            else:
                print("Jarvis: ","Sorry I didnt understand that...") #if the probability values < TH then the bot doesnt
                #recgonise the message

chat()