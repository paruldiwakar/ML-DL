#!/usr/bin/env python
# coding: utf-8

# # Agenda
# - Libraries & Data
# - Initialising Chatbot Training
# - Building the Deep Learning Model
# - Building Chatbot GUI
# - Running Chatbot
# - Conclusion

# In[25]:


import nltk
from nltk.stem import WordNetLemmatizer    
import json 
import pickle
import numpy as np
import random


# In[24]:


nltk.download('punkt')
nltk.download('wordnet')


# In[26]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD


# ## Initializing Chatbot Training

# In[59]:


words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


# In[60]:


#intents


# In[61]:


for intent in intents.get('intents'):
    for pattern in intent.get('patterns'):
       # take each word and tokenize it
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # adding documents
        documents.append((word,intent['tag']))
        
        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[62]:


lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]

classes = sorted(list(set(classes)))


# In[63]:


print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


# ## Preparing Training Data

# In[68]:


'''
We’re creating a giant nested list which contains bags of words for each of our documents.
We have a feature called output_row which simply acts as a key for the list.
We then shuffle our training set and do a train-test-split, with the patterns being the X variable and the intents being the Y variable.
'''

# initializing training data
training = []
output_empty = [0]*len(classes)

for doc in documents:
    #initialising training data
    bag = []
    #list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    
    training.append([bag,output_row])
    
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# ## Building the Deep Learning Model

# In[72]:


'''
Create model - 3 layers.
First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
equal to number of intents to predict output intent with softmax.
The point of this network is to be able to predict which intent to choose given some data.
'''

model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))
("")
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True )
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")


# In[ ]:





# In[ ]:





# In[ ]:




