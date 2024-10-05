# %%
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
x_set = []
y_set = []

with open("data/dialog_acts.dat", 'r') as file:
    for line in file:
        y_set.append(line.split()[0])
        x_set.append(" ".join(line.split()[1:]).lower())

y_set[0], x_set[0]
    

# %%
#maybe split it in train, test and dev?

x_train, x_test = x_set[int(len(x_set)*0.15):], x_set[:int(len(x_set)*0.15)]
y_train, y_test = y_set[int(len(y_set)*0.15):], y_set[:int(len(y_set)*0.15)]
len(x_train), len(x_test), len(y_train), len(y_test)

x_train_no_dupl = set(x_train)
x_test_no_dupl = set(x_test)
print(len(x_train))
print(len(x_test_no_dupl))

y_train = pd.Series(y_train)
y_test = pd.Series(y_test)
y_train, word_list = pd.factorize(y_train)
y_test = pd.factorize(y_test)[0]

def to_cat(data, num_tokens):
    encoder = keras.layers.CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
    y_train_cat = encoder(data)
    return y_train_cat

y_train_cat= to_cat(y_train, 15)
y_test_cat = to_cat(y_test, 15)


# %% [markdown]
# ## Dummy baseline

# %%
def dummy_baseline(query: str, word_list: list, y_train: np.array) -> str:
    return word_list[np.bincount(y_train).argmax()]

# %%
dummy_baseline('hi', word_list, y_train)

# %%
x_test_basic = x_set[:int(len(x_set)*0.15)]
y_test_basic = y_set[:int(len(y_set)*0.15)]

suma = 0
for x, y in zip(x_test_basic, y_test_basic):
    if dummy_baseline(x, word_list, y_train) == y:
        suma += 1

dummy_acc = suma / len(x_test_basic)
dummy_acc

# %% [markdown]
# ## Rule-based baseline

# %%
def rule_based_baseline(sentence: str, keyword_dict: dict):
    """
    Function returns a encoded category, encoding matches what a neural network would return.
    This is done for ease of integration with other systems that may be built during this project.
    :param sentence: sentence
    :param keyword_dict: category-keyword
    :return: 
    """
    result = [0] * 15
    count = [0] * 15
    words = sentence.split()
    i = 0
    for keywords in keyword_dict.values():  # Iterating over categories
        for key in keywords:                # Iterating over keywords in the categories
            for word in words:              # Iterating over words in the sentence
                if key == word:
                    count[i] += 1
        i += 1
    max_keywards = max(count)                           # Finding number of the most maching keywords within a category
    index_of_max_keywords =  count.index(max_keywards)  # Finding the index of the category with the most matching keywords
    result[index_of_max_keywords] = 1                   # Encoding the category 
    return result 

# %%
# Category-keyword dictionary

keyword_dict = {
    'ack': ['kay', 'okay', 'good', 'fine'],
    'affirm': ['yes', 'right', 'correct', 'yeah', 'ye', 'right', 'correct', 'perfect'],
    'bye': ['good', 'bye'],
    'confirm': ['does', 'is', 'it'],
    'deny': ['wrong', 'want', 'dont'],
    'hello': ['hi', 'hello', 'im', 'looking'],
    'inform': ['any', 'food', 'dont', 'care', 'expensive', 'moderate', 'cheap', 'east', 'west', 'north', 'south', 'centre', 'town', 'area', 'im', 'need', 'restaurant', 'looking'],
    'negate': ['no'],
    'null': ['unintelligible', 'noise', 'what', 'uh', 'sil', 'laughing'],
    'repeat': ['repeat', 'back', 'go', 'again'],
    'reqalts': ['else', 'next', 'how', 'about', 'any', 'anything', 'is', 'there', 'other'],
    'reqmore': ['more'],
    'request': ['type', 'phone', 'number', 'address', 'post', 'code', 'could', 'what', 'is', 'the', 'type', 'whats', 'may', 'i'],
    'restart': ['start', 'over', 'reset'],
    'thankyou': ['thank', 'you', 'good', 'bye', 'goodbye'],
}

# %%
# Category-code dictionary

result_dict = {
    'ack': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'affirm': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'bye': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'confirm': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'deny': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'hello': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'inform': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'negate': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'null': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'repeat': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'reqalts': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'reqmore': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'request': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'restart': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'thankyou': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}

# %%
# Making predictions and calculating the success rate

positive = 0
for x in range(len(x_test)): # Iterating over sentences in the set
    pred_coded = rule_based_baseline(x_test[x], keyword_dict) # Calling the predicting function to get the coded predicted category
    for key, value in result_dict.items(): # Decoding categories
        if pred_coded == value:
            pred = key
    if pred == y_test_basic[x]: # Adding 1 to successful prediction count if the prediction matches data
        positive += 1
rate = positive / len(y_test_basic) # Calculating and displaying a success rate
# %%
# Doc2Vec (Sentences to vector)
# We use this to convert the whole input paragraph to a vector
# Upon reading the exercise again, they seem to suggest using bag-of-words. However, I think this is better. Lets discuss with the TA!

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

tagged_x_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(x_train)]
model = Doc2Vec(vector_size=50,
                min_count=2, epochs=50)
#print(tagged_x_data)
model.build_vocab(tagged_x_data)
model.train(tagged_x_data,
            total_examples=model.corpus_count,
            epochs=model.epochs)

def to_vector(list_of_words):
    vectorized = [model.infer_vector(word_tokenize(doc.lower())) for doc in list_of_words]
    return np.array(vectorized)


# %%
from collections import defaultdict, Counter
import regex as re
def tokenization(data):
    tokenized = []
    for sent in data:
        tokens = word_tokenize(sent)
        token_sent = [w.lower() for w in tokens if w.isalpha() ]
        tokenized.extend(token_sent)
    tokenized = sorted(list(set(tokenized)))
    return tokenized

def word_extraction(sentence):    
    #ignore = ['a', "the", "is"]
    ignore = []    
    words = re.sub("[^\w]", " ",  sentence).split()    
    cleaned_text = [w.lower() for w in words if w not in ignore]    
    return cleaned_text


def generate_vec(data, vocab):
    vectors =[]
    for sentence in data:                
        bag_vector = np.zeros(len(vocab))  
        words = word_extraction(sentence)    
        for w in words:            
            for i,word in enumerate(vocab):               
                if word == w:                     
                    bag_vector[i] += 1
        vectors.append(bag_vector)
    return vectors

def generate_bow(train, test):       
    vocab = tokenization(train)  
    train =generate_vec(train, vocab)
    test = generate_vec(test, vocab)
    return train, test

x_train_bow, x_test_bow = generate_bow(x_train, x_test)
#bag_of_words(x_train_tokenized, vocab, word_with_index)

# %%
document_vectors = to_vector(x_train)
x_test_vector = to_vector(x_test)

# %%
def plot(history, epochs):
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label = "loss")
    ax.plot(history.history["val_loss"], label = "Valditation loss")
    ax.set_title(f"Loss in {epochs} epochs")
    fig.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(history.history["accuracy"], label = "accuracy")
    ax2.plot(history.history["val_accuracy"], label = "Validation accuracy")
    ax2.set_title(f"Accuracy in {epochs} epochs")
    fig2.legend()

# %%
#FNN with BOW

epochs = 12
model_FNN = keras.Sequential()
model_FNN.add(keras.Input(shape=(731,)))
model_FNN.add(keras.layers.Dense(256, activation="relu"))
model_FNN.add(keras.layers.Dense(15, activation="softmax"))

model_FNN.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_FNN.fit(np.array(x_train_bow), y_train_cat, batch_size=64,
epochs=epochs, verbose=1,validation_data=(np.array(x_test_bow), y_test_cat))

plot(history, epochs)

# %%
#FNN with word embeddings
epochs = 12
model_FNN = keras.Sequential()
model_FNN.add(keras.Input(shape=(50,)))
model_FNN.add(keras.layers.Dense(256, activation="relu"))
model_FNN.add(keras.layers.Dense(15, activation="softmax"))

model_FNN.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_FNN.fit(document_vectors, y_train_cat, batch_size=64,
epochs=12, verbose=1,validation_data=(x_test_vector, y_test_cat))

plot(history, epochs)

# %%
# Logistic regression with BOW
model_LR = keras.Sequential()
model_LR.add(keras.Input(shape=(731,)))
model_LR.add(keras.layers.Dense(15, activation="softmax"))

model_LR.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_LR.fit(np.array(x_train_bow), y_train_cat, batch_size=64, epochs=12, verbose=1,
                       validation_data=(np.array(x_test_bow), y_test_cat))

plot(history, epochs)

# %%
# Logistic regression with word embeddings
model_LR = keras.Sequential()
model_LR.add(keras.Input(shape=(50,)))
model_LR.add(keras.layers.Dense(15, activation="softmax"))

model_LR.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_LR.fit(document_vectors, y_train_cat, batch_size=64, epochs=12, verbose=1,
                       validation_data=(x_test_vector, y_test_cat))

plot(history, epochs)

# %%
Xnew = ["yes", "thai food", "what is the cheapest restaurant in london?"]
# make a prediction
def prediction(Xnew, verbose = False):
    X_new_vector = to_vector(Xnew)
    X_new_vector = np.array(X_new_vector)
    ynew = model_LR.predict(X_new_vector, verbose=False)
# show the inputs and predicted outputs
    for i in range(len(Xnew)):
        outcome =  word_list[np.argmax(ynew[i])]
        if verbose:
            print("X=%s, Predicted=%s" % (Xnew[i], outcome))
            print(outcome)
    return word_list[np.argmax(ynew[i])]


# %%
