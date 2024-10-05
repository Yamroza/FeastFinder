from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
from collections import defaultdict, Counter
import regex as re

class ModelTrainer:
    def __init__(self, file_name) -> None:
        self.x_set = []
        self.y_set = []

        with open(file_name, 'r') as file:
            for line in file:
                self.y_set.append(line.split()[0])
                self.x_set.append(" ".join(line.split()[1:]).lower())
    
        self.x_train, self.x_test = self.x_set[int(len(self.x_set)*0.15):], self.x_set[:int(len(self.x_set)*0.15)]
        self.y_train, self.y_test = self.y_set[int(len(self.y_set)*0.15):], self.y_set[:int(len(self.y_set)*0.15)]

        x_train = self.x_train
        x_train_no_dupl = set(self.x_train)
        x_test_no_dupl = set(self.x_test)

        y_train = pd.Series(self.y_train)
        y_test = pd.Series(self.y_test)
        y_train, self.word_list = pd.factorize(self.y_train)
        y_test = pd.factorize(self.y_test)[0]

        self.x_train_bow = []
        self.x_test_bow = []
        self.y_train_cat= self.__to_cat(y_train, 15)
        self.y_test_cat = self.__to_cat(y_test, 15)

        self.keyword_dict = {
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

        #Can maybe be deleted
        one_hot_enc_dict = {
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
        
        model = None
    def __to_cat(self, data, num_tokens):
        encoder = keras.layers.CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
        y_train_cat = encoder(data)
        return y_train_cat
    
    def __max_words(self, query: str, word_list: list, y_train: np.array) -> str:
        return word_list[np.bincount(y_train).argmax()]

    def __to_vector(self, list_of_words: list, model = "doc2vec_model"):
        try:
            model = Doc2Vec.load(model)
            vectorized = [model.infer_vector(word_tokenize(doc.lower())) for doc in list_of_words]
        except Exception as e: 
            print("please train a model first, using doc2vec_train")
            print(e)
        return np.array(vectorized)
    
    def baseline(self, x_set: list, y_set: list, y_train: list):
        """
        The baseline accuracy is calculated as the ratio of correct predictions to the total number of test examples.
        """
        x_test_basic = x_set[:int(len(x_set)*0.15)]
        y_test_basic = y_set[:int(len(y_set)*0.15)]

        suma = 0
        for x, y in zip(x_test_basic, y_test_basic):
            if self.__max_words(x, self.word_list, y_train) == y:
                suma += 1

        dummy_acc = suma / len(x_test_basic)
        return dummy_acc
    
    def rule_based_baseline(self, sentence: str, keyword_dict: dict):
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
        for keywords in self.keyword_dict.values():  # Iterating over categories
            for key in keywords:                # Iterating over keywords in the categories
                for word in words:              # Iterating over words in the sentence
                    if key == word:
                        count[i] += 1
            i += 1
        max_keywards = max(count)                           # Finding number of the most maching keywords within a category
        index_of_max_keywords =  count.index(max_keywards)  # Finding the index of the category with the most matching keywords
        result[index_of_max_keywords] = 1                   # Encoding the category 
        return result 

    #TODO double check: do we really vectorize docs (sentences), or do we vectorize words later on
    def doc2vec_train(self):
        """
        We use doc2vec based on the nltk punkt wordset to be able to vectorize the docs (sentences)
        """    
        tagged_x_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(self.x_train)]
        self.model = Doc2Vec(vector_size=50,
                        min_count=2, epochs=50)
        #print(tagged_x_data)
        self.model.build_vocab(tagged_x_data)
        self.model.train(tagged_x_data,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs)
        self.model.save("doc2vec_model")

    def __tokenization(self, data):
        tokenized = []
        for sent in data:
            tokens = word_tokenize(sent)
            token_sent = [w.lower() for w in tokens if w.isalpha() ]
            tokenized.extend(token_sent)
        tokenized = sorted(list(set(tokenized)))
        return tokenized

    def __word_extraction(self, sentence, ignore):    
        words = re.sub("[^\w]", " ",  sentence).split()    
        cleaned_text = [w.lower() for w in words if w not in ignore]    
        return cleaned_text


    def __generate_vec(self, data, vocab, words_to_ignore):
        vectors =[]
        for sentence in data:                
            bag_vector = np.zeros(len(vocab))  
            words = self.__word_extraction(sentence, words_to_ignore)    
            for w in words:            
                for i,word in enumerate(vocab):               
                    if word == w:                     
                        bag_vector[i] += 1
            vectors.append(bag_vector)
        return vectors

    def generate_bow(self, train, test, words_to_ignore=[]):
        """
        Bag of words is one of the tokenization methods. The use can specify words to ignore, like 'a', 'the', etc, 
        to improve the performance. 
        """       
        vocab = self.__tokenization(train)  
        self.x_train_bow = self.__generate_vec(train, vocab, words_to_ignore)
        self.x_test_bow = self.__generate_vec(test, vocab, words_to_ignore)
        return self.x_train_bow, self.x_test_bow

    def __plot(history, epochs):
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
    

    #TODO make it work with bow and lr and the different options - shape etc
    def train_model(self, plot_model_results: bool, d2v_model = "doc2vec_model", embedding = "BOW", model_type = "LR", epochs = 12):
        """
        This method trains a model based on either Bag of Words (BOW) or Doc2Vec (W2V) embeddings and a model type (logistic regression or feedforward neural network).
        The method preps the input data, defines the model architecture, and trains it using the specified parameters. 
        It saves the trained model and optionally plots the results.
        """
        model = keras.Sequential()
        
        if embedding == "BOW":
            if self.x_train_bow == []:
                self.x_train_bow, self.x_test_bow = self.generate_bow(self.x_train, self.x_test)
            x_train = np.array(self.x_train_bow)
            y_train = self.y_train_cat
            x_test = np.array(self.x_test_bow)
            y_test =  self.y_test_cat
            model.add(keras.Input(shape=(731,)))
        elif embedding == "D2V":
            x_train = self.__to_vector(self.x_train, d2v_model)
            y_train = self.y_train_cat
            x_test = self.__to_vector(self.x_test, d2v_model)
            y_test =  self.y_test_cat
            model.add(keras.Input(shape=(50,)))

        
        if model_type == "FFN":
            model.add(keras.layers.Dense(256, activation="relu"))
        
        model.add(keras.layers.Dense(15, activation="softmax"))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(x_train, y_train, batch_size=64,
        epochs=epochs, verbose=1,validation_data=(x_test, y_test))
        model.save(model_type + "_" + embedding + "_model")
        if plot_model_results:
            self.__plot(history, epochs)


    def prediction(self, Xnew, model, input_type="BOW", verbose = False):
        """
        This loads a model and predicts the category of the input based on this model.
        It can print its output using the verbose 
        """
        try: 
            model = keras.models.load_model(model)
        except Exception as e: 
            print("please train a model first, using train_model")
            print(e)
        
        if input_type == "BOW":
            vocab = self.__tokenization(self.x_train) 
            X_new_vector = self.__generate_vec(Xnew, vocab, words_to_ignore=[])
            X_new_vector = np.array(X_new_vector)
        elif input_type == "D2V":
            X_new_vector = self.__to_vector(Xnew)
            X_new_vector = np.array(X_new_vector)

        ynew = model.predict(X_new_vector, verbose=False)

        for i in range(len(Xnew)):
            outcome =  self.word_list[np.argmax(ynew[i])]
            if verbose:
                print("X=%s, Predicted=%s" % (Xnew[i], outcome))
                print(outcome)
        return self.word_list[np.argmax(ynew[i])]


model_trainer = ModelTrainer("data/dialog_acts.dat")
#model_trainer.doc2vec_train()
model_trainer.train_model(False, embedding = "D2V",model_type="FFN")
model_trainer.train_model(False, embedding = "D2V",model_type="LR")

#print(model_trainer.prediction(["ues"]))

model_trainer.prediction(["yes"], "LR_D2V_model", input_type="D2V", verbose=True)
model_trainer.prediction(["yes"], "FFN_D2V_model", input_type="D2V", verbose=True)
model_trainer.prediction(["yes"], "LR_BOW_model", input_type="BOW", verbose=True)
model_trainer.prediction(["yes"], "FFN_BOW_model", input_type="BOW", verbose=True)
# For testing purposes

# positive = 0
# for x in range(len(x_test)): # Iterating over sentences in the set
#     pred_coded = rule_based_baseline(x_test[x], keyword_dict) # Calling the predicting function to get the coded predicted category
#     for key, value in result_dict.items(): # Decoding categories
#         if pred_coded == value:
#             pred = key
#     if pred == y_test_basic[x]: # Adding 1 to successful prediction count if the prediction matches data
#         positive += 1
# rate = positive / len(y_test_basic) # Calculating and displaying a success rate
# print('Rate is: ', rate)