import numpy as np
import regex as re
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from models import Model


def tokenization(data):
    tokenized = []
    for sent in data:
        tokens = word_tokenize(sent)
        token_sent = [w.lower() for w in tokens if w.isalpha()]
        tokenized.extend(token_sent)
    tokenized = sorted(list(set(tokenized)))
    return tokenized


def word_extraction(sentence):
    # ignore = ['a', "the", "is"]
    ignore = []
    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text


def generate_vec(data, vocab):
    vectors = []
    for sentence in data:
        bag_vector = np.zeros(len(vocab))
        words = word_extraction(sentence)
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1
        vectors.append(bag_vector)
    return vectors


def generate_bow(vocab_source, dataset):
    vocab = tokenization(vocab_source)
    dataset_bow = generate_vec(dataset, vocab)
    return dataset_bow


def to_vector(model, list_of_words):
    vectorized = [model.infer_vector(word_tokenize(doc.lower())) for doc in list_of_words]
    return np.array(vectorized)


def to_cat(data, num_tokens):
    encoder = keras.layers.CategoryEncoding(num_tokens=num_tokens, output_mode="one_hot")
    return encoder(data)


class MLModel(Model):
    def __init__(self):
        super().__init__()
        self.model = keras.Sequential()
        self.word_list = None

    def prepare_y_set(self, y_set):
        y_set = pd.Series(y_set)
        y_set, self.word_list = pd.factorize(y_set)
        y_set = to_cat(y_set, 15)
        return y_set

    def plot(self, history, epochs):
        fig, ax = plt.subplots()
        ax.plot(history.history["loss"], label="loss")
        ax.plot(history.history["val_loss"], label="Valditation loss")
        ax.set_title(f"Loss in {epochs} epochs")
        fig.legend()

        fig2, ax2 = plt.subplots()
        ax2.plot(history.history["accuracy"], label="accuracy")
        ax2.plot(history.history["val_accuracy"], label="Validation accuracy")
        ax2.set_title(f"Accuracy in {epochs} epochs")
        fig2.legend()


class BOW_Model(MLModel):
    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train, batch_size=64, epochs=12, verbose=0):
        X_train_bow = generate_bow(X_train, X_train)
        y_train_cat = self.prepare_y_set(y_train)
        history = self.model.fit(X_train_bow, y_train_cat, batch_size=batch_size, epochs=epochs, verbose=verbose)
        return history


class FNN_BOW_Model(BOW_Model):
    def __init__(self):
        super().__init__()
        self.model.add(keras.Input(shape=(731,)))
        self.model.add(keras.layers.Dense(256, activation="relu"))
        self.model.add(keras.layers.Dense(15, activation="softmax"))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


class LR_BOW_Model(BOW_Model):
    def __init__(self):
        super().__init__()
        self.model.add(keras.Input(shape=(731,)))
        self.model.add(keras.layers.Dense(15, activation="softmax"))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


class WE_Model(MLModel):
    def __init__(self):
        super().__init__()
        self.model = keras.Sequential()
        self.vectorizing_model = None

    def prepare_vectorizing_model(self, X_train):
        tagged_x_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                                        tags=[str(i)]) for i, doc in enumerate(X_train)]
        model = Doc2Vec(vector_size=50, min_count=2, epochs=50)
        model.build_vocab(tagged_x_data)
        model.train(tagged_x_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        return model

    def train(self, X_train, y_train, batch_size=64, epochs=12, verbose=0):
        self.vectorizing_model = self.prepare_vectorizing_model(X_train)
        document_vectors = to_vector(self.vectorizing_model, X_train)
        y_train_cat = self.prepare_y_set(y_train)
        history = self.model.fit(document_vectors, y_train_cat, batch_size=batch_size, epochs=epochs, verbose=verbose)
        return history

    def predict(self, X_test):
        X_test = to_vector(self.vectorizing_model, X_test)
        X_test = np.array(X_test)
        ynew = self.model.predict(X_test, verbose=False)
        return self.word_list[np.argmax(ynew[-1])]


class FNN_WE_Model(WE_Model):
    def __init__(self):
        super().__init__()
        self.model.add(keras.Input(shape=(50,)))
        self.model.add(keras.layers.Dense(256, activation="relu"))
        self.model.add(keras.layers.Dense(15, activation="softmax"))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


class LR_WE_Model(WE_Model):
    def __init__(self):
        super().__init__()
        self.model.add(keras.Input(shape=(50,)))
        self.model.add(keras.layers.Dense(15, activation="softmax"))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
