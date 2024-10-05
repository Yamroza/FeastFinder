import numpy as np

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


class MLModel(Model):
    def __init__(self):
        super().__init__()
        self.model = keras.Sequential()


class FNN_BOW_Model(MLModel):
    def __init__(self):
        super().__init__()
        self.model.add(keras.Input(shape=(731,)))
        self.model.add(keras.layers.Dense(256, activation="relu"))
        self.model.add(keras.layers.Dense(15, activation="softmax"))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, batch_size=64, epochs=12, verbose=1):
        tagged_x_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in
                         enumerate(X_train)]
        model = Doc2Vec(vector_size=50,
                        min_count=2, epochs=50)
        model.build_vocab(tagged_x_data)
        model.train(tagged_x_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def predict(self, input):
        return self.model.predict(input)