"""
Training and saving ML models for later use.
Allows avoiding training every time a system is run.
Consists of:
1. Utterance category classifier training
2. Doc2Vec model training
"""

from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from ml_models import LR_WE_Model

# Train & test set preparation
x_set = []
y_set = []

with open("data/dialog_acts.dat", 'r') as file:
    for line in file:
        y_set.append(line.split()[0])
        x_set.append(" ".join(line.split()[1:]).lower())

x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.2, random_state=42)


# Utterance category classifier
model = LR_WE_Model()
model.train(x_train, y_train)
model.save('./models/lr_we_classifier.keras')


# Doc2Vec model
tagged_x_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                                tags=[str(i)]) for i, doc in enumerate(x_train)]
vec_model = Doc2Vec(vector_size=50, min_count=2, epochs=50)
vec_model.build_vocab(tagged_x_data)
vec_model.train(tagged_x_data,
            total_examples=vec_model.corpus_count,
            epochs=vec_model.epochs)
vec_model.save("./models/doc2vec_model")