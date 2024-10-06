from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Model:
    def __init__(self):
        self_model = None

    def train(self):
        pass

    def evaluate(self, X_test, y_test):
        y_pred_list = []
        for input in X_test:
            y_pred_list.append(self.predict(input))

        accuracy = accuracy_score(y_test, y_pred_list)
        precision = precision_score(y_test, y_pred_list, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_list, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_list, average='weighted')

        return accuracy, precision, recall, f1

    def predict(self, input):
        pass


class BaselineModel(Model):
    def __init__(self):
        super().__init__()
        self.prediction = None

    def train(self, X_train, y_train):
        self.prediction = max(set(y_train), key=y_train.count)

    def predict(self, input):
        return self.prediction


class RuleBasedModel(Model):
    def __init__(self):
        super().__init__()
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

    def predict(self, input):
        words = input[0].lower().split()
        prediction_dict = {key: 0 for key in self.keyword_dict.keys()}

        for key in self.keyword_dict.keys():            # Iterating over categories
            for keyword in self.keyword_dict[key]:      # Iterating over keywords in the category
                for word in words:                      # Iterating over words in the sentence
                    if keyword == word:
                        prediction_dict[key] += 1

        return max(prediction_dict, key=prediction_dict.get)
