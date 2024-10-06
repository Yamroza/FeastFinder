import pandas as pd
import configparser
import time
import regex as re
import nltk


from sklearn.model_selection import train_test_split
from ml_models import LR_WE_Model
from pref_extract import find_restaurants, extract_all_preferences, extract_preference, find_add_preferences, extract_all_preferences_add

# Opening train & test set
x_set = []
y_set = []

with open("data/dialog_acts.dat", 'r') as file:
    for line in file:
        y_set.append(line.split()[0])
        x_set.append(" ".join(line.split()[1:]).lower())

x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.2, random_state=42)

# Initializing ml model for category classification
model = LR_WE_Model()
model.train(x_train, y_train)

# Importing restaurant info
restaurant_info = pd.read_csv('data/restaurant_info.csv')

# %%
# Possible state transitions, not used, only to look at
# state_transition_possibilities = {
#     1: [2, 9],          
#     2: [2, 3, 9],
#     3: [3, 4, 9],
#     4: [4, 5, 9],
#     5: [6, 7, 9],
#     6: [5, 9],
#     7: [5, 8, 9],
#     8: [8, 9]
# }

# %%
class StateMachine:
    def __init__(self, restaurant_info):

        self.configParser = configparser.ConfigParser()
        self.filename = "config.ini"
        self.configParser.read(self.filename)
        self.setting = self.configParser.sections()
        self.delay = self.configParser[self.setting[0]].getfloat('delay')
        self.if_caps = self.configParser[self.setting[1]].getboolean('caps')
        self.style = self.configParser[self.setting[1]]['style']
        self.if_restart = self.configParser[self.setting[1]].getboolean('restart')

        self.state = 1
        self.preferences = {
            'food_type': None,
            'area': None,
            'price': None
        }
        self.add_preferences = {
            'touristic': False,
            'assigned seats': False,
            'romantic' : False,
            'children' : False
        }
        self.restaurant_info = restaurant_info
        self.restaurants_options = pd.DataFrame()
        self.selected_rest = ""
        self.restaurant_name = ""
        self.area = ""
        self.price = ""
        self.food_type = ""

        self.romantic = ""
        self.touristic = ""
        self.assigned_seats = ""
        self.children = ""

        self.request_answer = ""
        self.requested_info = ""
        self.input_text = ""
        self.synonyms = {
            'area': ["area", "location", "part", "zone","sector","district"],
            'food_type': ["food", "type", "kitchen", "food type","cuisine","taste"],
            'price': ["price", "costs", "luxury", "amount","costs"]
        }
        self.rest_additional = None

        if self.style == 'informal':
            self.message_dict = {
                1: "Yo! This is FeastFinder, the best way to find a restaurant for you and your hommies. Let me know what do you crave, where, and how expensive. Also if you have any additional preferences!",
                2: "In what hood do you want to eat?",
                3: "What type of food are you looking for?",
                4: "How expensive should it be?",
                5: 'You want anything else?',
                6: "Sorry, no such place in my database. We can try to look for something else, give me different preferences.",
                7: f'{self.restaurant_name} is in {self.area} and is a {self.price} {self.food_type}. Does it sounds good?',
                8: f'{self.requested_info} of {self.restaurant_name} is {self.request_answer}',
                9: "Have a great one. Peace!",
                10: f"I have a different one for you. What about {self.restaurant_name}",
                11: "That is not what I expected"
            }
        else:
            self.message_dict = {
                1: "Welcome to FeastFinder, I will help find you a restaurant. Please tell what type of food would you like to eat, where, and in what price range. Also if you have any additional preferences!",
                2: "In what area do you want to have dinner?",
                3: "What cuisine are you looking for?",
                4: "In what price range should the restaurant be?",
                5: 'Do you have any additional requirements?',
                6: "I'm sorry, I don't see any restaurants matching your preferences in my database. Try providing different preferences.",
                7: f'{self.restaurant_name} is located in {self.area} and is a {self.price} {self.food_type}. Does it sounds good?',
                8: f'{self.requested_info} of {self.restaurant_name} is {self.request_answer}',
                9: "I hope you'll have a great time. Bye bye!",
                10: f"That is unfortunate. What do you think of {self.restaurant_name}",
                11: "That is not what I expected"
            }
        
        if self.if_caps:
            print(self.message_dict[self.state].upper())
        else:
            print(self.message_dict[self.state])

        if self.if_restart:
            print('To start over, just type \'reset\'.')

        utterance = input()
        self.change_state(utterance)


    def pattern_recog(self, utterance):
        """
        We recognize the pattern "any X". It returns X. It returns a list of all instances of X
        """
        matches = re.findall(r'\bany\b\s+(\w+)', utterance)
        if matches:
            return matches
        else:
            return None
        
    def update_dict(self, old: dict, new: dict) -> dict:
        old.update( (k,v) for k,v in new.items() if v is not None)
        return old

    def any_update(self, words_after_any: list, orig_words: list):
        for word in words_after_any:
            for orig_word in orig_words: 
                if extract_preference(word, self.synonyms[orig_word], 2):
                    self.preferences = self.update_dict(self.preferences, {orig_word: "any"})

    def change_state(self, utterance = None):
        next_state, if_message = self.predict_next_state(utterance)
        if if_message:
            if(self.if_caps):
                time.sleep(self.delay)
                print(self.message_dict[next_state].upper())
            else:
                time.sleep(self.delay)
                print(self.message_dict[next_state])
            utterance = input()
            self.state = next_state
            self.change_state(utterance)
        else:
            self.state = next_state
            self.change_state(utterance)

    def predict_next_state(self, utterance = None) -> tuple[int, bool]:
        """
        If the state should send a message, and then wait for user answer,
        it returns 'True'. If not, eg. state 2->4, returns False.
        :param utterance: 
        :return: tuple(next state number, if message sent) 
        """
        category = ""
        if self.if_restart == True:
            if utterance == 'reset':
                return 1, True        
        if utterance is not None:
            category = model.predict([utterance])
        if category == 'thankyou':
            return 9, True

        
        if self.state == 1:
            if utterance is not None:
                self.preferences = extract_all_preferences(utterance)
            return 2, False
        
        if self.state == 2:
            self.preferences = self.update_dict(self.preferences, extract_all_preferences(utterance))
            if self.pattern_recog(utterance) is not None:
                self.any_update( self.pattern_recog(utterance), ["area"])
            if self.preferences['area'] is None:
                return 2, True
            else:
                return 3, False
            
        if self.state == 3:
            self.preferences = self.update_dict(self.preferences, extract_all_preferences(utterance))
            if self.pattern_recog(utterance) is not None:
                self.any_update( self.pattern_recog(utterance), ["food_type"])
            if self.preferences['food_type'] is None:
                return 3, True
            else:
                return 4, False
            
        if self.state == 4:
            self.preferences = self.update_dict(self.preferences, extract_all_preferences(utterance))
            if self.pattern_recog(utterance) is not None:
                self.any_update( self.pattern_recog(utterance), ["price"])
            if self.preferences['price'] is None:
                return 4, True
            else:
                return 5, False
        
        if self.state == 5:
            print("pref: ", self.preferences)
            self.restaurants_options = find_restaurants(restaurant_info, self.preferences)
            self.add_preferences = self.update_dict(self.add_preferences, extract_all_preferences_add(utterance))
            self.restaurants_options = find_add_preferences(self.restaurants_options, self.add_preferences)
            if self.restaurants_options.empty:
                return 6, True
            else:
                self.get_restaurant()
                message = f'{self.restaurant_name} is located in {self.area} and is a {self.price} {self.food_type}. Does it sound good? {self.reason}'
                self.message_dict[7] = message 
                return 7, True

        if self.state == 6:
            self.preferences = self.update_dict(self.preferences, extract_all_preferences(utterance))
            if self.pattern_recog(utterance) is not None:
                self.any_update( self.pattern_recog(utterance), ["price"])
            self.restaurants_options = find_restaurants(self.restaurant_info, self.preferences)
            return 5, False
            
        if self.state == 7:
            # TODO - category prediction, now - hardcoded category
            # category = predict(utterance)
            # category = 'request' / 'reqalts'
            if category == 'request':
                return 8, False
            if category == 'regalts':
                return 5, False
            if category == "affirm":
                return 9, False
            if category == "negate":
                return 10, False
            else:
                return 11, False
            # TODO: what happens else?
            #return ???
            
        if self.state == 8:
            # TODO bug fixing (phone) text not displayed
            message = ""
            request_dict = {
                'food': ["type", "food"],
                'phone': ["phone", "number"],
                'addr': ["adress", "location"],
                'postcode': ["postal", "postcode"]
             }
            for key, item in request_dict.items():
                if extract_preference(utterance, item, 2) is not None:
                    if key == "food":
                        key_name = "cuisine"
                    elif key == "phone":
                        key_name = "phone number"
                    elif key == "addr":
                        key_name = "location"
                    elif key == "postcode":
                        key_name = "postal code"

                    message = message + f"The {key_name} of {self.restaurant_name} is {self.selected_rest[key].values[0]} \n"
            
            self.message_dict[8] = message
            return 8, True
        
        if self.state == 9:
            # TODO - its basically end of dialog
            return 9, True
        
        if self.state == 10:
            if self.restaurants_options.empty:
                return 6, True
            self.get_restaurant()
            message1 = f"That is unfortunate. Does {self.restaurant_name} sound good?"
            self.message_dict[7] = message1
            return 7, True

    def get_restaurant(self):
        rest = self.restaurants_options.sample()
        self.selected_rest = rest
        self.restaurant_name =rest["restaurantname"].values[0]
        self.area = rest["area"].values[0]
        self.price = rest["pricerange"].values[0]
        self.food_type = rest["food"].values[0]
        self.reason = " "

        for pref, valbool in self.add_preferences.items():

            if valbool:
                if pref == 'assigned_seats':
                    self.reason = "The restaurant has assigned seats because the restaurant is busy."
                elif pref == 'children':
                    self.reason = "The restaurant is suitable for children as it does not have a long stay."
                elif pref == 'romantic':
                    if  rest["length_of_stay"].values[0] == "long stay":
                        self.reason = "The restaurant is romantic because you can stay for a long time."
                elif pref == 'touristic':
                    if self.food_type != "romanian":
                        self.reason = "The restaurant is touristic because it serves cheap and good food."

        self.restaurants_options = self.restaurants_options.drop(rest.index)
