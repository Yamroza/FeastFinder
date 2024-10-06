"""
Functions for preference extraction
"""

import pandas as pd
pd.options.mode.chained_assignment = None
from Levenshtein import distance

restaurant_info = pd.read_csv('data/restaurant_info_expanded.csv')

# extracting preferences available in out dataset
food_types = pd.unique(restaurant_info['food'].dropna())
areas = pd.unique(restaurant_info['area'].dropna())
prices = pd.unique(restaurant_info['pricerange'].dropna())


# some of the food types are not appearing in the database,
# but we want them to be recognized as well
external_food_types = ['hindi', 'greek', 'scottish', 'corsica', 'christmas',
                       'mexican', 'kosher', 'belgium', 'world', 'creative',
                       'cantonese', 'basque', 'brazilian', 'hungarian',
                       'vegetarian', 'halal', 'austrian', 'german', 'kosher',
                       'australian', 'barbecue', 'scandinavian', 'swedish',
                       'danish', 'afghan']
food_types = list(food_types)
food_types.extend(external_food_types)


def extract_preference(utterance: str, category_list: list, threshold_distance: int) -> str:
    """
    Checks if specific words appear in the utterance and extracts them as a preference
    :param utterance: User input
    :param category_list: List of words we are looking for
    :param threshold_distance: Measure for Levenshtein distance
    :return Word from the list with a best match
    """
    words = utterance.lower().split()
    
    best_word = None
    for word in words:
        for keyword in category_list:
            word_distance = distance(word, keyword)
            if word_distance <= threshold_distance:
                threshold_distance = word_distance
                best_word = keyword

    return best_word


def extract_all_preferences(utterance: str, food_types: list = food_types, areas: list = areas, prices: list= prices) -> dict[str, str]:
    """
    Extracts all preferences from a single 'inform' utterance at once
    In the exercise description maximal Levenshtein distance is 3. In our opinion 3
    is too much. Examples where 3 still finds a preference (word in utterance -> preference):
    food -> seafood
    english -> polish
    care -> centre
    """
    value_dict = dict()
    value_dict['food_type'] = extract_preference(utterance, food_types, 2)
    value_dict['area'] = extract_preference(utterance, areas, 1)
    value_dict['price'] = extract_preference(utterance, prices, 2)
    
    return value_dict


def find_restaurants(restaurant_info: pd.DataFrame, preferences: dict) -> pd.DataFrame:
    """
    Searches for matching restaurants based on preferences
    :param restaurant_info: dataset with all available restaurants
    :param preferences: dictionary with 3 keys: food_type, area and price. 
    If key is 'any', the constraint is skipped. At least 1 key is required.
    :return: pd.Dataframe of matching restaurants
    """
    restaurants = restaurant_info[
        (restaurant_info['food'] == preferences['food_type'] if preferences['food_type'] != 'any' else True) & 
        (restaurant_info['area'] == preferences['area'] if preferences['area'] != 'any' else True) &
        (restaurant_info['pricerange'] == preferences['price'] if preferences['price'] != 'any' else True)
    ]
    return restaurants

add_pref = ['romantic', 'touristic', 'assigned seats', 'children']

def extract_all_preferences_add(utterance: str) -> dict:
    """
    Looks for additional preferences in user utterance
    """
    value_dict = {
    'touristic': False,
    'assigned seats': False,
    'romantic' : False,
    'children' : False }
    pref = extract_preference(utterance, add_pref, 2)

    if pref in value_dict:
        value_dict[pref] = True
 
    return value_dict 


def add_requirements(subset_rest: pd.DataFrame) -> pd.DataFrame:
    """
    Adding additional info to a chosen subset of restaurant database, using a rule system
    :param subset_rest: Database fragment
    :return: database with additional info
    """
    subset_rest['romantic'] = None
    subset_rest['touristic'] = None
    subset_rest['assigned_seats'] = None
    subset_rest['children'] = None
    
    subset_rest.loc[subset_rest['food'] == 'romanian', 'touristic'] = False
    subset_rest.loc[(subset_rest['food_quality'] == 'good') & (subset_rest['pricerange'] == 'cheap'), 'touristic'] = True
    subset_rest.loc[subset_rest['crowdedness'] == 'busy', 'assigned_seats'] = True
    subset_rest.loc[subset_rest['length_of_stay'] == 'long stay', 'children'] = False
    subset_rest.loc[subset_rest['crowdedness'] == 'busy','romantic'] = False
    subset_rest.loc[subset_rest['length_of_stay'] == 'long stay','romantic'] = True
    
    return subset_rest


def find_add_preferences(restaurant_info: pd.DataFrame, add_preferences: dict) -> pd.DataFrame:
    """
    Choosing restaurants from database based on additional user preferences
    :param restaurant_info: chosen database subset
    :param add_preferences: chosen preferences
    :return: filtered restaurants
    """
    spec_restaurants = add_requirements(restaurant_info)

    if add_preferences.get('touristic') is True:
        spec_restaurants = spec_restaurants[spec_restaurants['touristic'] == True]
        
    if add_preferences.get('assigned seats') is True:
        spec_restaurants = spec_restaurants[spec_restaurants['assigned_seats'] == True]
        
    if add_preferences.get('children') is True:
        spec_restaurants = spec_restaurants[spec_restaurants['children'] == True]
        
    if add_preferences.get('romantic') is True:
        spec_restaurants = spec_restaurants[spec_restaurants['romantic'] == True]
    
    return spec_restaurants
