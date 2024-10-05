# %%
import pandas as pd
from Levenshtein import distance

# %%
restaurant_info = pd.read_csv('data/restaurant_info.csv')

# extracting preferences available in out dataset
food_types = pd.unique(restaurant_info['food'].dropna())
areas = pd.unique(restaurant_info['area'].dropna())
prices = pd.unique(restaurant_info['pricerange'].dropna())

# %%
external_food_types = ['hindi', 'greek', 'scottish', 'corsica', 'christmas',
                       'mexican', 'kosher', 'belgium', 'world', 'creative',
                       'cantonese', 'basque', 'brazilian', 'hungarian',
                       'vegetarian', 'halal', 'austrian', 'german', 'kosher',
                       'australian', 'barbecue', 'scandinavian', 'swedish',
                       'danish', 'afghan']
food_types = list(food_types)
food_types.extend(external_food_types)

# %%
def extract_preference(utterance: str, category_list: list, threshold_distance: int) -> str:
    words = utterance.lower().split()
    
    best_word = None
    for word in words:
        for keyword in category_list:
            word_distance = distance(word, keyword)
            if word_distance <= threshold_distance:
                threshold_distance = word_distance
                best_word = keyword

    return best_word


# %%
def extract_all_preferences(utterance: str, food_types: list = food_types, areas: list = areas, prices: list= prices) -> dict[str, str]:
    """
    Extracts all preferences from a single 'inform' utterance at once
    TO DISCUSS:
    In the exercise description maximal Levenshtein distance is 3. Imo 3 is too much.
    Examples where 3 still finds a preference (word in utterance -> preference):
    food -> seafood
    english -> polish
    care -> centre
    """
    value_dict = dict()
    value_dict['food_type'] = extract_preference(utterance, food_types, 2)
    value_dict['area'] = extract_preference(utterance, areas, 2)
    value_dict['price'] = extract_preference(utterance, prices, 2)
    
    return value_dict    

# %%
# idc = ['t care', 'any', 't matter']

sample_utterance = 'cheep, chinese food in amazing sothu of Utrecht'
preferences = extract_all_preferences(sample_utterance, food_types, areas, prices)
preferences

# %%
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

# %%
find_restaurants(restaurant_info, preferences)


def extract_all_preferences_add(utterance, add_pref):

    value_dict = dict()
    value_dict['add_pref'] = extract_preference(utterance, add_pref, 2)
    
    return value_dict 

def add_requirements(subset_rest):

    subset_rest['romantic'] = None
    subset_rest['touristic'] = None
    subset_rest['assigned_seats'] = None
    subset_rest['children'] = None
    
    subset_rest.loc[subset_rest['cuisine'] == 'romanian', 'touristic'] = False
    subset_rest.loc[(subset_rest['food_quality'] == 'good') & (subset_rest['price'] == 'cheap'), 'touristic'] = True
    subset_rest.loc[subset_rest['crowdedness'] == 'busy', 'assigned_seats'] = True
    subset_rest.loc[subset_rest['length_of_stay'] == 'long', 'children'] = False
    subset_rest.loc[subset_rest['crowdedness'] == 'busy','romantic'] = False
    subset_rest.loc[subset_rest['length_of_stay'] == 'long','romantic'] = True
    
    return subset_rest

def find_add_preferences(restaurant_info: pd.DataFrame, add_preferences: dict) -> pd.DataFrame:
    """
    """
    
    spec_restaurants = add_requirements(restaurant_info)

    if 'touristic' in add_preferences:
        spec_restaurants = spec_restaurants[spec_restaurants['touristic'] == True]
        
    if 'assigned seats' in add_preferences:
        spec_restaurants = spec_restaurants[spec_restaurants['assigned_seats'] == True]
        
    if 'children' in add_preferences:
        spec_restaurants = spec_restaurants[spec_restaurants['children'] == True]
        
    if 'romantic' in add_preferences:
        spec_restaurants = spec_restaurants[spec_restaurants['romantic'] == True]
    
    return spec_restaurants
