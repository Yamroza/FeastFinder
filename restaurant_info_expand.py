import pandas as pd
import numpy as np

"""
Part 1C: Adding reasoning & Inference
    Adding the properties food quality, crowdedness and length of stay to the csv file
    using random values
"""

restaurant_info = pd.read_csv('data/restaurant_info.csv')

# available values
food_quality_ant = ['cheap', 'good', 'expensive']
crowdedness_ant = ['busy', 'quiet', 'moderate', 'packed']
length_of_stay_ant = ['long stay', 'short stay', 'moderate stay']

# Initialization of 1D arrays
food_quality = np.random.choice(food_quality_ant, size=len(restaurant_info))
crowdedness = np.random.choice(crowdedness_ant, size=len(restaurant_info))
length_of_stay = np.random.choice(length_of_stay_ant, size=len(restaurant_info))

# Adding attributes to the DataFrame
restaurant_info["food_quality"] = food_quality
restaurant_info["crowdedness"] = crowdedness
restaurant_info["length_of_stay"] = length_of_stay

# Saving expanded dataset
restaurant_info.to_csv("./data/restaurant_info_expanded.csv", index=False)