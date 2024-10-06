import pandas as pd
from state_machine_code import StateMachine

if __name__ == '__main__':
    restaurant_info = pd.read_csv('data/restaurant_info_expanded.csv')
    SM = StateMachine(restaurant_info, "./models/lr_bow_classifier.keras")
