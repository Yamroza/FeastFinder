import pandas as pd

print('System is loading. It usually takes a few seconds')
from state_machine_code import StateMachine

restaurant_info = pd.read_csv('data/restaurant_info_expanded.csv')
model_path = './models/lr_we_classifier.keras'

SM = StateMachine(restaurant_info, model_path)