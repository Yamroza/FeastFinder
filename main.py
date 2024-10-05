import pandas as pd

print('System is loading. It usually takes around 30 seconds')
from state_machine_code import StateMachine

restaurant_info = pd.read_csv('data/restaurant_info_expanded.csv')
SM = StateMachine(restaurant_info)