import pandas as pd
from state_machine_code import StateMachine

restaurant_info = pd.read_csv('data/restaurant_info.csv')
SM = StateMachine(restaurant_info)