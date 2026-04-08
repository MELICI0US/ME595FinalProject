from stagHare.agents.agent import Agent
from stagHare.agents.rl_agent.q_table_abstracted_manager import QTableAbstractedManager
from stagHare.environment.state import State
import numpy as np
from typing import Tuple

from stagHare.utils.utils import HARE_NAME, POSSIBLE_DELTA_VALS, POSSIBLE_MOVEMENTS, STAG_NAME

class StagGreedyAgent(Agent): 
    # This agents abstracts the state and action space so state is each agent's distance to the animals and the actions are just hunt stag or hare

    def __init__(self,  id: int, name: str) -> None:
        Agent.__init__(self, name)
        self.id = id
        self.state_action_history = []
        self.hare = False

    def is_hunting_hare(self) -> bool:
        return self.hare

    def act(self, state: State, reward: float, round_num: int):
        action = self.hunt_animal(state, STAG_NAME)

        return action
    
    def hunt_animal(self, state, animal_name) -> bool:
        # self.hare = (animal_name == HARE_NAME)

        agent_positions = state.agent_positions 
        other_agent_positions = {name: pos for name, pos in agent_positions.items() if name != self.name and name != animal_name}

        for name, position in agent_positions.items():
            if name == animal_name:   
                animal_row, animal_col = position
                break
        
        # move towards the animal, but stay if we are already adjacent
        my_row, my_col = state.agent_positions[self.name]
        action = [my_row, my_col]  

        positions_adjacent_to_animal = [(animal_row + 1, animal_col), (animal_row - 1, animal_col), (animal_row, animal_col + 1), (animal_row, animal_col - 1)]
        possible_actions = [(my_row + delta, my_col) for delta in POSSIBLE_DELTA_VALS] + [(my_row, my_col + delta) for delta in POSSIBLE_DELTA_VALS]

        # filter positions that are occupied by other agents
        possible_actions = [action for action in possible_actions if action not in other_agent_positions.values()]
        positions_adjacent_to_animal = [pos for pos in positions_adjacent_to_animal if pos not in other_agent_positions.values()]

        if (my_row, my_col) in positions_adjacent_to_animal:
            return action
        else:
            # find action that gets closest to one of the adjacent positions to the animal
            best_action = None
            best_distance = float('inf')
            for action in possible_actions:
                for adjacent_position in positions_adjacent_to_animal:
                    distance = abs(action[0] - adjacent_position[0]) + abs(action[1] - adjacent_position[1])
                    if distance < best_distance:
                        best_distance = distance
                        best_action = action

            return best_action

    



            