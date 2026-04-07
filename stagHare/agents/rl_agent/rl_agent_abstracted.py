from stagHare.agents.agent import Agent
from stagHare.agents.rl_agent.q_table_abstracted_manager import QTableAbstractedManager
from stagHare.environment.state import State
import numpy as np
from typing import Tuple

from stagHare.utils.utils import HARE_NAME, POSSIBLE_DELTA_VALS, POSSIBLE_MOVEMENTS, STAG_NAME

class QLearningAbstractedAgent(Agent): 
    # This agents abstracts the state and action space so state is each agent's distance to the animals and the actions are just hunt stag or hare

    def __init__(self,  id: int, name: str, q_table_manager:QTableAbstractedManager, epsilon:float = .1) -> None:
        Agent.__init__(self, name)
        self.id = id
        self.state_action_history = []
        self.hare = False
        self.epsilon = epsilon
        self.q_table_manager = q_table_manager

    def is_hunting_hare(self) -> bool:
        return self.hare

    def act(self, state: State, reward: float, round_num: int):
        action, hunt_hare = None, None

        # exploit with probability 1 - epsilon
        if np.random.rand() > self.epsilon:
            state_hash = self.make_state_key(state)
            if state_hash in self.q_table_manager.q_table and self.q_table_manager.q_table[state_hash]:
                best_action = max(self.q_table_manager.q_table[state_hash], key=lambda a: self.q_table_manager.q_table[state_hash][a][0]) 

                hunt_hare = best_action
                action = self.hunt_animal(state, HARE_NAME if hunt_hare else STAG_NAME)
                # print(f"{self.name} is exploiting with action {best_action} at round {round_num}")

        # explore with probability epsilon or if no known actions for this state
        if action is None:
            hunt_hare = np.random.choice([True, False])
            # hunt_hare = False # force stag to explore it's state
            if hunt_hare:
                action = self.hunt_animal(state, HARE_NAME)
            else:
                action = self.hunt_animal(state, STAG_NAME)
            # print(f"{self.name} is exploring with action {hunt_hare} at round {round_num}")

        # if hunt_hare is True:
        #     print(f"{self.name} is hunting the hare at round {round_num}")
            
        # store the history for later Q-table updates
        self.state_action_history.append((self.make_state_key(state), hunt_hare))   
 
        return action
    
    def hunt_animal(self, state, animal_name) -> bool:
        self.hare = (animal_name == HARE_NAME)

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
    
    def make_state_key(self, state: State):
        '''  returns a tuple of (distance_to_stag, distance_to_hare, other_agents_distance_to_stag, other_agents_distance_to_hare) 
        where the other agents distances are sorted lists of the distances to the stag and hare for the other agents in the environment.
        '''
        
        distance_to_stag = self.distance_to_animal(state, STAG_NAME, self.name)
        distance_to_hare = self.distance_to_animal(state, HARE_NAME, self.name)

        other_agents_distance_to_stag = []
        other_agents_distance_to_hare = []

        for agent_name in state.agent_positions:
            if agent_name != self.name and agent_name != HARE_NAME and agent_name != STAG_NAME:
                other_agents_distance_to_stag.append(self.distance_to_animal(state, STAG_NAME, agent_name))
                other_agents_distance_to_hare.append(self.distance_to_animal(state, HARE_NAME, agent_name))

        # for consistency, sort distances
        other_agents_distance_to_stag.sort()
        other_agents_distance_to_hare.sort()

        # Make sure the key is int instead of np.int64
        key = (distance_to_stag, distance_to_hare, other_agents_distance_to_stag[0], other_agents_distance_to_stag[1], other_agents_distance_to_hare[0], other_agents_distance_to_hare[1])
        int_key = []
        for item in key:
            int_key.append(int(item))

        return tuple(int_key)
    
    def distance_to_animal(self, state: State, animal_name:str, agent_name:str) -> int:
        agent_positions = state.agent_positions
        stag_position = None

        for name, position in agent_positions.items():
            if name == animal_name:
                stag_position = position
                break

        if stag_position is None:
            raise ValueError("Stag not found in agent positions")

        position = agent_positions[agent_name]

        # using Manhattan distance since we can't move diagonally
        distance = abs(position[0] - stag_position[0]) + abs(position[1] - stag_position[1])

        return distance

    def update_q_table(self, reward: float):
        self.q_table_manager.update_q_table(reward, self.state_action_history)

            