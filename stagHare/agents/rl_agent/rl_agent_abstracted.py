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
                action = self.hunt_hare(state) if hunt_hare else self.hunt_stag(state)

        # explore with probability epsilon or if no known actions for this state
        if action is None:
            # hunt_hare = np.random.choice([True, False])
            hunt_hare = False # force stag to explore it's state
            if hunt_hare:
                action = self.hunt_hare(state)
            else:
                action = self.hunt_stag(state)
            
        # store the history for later Q-table updates
        self.state_action_history.append((self.make_state_key(state), hunt_hare))   
 
        return action
    
    def hunt_hare(self, state) -> bool:
        self.hare = True

        agent_positions = state.agent_positions 
        for name, position in agent_positions.items():
            if name == HARE_NAME:
                hare_row, hare_col = position
                break
        
        # move towards the hare, but stay if we are already adjacent
        my_row, my_col = state.agent_positions[self.name]
        action = [my_row, my_col] 

        if my_row < hare_row - 1:
            action[0] += 1
        elif my_row > hare_row + 1:
            action[0] -= 1  
        elif my_col < hare_col - 1:
            action[1] += 1
        elif my_col > hare_col + 1:
            action[1] -= 1

        return action
    
    def hunt_stag(self, state) -> bool:
        # TODO: add something that moves around the other players if they ar in the way
        self.hare = False

        agent_positions = state.agent_positions 

        for name, position in agent_positions.items():
            if name == STAG_NAME:   
                stag_row, stag_col = position
                break
        
        # move towards the stag, but stay if we are already adjacent
        my_row, my_col = state.agent_positions[self.name]
        action = [my_row, my_col] 

        if my_row < stag_row -1:
            action[0] += 1
        elif my_row > stag_row + 1:
            action[0] -= 1  
        elif my_col < stag_col -1:
            action[1] += 1
        elif my_col > stag_col + 1:
            action[1] -= 1

        return action
    
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

            