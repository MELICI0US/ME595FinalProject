from stagHare.agents.agent import Agent
from stagHare.agents.rl_agent.q_table_manager import QTableManager
from stagHare.environment.state import State
import numpy as np
from typing import Tuple

from stagHare.utils.utils import HARE_NAME, STAG_NAME

class QLearningAgent(Agent):    
    def __init__(self,  id: int, name: str, q_table_manager:QTableManager, epsilon:float = .1) -> None:
        Agent.__init__(self, name)
        self.id = id
        self.state_action_history = []
        self.hare = False
        self.epsilon = epsilon
        self.q_table_manager = q_table_manager

    def act(self, state: State, reward: float, round_num: int):
        action, hunt_hare = None, None

        # exploit with probability 1 - epsilon
        if np.random.rand() > self.epsilon:
            state_hash = self.make_state_key(state)
            if state_hash in self.q_table_manager.q_table and self.q_table_manager.q_table[state_hash]:
                best_action = max(self.q_table_manager.q_table[state_hash], key=self.q_table_manager.q_table[state_hash].get)
                
                action = [best_action[0], best_action[1]]
                hunt_hare = best_action[2]
        
        # explore with probability epsilon or if no known actions for this state
        if action is None:
            action = self.random_action(state)
            # hunt_hare = np.random.choice([True, False])     
            hunt_hare = False # for now, only hunt stag so we learn how  

        self.hare = hunt_hare  
            
        # store the history for later Q-table updates
        self.state_action_history.append((self.make_state_key(state), [int(action[0]), int(action[1]), bool(hunt_hare)]))
 
        return action
    
    def make_state_key(self, state: State):
        grid = state.grid
        agent_positions = state.agent_positions

        # make the state space slightly smaller and encode this agent's identity
        for agent_name, position in agent_positions.items():
            if agent_name == self.name:
                grid[position[0]][position[1]] = 'A'  
            elif agent_name == HARE_NAME:
                grid[position[0]][position[1]] = 'H' 
            elif agent_name == STAG_NAME:
                grid[position[0]][position[1]] = 'S' 
            else:
                # ignore the ids of the other agents and just mark their positions (we don't support self-identity here - not even a number).
                grid[position[0]][position[1]] = 'O' 

        return hash(str(grid))

    def is_hunting_hare(self) -> bool:
        return self.hare

    def update_q_table(self, reward: float):
        self.q_table_manager.update_q_table(reward, self.state_action_history)

            