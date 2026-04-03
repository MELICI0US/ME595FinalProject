from stagHare.agents.agent import Agent
from stagHare.environment.state import State
import numpy as np
from typing import Tuple

from stagHare.utils.utils import HARE_NAME, STAG_NAME

class QLearningAgent(Agent):    
    def __init__(self,  id: int, name: str, alpha: float = .1, gamma: float = .9, epsilon:float = .1, q_table_file = 'stagHare/agents/rl_agent/q_table.txt') -> None:
        Agent.__init__(self, name)
        self.id = id
        self.state_action_history = []
        self.hare = False
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_file = q_table_file
        self.q_table = self.load_q_table(self.q_table_file) 

    def load_q_table(self, file_path):
        q_table = {}
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    state_hash, action_key, q_value = line.strip().split(';')

                    # convert action_key back to a tuple of (x, y, hunt_hare)
                    action_key = action_key.strip('()').split(', ')
                    action_key = [int(action_key[0]), int(action_key[1]), action_key[2] == 'True']  # convert to correct types

                    if state_hash not in q_table:
                        q_table[state_hash] = {}
                    q_table[state_hash][tuple(action_key)] = float(q_value)
            # print(f"Loaded Q-table from {file_path}")
        except FileNotFoundError:
            print(f"No existing Q-table found at {file_path}. Starting with an empty Q-table.")

        return q_table

    def act(self, state: State, reward: float, round_num: int):
        action, hunt_hare = None, None

        # exploit with probability 1 - epsilon
        if np.random.rand() > self.epsilon:
            state_hash = self.make_state_key(state)
            if state_hash in self.q_table and self.q_table[state_hash]:
                best_action = max(self.q_table[state_hash], key=self.q_table[state_hash].get)
                
                action = [best_action[0], best_action[1]]
                hunt_hare = best_action[2]
        
        # explore with probability epsilon or if no known actions for this state
        if action is None:
            action = self.random_action(state)
            hunt_hare = np.random.choice([True, False])        

        self.hare = hunt_hare  
            
        # store the history for later Q-table updates
        self.state_action_history.append((self.make_state_key(state), [action[0], action[1], hunt_hare]))
 
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
                # ignore the ids of the other agents and just mark their positions (we don't support self-identity here - not even a numbers).
                grid[position[0]][position[1]] = 'O' 

        return hash(str(grid))

    def is_hunting_hare(self) -> bool:
        return self.hare

    def update_q_table(self, state: State, reward: float):

        # load the existing q-table from the file in case it was updated by another agent
        self.q_table = self.load_q_table(self.q_table_file)

        # work backwards so we have the future rewards available
        reverse_history = self.state_action_history[::-1]
        s_prime, a_prime = None, None

        for state_hash, action in reverse_history:
            # initialize q-table entries if they don't exist
            if state_hash not in self.q_table:
                self.q_table[state_hash] = {}
            action_key = tuple(action)

            if action_key not in self.q_table[state_hash]:
               self.q_table[state_hash][action_key] = 0

            # if this is the end state, update with full reward
            if s_prime is None:
                s_prime = state_hash
                self.q_table[state_hash][action_key] += self.alpha * reward
            # otherwise, update with discounted future reward
            else:
                max_future_q = 0
                for a_prime in self.q_table[s_prime]:
                    if self.q_table[s_prime][a_prime] > max_future_q:
                        max_future_q = self.q_table[s_prime][a_prime]      

                self.q_table[state_hash][action_key] += self.alpha * (self.gamma * max_future_q - self.q_table[state_hash][action_key])
                s_prime, a_prime = state_hash, action_key

        # save the q-table to a file for the next iteration
        with open(self.q_table_file, 'w') as f:
            for state_hash in self.q_table:
                for action_key in self.q_table[state_hash]:
                    f.write(f'{state_hash};{action_key};{self.q_table[state_hash][action_key]}\n')

            