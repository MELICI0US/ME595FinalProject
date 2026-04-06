from stagHare.environment.state import State

class QTableManager():    
    def __init__(self, alpha: float = .1, gamma: float = .9, epsilon:float = .1, q_table_file = 'stagHare/agents/rl_agent/q_table_4x4.txt') -> None:
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
            print(f"Loaded Q-table from {file_path}")
        except FileNotFoundError:
            print(f"No existing Q-table found at {file_path}. Starting with an empty Q-table.")

        return q_table
        
    def update_q_table(self, reward: float, state_action_history: list):

        # # load the existing q-table from the file in case it was updated by another agent
        # self.q_table = self.load_q_table(self.q_table_file)

        # work backwards so we have the future rewards available
        reverse_history = state_action_history[::-1]
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

    def save_q_table(self):
        # save the q-table to a file for the next iteration
        with open(self.q_table_file, 'w') as f:
            for state_hash in self.q_table:
                for action_key in self.q_table[state_hash]:
                    f.write(f'{state_hash};{action_key};{self.q_table[state_hash][action_key]}\n')

            