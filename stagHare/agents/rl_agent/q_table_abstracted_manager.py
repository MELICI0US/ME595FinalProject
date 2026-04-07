from stagHare.environment.state import State

class QTableAbstractedManager():    
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
                    state_raw, action_raw, q_value_raw = line.strip().split(';')

                    state_raw = state_raw.strip('()').split(', ')

                    state = []
                    for i in range(len(state_raw)):
                        state.append(int(state_raw[i]))

                    state = tuple(state)

                    q_value_raw = q_value_raw.strip('()').split(', ')
                    q_value_count = (float(q_value_raw[0]), int(q_value_raw[1]))

                    action_raw = action_raw.strip('()').split(', ')
                    action = bool(action_raw == 'True')

                    if state not in q_table:
                        q_table[state] = {}
                    q_table[state][action] = q_value_count
            print(f"Loaded Q-table from {file_path}")
        except FileNotFoundError:
            print(f"No existing Q-table found at {file_path}. Starting with an empty Q-table.")

        return q_table
        
    def update_q_table(self, reward: float, state_action_history: list):
        # work backwards so we have the future rewards available
        reverse_history = state_action_history[::-1]
        s_prime, a_prime = None, None

        # print('Number of state-action pairs in history:', len(state_action_history))
        state_add_count = 0
        action_add_count = 0
        for state, action in reverse_history:
            # initialize q-table entries if they don't exist
            state_tuple = state
            
            if state_tuple not in self.q_table:
                self.q_table[state_tuple] = {}
                state_add_count += 1
            action_key = action

            if action_key not in self.q_table[state_tuple]:
                self.q_table[state_tuple][action_key] = (0, 0)
                action_add_count += 1

            # if this is the end state, update with full reward
            if s_prime is None:
                s_prime = state_tuple

                old_count = self.q_table[state_tuple][action_key][1]
                new_count = old_count + 1
                old_q_value = self.q_table[state_tuple][action_key][0]
                new_q_value = ((old_count * old_q_value) + self.alpha * reward) / new_count

                self.q_table[state_tuple][action_key] = (new_q_value, new_count)
            # otherwise, update with discounted future reward
            else:
                max_future_q = 0
                for a_prime in self.q_table[s_prime]:
                    if self.q_table[s_prime][a_prime][0] > max_future_q:
                        max_future_q = self.q_table[s_prime][a_prime][0]    
                
                old_count = self.q_table[state_tuple][action_key][1]
                new_count = old_count + 1
                old_q_value = self.q_table[state_tuple][action_key][0]
                new_q_value = ((old_count * old_q_value) + self.alpha * (self.gamma * max_future_q - self.q_table[state_tuple][action_key][0])) / new_count
                
                self.q_table[state_tuple][action_key] = (new_q_value, new_count)
                s_prime, a_prime = state_tuple, action_key

        # print(f"Added {state_add_count} new states and {action_add_count} new actions to the Q-table.")

    def save_q_table(self):
        # save the q-table to a file for the next iteration        
        with open(self.q_table_file, 'w') as f:
            for state_hash in self.q_table:
                for action_key in self.q_table[state_hash]:
                    f.write(f'{state_hash};{action_key};{self.q_table[state_hash][action_key]}\n')

            