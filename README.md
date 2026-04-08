# ME595FinalProject

## Elevator Pitch (Revised Version): Q-Learning On Grid-Hunt

Coordination problems arise when multiple agents must cooperate to achieve a high reward, but acting alone provides a safer alternative. The Stag Hunt game is a classic example of this dilemma, where agents must decide whether to cooperate to capture a stag or act individually to hunt smaller prey. While the traditional Stag Hunt is modeled as a single-step decision, many real-world coordination problems involve sequential decision making in space, where agents must move and position themselves before cooperation can occur. This project investigates how agents learn to coordinate in a grid version of the Stag Hunt dilemma.

I will model the problem as a grid world environment with three agents, a hare, and a stag placed at different positions in the grid. The agents must move through the environment so that all three reach positions adjacent to the stag in order to successfully hunt it (a high reward for everyone). Alternatively, one agent can approach the hare to hunt it and end the round for everyone (a small reward for only the player that hunted). I will train an agent using Q-learning, updating its action-value estimates based on the rewards received during each round of simulation. The agent will have no knowledge of the other players’ intentions and must make decisions based on the observations of their actions. Additionally, I will test the agent against different static algorithms designed for the Stag Hunt problem to see how well it does, such as if the other agents always go for the hare. The agent will ultimately be evaluated on its average reward over N rounds with a range of opponents. 

Q-learning is a suitable method for this problem because it allows agents to learn effective policies through interaction with the environment without requiring a known model of the system or other players. Other algorithms may be better suited to this problem, but my goal is to better understand Q-learning which is why I chose it. 

## Important Files

Here are some of the files you may want to look at. Most of this repository deals with the environment and simulaton, which was existing code, so I wouldn't recommend spending time outside of these files. 

- [stagHare/run_example.py](./stagHare/run_example.py): Runs an example visualization of 5 games with all Q-learning agents.
- [stagHare/run_eval.py](./stagHare/run_eval.py): Evaluates the Q-learning agent against other types of agents.
- [stagHare/run_trainer.py](./stagHare/run_trainer.py): Trains the Q-learning agent.
- [stagHare/rl_agent/rl_agent_abstracted.py](./stagHare/rl_agent/rl_agent_abstracted.py): Controls what actions the Q-Learning agent takes.
- [stagHare/rl_agent/q_table_abstracted_manager.py](./stagHare/rl_agent/q_table_abstracted_manager.py): Stores and updates the Q-table as the agents are training.
- [stagHare/rl_agent/stag_greedy.py](stagHare/rl_agent/stag_greedy.py): Controls the Stag Greedy agent
- [stagHare/rl_agent/hare_greedy.py](stagHare/rl_agent/hare_greedy.py): Controls the Hare Greedy agent

## Setup

- (recommended) Create and activate your favorite virtual environment (I like conda)
- Install dependencies: `pip install -r requirements.txt`

Note: I am using Python 3.12.12

## Running The Code

There are two parts to look at in this project. 

The first is an example of the Q-learning agents playing the game. The agents (hunters) will be represented as blue squares on the grid. The stag will be black, and the hare will be grey. If all three hunters are adjacent to the stag and choose to hunt it, they will each be rewarded 30 points. If one hunter is adjacent to the hare and chooses to hunt it, they will be rewarded 10 points (this amount will be divided if more than one hunter is next to the hare). Note that the grid wraps around to avoid agents getting stuck, this means if an agent goes off one side of the grid they will show up on the other side. 

The second is an evaluation against other agents. The agents include RL: the Q-learning agent, Stag Greedy: only hunts stag, Hare Greedy: only hunts hare, and AlegAATr: an intelligent agent that was developed for game-theoretic games. Different combinations of agents play together for 100 games and their average reward is reported at the end. 

All scenarios:
- Scenario 0: RL, RL, RL
- Scenario 1: RL, RL, Stag Greedy
- Scenario 2: RL, RL, Hare Greedy
- Scenario 3: RL, RL, AlegAATr
- Scenario 4: RL, Stag Greedy, Stag Greedy
- Scenario 5: RL, Stag Greedy, Hare Greedy
- Scenario 6: RL, Stag Greedy, AlegAATr
- Scenario 7: RL, Hare Greedy, Hare Greedy
- Scenario 8: RL, Hare Greedy, AlegAATr
- Scenario 9: RL, AlegAATr, AlegAATr
- Scenario 10: Stag Greedy, Stag Greedy, Stag Greedy
- Scenario 11: Stag Greedy, Stag Greedy, Hare Greedy
- Scenario 12: Stag Greedy, Stag Greedy, AlegAATr
- Scenario 13: Stag Greedy, Hare Greedy, Hare Greedy
- Scenario 14: Stag Greedy, Hare Greedy, AlegAATr
- Scenario 15: Stag Greedy, AlegAATr, AlegAATr
- Scenario 16: Hare Greedy, Hare Greedy, Hare Greedy
- Scenario 17: Hare Greedy, Hare Greedy, AlegAATr
- Scenario 18: Hare Greedy, AlegAATr, AlegAATr
- Scenario 19: AlegAATr, AlegAATr, AlegAATr

### Running Example

- Run [stagHare/run_example.py](./stagHare/run_example.py) as a python module from the top project directiory: `python -m stagHare.run_example`.
- You should see a figure pop up that steps through a game. It will animate until the end of the game is reached, then it will stop moving.
- Press 'q' to close the figure and move to the next game.
- This will repeat for a total of 5 games. Press 'q' to close the figure and end the program after all 5 games.

### Running Evaluation

- Run [stagHare/run_eval.py](./stagHare/run_eval.py) as a python module from the top project directiory: `python -m stagHare.run_eval`.
- The simulation will take a moment to run, then a chart should pop up for one of the scenarios. Press 'q' to close it and move to the next scenario.
- I have it set to only show charts for a few scenarios, if you would like to see different scenarios, change line 174 `if scenario == 0 or scenario == 4 or scenario == 2 or scenario == 9: # only plot some of the scenarios` to include scenarios listed above that you want to see. 

## Results

The Q-learning agent essentially learned how to exclusivley hunt stag. This makes it so that it does really well when it is with fully cooperative agents. However, when against the Hare Greedy agents it does extremely bad. The AlegAATr agent also does not do very well against the Hare Greedy agent, but it does do slightly better against the Q-learning agent. 

The poor performance of the Q-learning agent against hare hunters is likely due to the fact that it was only trained in self-play. As it learned that there was a higher reward for hunting the stag, so did those it was hunting with and everyone just hunted the stag. I suspect if there were hare agents in the training, the Q-values would favor hunting hare more. Additionally, this was only played in state snapshots, so the agents had no history of what the other's previously did. 

## External Code Contributions

For this project, I built off of an existing envrionment and simulator. The majority of the code regarding the mechanics of the game and grid visualizations was produced by an outside party. Additionally, the AlegAATr algorithm was developed by another student, Ethan Pedersen, and is only used as a comparison. Everything regarding the Q-learning agent is original code.