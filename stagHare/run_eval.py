from tqdm import tqdm

from offlineSimStuff.runningTools.runnerHelper import create_jhg_sim, create_total_order, create_jhg_engine
from stagHare.agents.cabAgentThing import CabAgent
from stagHare.agents.fetcherBot import FetcherBot
from stagHare.agents.rl_agent.hare_greedy import HareGreedyAgent
from stagHare.agents.rl_agent.q_table_abstracted_manager import QTableAbstractedManager
from stagHare.agents.rl_agent.q_table_manager import QTableManager
from stagHare.agents.rl_agent.stag_greedy import StagGreedyAgent
from stagHare.environment.world import StagHare
from stagHare.environment.allocationTranslator import allocation_to_movement, movement_to_allocation
from stagHare.loggingStuff.stagHareLogger import stagHareLogger
from stagHare.visualziationTools.batchLogger import BatchLogger
from stagHare.visualziationTools.inviduvalRoundGrapher import IndividualRoundGrapher
from stagHare.visualziationTools.gameGrapher import GameGrapher
from stagHare.visualziationTools.gameLogger import GameLogger


# so what do we actually need to do
# lets create some cab agents
# and get them to play this fetcher
# we also need to work on the trnaslation machinery as well
# so that will be interesting.
# this is going ot be strange bc the simulator is VERY different from what I have worked with before
# the SC sim I created and the JHG sim was sort of built for cab agents
# this one has not been built for either of those things.


from stagHare.runnerHelper import *

def run_game(q_table_manager, agent_scenario=0):
    height, width = 4, 4 # lets start there, not too big but there.
    forcedRandom = True
    random_agents = True # better for human distribution

    num_players = 3 # as dictated by the stag hare thing
    num_humans = 0 # yeah...

    num_attempts = 1 # don't worry about this

    # with that out of the way, its time to angrily insert the logger in here.
    curr_logger = stagHareLogger()

    # agent_names = ["gen_199.csv", "6x6Round1.csv", "gen_Z.csv", "homoJHGSelfPlay.csv", "homoSCselfPlayMFalse.csv", "homoSCselfPlayMTrue.csv", "mixedJHGSelfPlay.csv", "mixedSCselfPlayMFalse.csv", "mixedSCselfPlayMTrue.csv"]
    agent_names = ["6x6Round3.csv"]

    # keep this as cab for now. we will figure out the rest later.
    # this only works assuming that we are doing self play. use the agent scenario instead.
    agent_type = 3 # -1 is ALLEGATR, 0 is a random agent, 1 is the hare greedy agent, 2 is stag greedy agent, 3 is CAB
    # 0 is standard, 1 is nothing, 2 is 2 of whatever bots with a fectcher bot, 3 is a cab with 2 stag and 4 is a cab with 2 hares.
                                                                     # 5 is 2 cabs with 1 stag and 6 is 2 cabs with 1 hare.

    scores = []

    for agent_name in agent_names:
        # print("Agent name: " + agent_name)
        current_batch_logger = BatchLogger()

        for attempt in range(num_attempts):
            current_game_logger = GameLogger(height, width) # need this per game, not per batch.
            hunters = create_rl_hunters_from_scenario(agent_scenario=agent_scenario, q_table_manager=q_table_manager, epsilon=0) # Don't explore when running the eval
            current_round_grapher = IndividualRoundGrapher()
            while True:
                stag_hare = StagHare(height, width, hunters)
                if not stag_hare.is_over():
                    break

            # does this suck? possibly.
            stag_hare.state.hunting_hare_map = {"R"+str(i) : 2 for i in range(3)} # value that it can never be, sort of a NAN. 

            # just run the fetcher.
            new_score, intents, rewards = run_trial_graphing(stag_hare, current_round_grapher, current_game_logger, graph=False)
            scores.append(new_score)
            current_batch_logger.add_game(stag_hare)

            # game_grapher = GameGrapher(stag_hare)

            # game_grapher.playback_game(current_game_logger)
            # game_grapher.create_game_graph(current_game_logger)
        cooperation_score, scores_per_player = process_scores(scores)

        # curr_logger.add_information_game(agent_scenario, cooperation_score, scores_per_player, agent_name)
    return cooperation_score, scores_per_player, rewards

def create_rl_hunters_from_scenario(agent_scenario=0, q_table_manager=None, epsilon=0.1):
    new_hunters = []

    if agent_scenario == 1:
        new_name = "R0"
        new_hunters.append(QLearningAbstractedAgent(0, new_name, q_table_manager, epsilon=epsilon))
        new_name = "R1"
        new_hunters.append(StagGreedyAgent(1, new_name))
        new_name = "R2"
        new_hunters.append(StagGreedyAgent(2, new_name))
    elif agent_scenario == 2:
        new_name = "R0"
        new_hunters.append(QLearningAbstractedAgent(0, new_name, q_table_manager, epsilon=epsilon))
        new_name = "R1"
        new_hunters.append(StagGreedyAgent(1, new_name))
        new_name = "R2"
        new_hunters.append(HareGreedyAgent(2, new_name))
    elif agent_scenario == 3:
        new_name = "R0"
        new_hunters.append(QLearningAbstractedAgent(0, new_name, q_table_manager, epsilon=epsilon))
        new_name = "R1"
        new_hunters.append(HareGreedyAgent(1, new_name))
        new_name = "R2"
        new_hunters.append(HareGreedyAgent(2, new_name))
    elif agent_scenario == 4:
        new_name = "R0"
        new_hunters.append(QLearningAbstractedAgent(0, new_name, q_table_manager, epsilon=epsilon))
        new_name = "R1"
        new_hunters.append(QLearningAbstractedAgent(1, new_name, q_table_manager, epsilon=epsilon))
        new_name = "R2"
        new_hunters.append(HareGreedyAgent(2, new_name))
    elif agent_scenario == 5:
        new_name = "R0"
        new_hunters.append(QLearningAbstractedAgent(0, new_name, q_table_manager, epsilon=epsilon)) 
        new_name = "R1"
        new_hunters.append(QLearningAbstractedAgent(1, new_name, q_table_manager, epsilon=epsilon))
        new_name = "R2"
        new_hunters.append(StagGreedyAgent(2, new_name))
    elif agent_scenario == 6:
        new_name = "R0"
        new_hunters.append(QLearningAbstractedAgent(0, new_name, q_table_manager, epsilon=epsilon))
        new_name = "R1"
        new_hunters.append(AlegAATr(name=new_name, lmbda=0.0, ml_model_type='knn', enhanced=True))
        new_name = "R2"
        new_hunters.append(AlegAATr(name=new_name, lmbda=0.0, ml_model_type='knn', enhanced=True))
    elif agent_scenario == 7:
        new_name = "R0"
        new_hunters.append(QLearningAbstractedAgent(0, new_name, q_table_manager, epsilon=epsilon))
        new_name = "R1"
        new_hunters.append(QLearningAbstractedAgent(1, new_name, q_table_manager, epsilon=epsilon))
        new_name = "R2"
        new_hunters.append(AlegAATr(name=new_name, lmbda=0.0, ml_model_type='knn', enhanced=True))
    else:
        new_name = "R0"
        new_hunters.append(QLearningAbstractedAgent(0, new_name, q_table_manager, epsilon=epsilon))
        new_name = "R1"
        new_hunters.append(QLearningAbstractedAgent(1, new_name, q_table_manager, epsilon=epsilon))
        new_name = "R2"
        new_hunters.append(QLearningAbstractedAgent(2, new_name, q_table_manager, epsilon=epsilon))

    return new_hunters

if __name__ == '__main__':
    print("RUNNING SIMULATION...")
    start_time = time.time()
    q_tabel_manager = QTableAbstractedManager(q_table_file='stagHare/agents/rl_agent/q_table_4x4_abstracted_stag_only.txt')

    # 0: all RL agents; 1: RL, Stag, Stag; 2: RL, Stag, Hare; 3: RL, Hare, Hare; 4: Hare, RL, RL; 5: Stag, RL, RL; 6: RL, AlegAATr, AlegAATr; 7: RL, RL, AlegAATr
    scenarios = [ "All RL Agents",  "One RL Agent with 2 Stag Greedy Agents",  "One RL Agent with 1 Stag Greedy Agent and 1 Hare Greedy Agent", "One RL Agent with 2 Hare Greedy Agents", "Two RL Agent with 1 Hare Greedy Agents", "Two RL Agent with 1 Stag Greedy Agents", "One RL Agent with 2 AlegAATr Agents", "Two RL Agents with 1 AlegAATr Agent"]

    for scenario, scenario_title in enumerate(scenarios):
        cooporation = []
        scores = []
        rewards_list = []
        
        for i in tqdm(range(100)):
            # print("RUNNING GAME ", i)
            try:
                cooporation_score, scores_per_player, rewards = run_game(q_tabel_manager, agent_scenario = scenario)
                cooporation.append(cooporation_score)
                scores.append(tuple(scores_per_player))
                rewards_list.append(rewards)
                
            except Exception as e:
                print(f"Error in game {i}: {e}")

        # print(f"rewards from the last game of scenario {scenario_title}: {rewards}")
        player_1_rewards = [reward[2] for reward in rewards_list]
        player_2_rewards = [reward[3] for reward in rewards_list]
        player_3_rewards = [reward[4] for reward in rewards_list]
        
        # # [none, hare, stag]
        # player_1_scores = [0, 0, 0]
        # player_2_scores = [0, 0, 0]
        # player_3_scores = [0, 0, 0]

        # for score in scores:
        #     player_1_scores = np.array(player_1_scores) + np.array(score[0])
        #     player_2_scores = np.array(player_2_scores) + np.array(score[1])
        #     player_3_scores = np.array(player_3_scores) + np.array(score[2])
        
        # print(f"Total scores for scenario {scenario_title}:")
        # print(f"Player 1: None: {player_1_scores[0]}, Hare  {player_1_scores[1]}, Stag {player_1_scores[2]}")
        # print(f"Player 2: None: {player_2_scores[0]}, Hare  {player_2_scores[1]}, Stag {player_2_scores[2]}")
        # print(f"Player 3: None: {player_3_scores[0]}, Hare  {player_3_scores[1]}, Stag {player_3_scores[2]}")

        print(f"Average rewards for scenario {scenario_title}:")
        print(f"Player 1: {np.mean(player_1_rewards):.4f}")
        print(f"Player 2: {np.mean(player_2_rewards):.4f}")
        print(f"Player 3: {np.mean(player_3_rewards):.4f}")

        print(f"Average cooperation score for scenario {scenario}: {sum(cooporation)/len(cooporation):.4f}")
    
    print("\nSIMULATION COMPLETE")
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    