import numpy as np
import warnings
warnings.filterwarnings('ignore')

from judgement.env import JudgementEnv
from rlcard.agents.random_agent import RandomAgent

env = JudgementEnv({'game_num_players': 4})
agent = RandomAgent(num_actions=env.num_actions)
env.set_agents([agent]*4)
payoffs = []
for _ in range(10000):
    _, p = env.run(is_training=False)
    payoffs.append(p)
payoffs = np.mean(payoffs, axis=0)
print("Random avg payoff:", payoffs)

print("\nHeuristic Perfect Bidding Payoff Calculation:")
# If everyone perfectly bids 3.25 average tricks
perfect_score = 10 + 3.25
perfect_payoff = perfect_score / 299.0
print("Perfect play average score:", perfect_score)
print("Perfect play average payoff:", perfect_payoff)
