import random

from rlcard.agents import RandomAgent
from rlcard.utils import tournament

from diff.env import DiffEnv
from diff.prediction import FixedPredictionStrategy

DEFAULT_GAME_CONFIG = {
    'players': 4,
    'rounds': 9,
    'prediction_strategy': FixedPredictionStrategy(157 // 4),
    'reward_strategy': 'winner_takes_all',
    'state_representation': 'compressed',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}

# Run script: python -m src/diff test_agent.py


def evaluate_performance():
    env = DiffEnv(DEFAULT_GAME_CONFIG)

    agent = None  # Todo: Add agent here
    agents = [agent] + [RandomAgent(num_actions=env.num_actions) for _ in range(1, env.num_players)]
    env.set_agents(agents)
    result = tournament(env, 100)
    print("Agent wins %d%% of the games" % (result[0] * 100))


if __name__ == '__main__':
    evaluate_performance()
