import random

import torch

from rlcard.agents import DQNAgent, RandomAgent
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


def evaluate_performance():
    env = DiffEnv(DEFAULT_GAME_CONFIG)

    agent = DQNAgent.from_checkpoint(checkpoint=torch.load("log/checkpoint_dqn.pt"))
    agents = [agent] + [RandomAgent(num_actions=env.num_actions) for _ in range(1, env.num_players)]
    env.set_agents(agents)
    result = tournament(env, 100)
    print(result)


if __name__ == '__main__':
    evaluate_performance()
