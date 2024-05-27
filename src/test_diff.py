import argparse
import pprint
import random

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, reorganize

from diff.env import DiffEnv


TRAIN_CONFIG = {
    'players': 4,
    'rounds': 1,
    'reward_strategy': 'constant',
    'state_representation': 'compressed',
    'first_player_strategy': 'random',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}


def run(args):
    # Make environment
    # env = rlcard.make(
    #     args.env,
    #     config={
    #         'seed': 42,
    #     }
    # )

    env = DiffEnv(TRAIN_CONFIG)

    # Seed numpy, torch, random
    set_seed(42)

    # Set agents
    agent = RandomAgent(num_actions=env.num_actions)
    env.set_agents([agent for _ in range(env.num_players)])

    # Generate data from the environment
    trajectories, player_wins = env.run(is_training=True)
    test = reorganize(trajectories, player_wins)
    # Print out the trajectories
    print('\nTrajectories:')
    # for i, t in enumerate(trajectories):
    #     print("==================== Trajectory %d ================" % i)
    #     for s in t[::2]:
    #         pprint.pprint(s['raw_obs'])
    print(trajectories)
    print('\nSample raw observation:')
    pprint.pprint(trajectories[0][0]['raw_obs'])
    print('\nSample raw legal_actions:')
    pprint.pprint(trajectories[0][0]['raw_legal_actions'])


if __name__ == '__main__':
    # parser = argparse.ArgumentParser("Random example in RLCard")
    # parser.add_argument(
    #     '--env',
    #     type=str,
    #     default='leduc-holdem',
    #     choices=[
    #         'blackjack',
    #         'leduc-holdem',
    #         'limit-holdem',
    #         'doudizhu',
    #         'mahjong',
    #         'no-limit-holdem',
    #         'uno',
    #         'gin-rummy',
    #         'bridge',
    #     ],
    # )
    #
    # run(parser.parse_args())
    run(None)
