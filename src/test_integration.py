import rlcard
from rlcard.agents import RandomAgent

from diff.env import DiffEnv
from diff.prediction import FixedPredictionStrategy


DEFAULT_GAME_CONFIG = {
    'players': 4,
    'rounds': 2,
    'prediction_strategy': FixedPredictionStrategy(157 // 4),
    'reward_strategy': 'winner_takes_all',
    'state_representation': 'compressed',
    'allow_step_back': False,
    'seed': 0
}


def test_integration():
    env = DiffEnv(DEFAULT_GAME_CONFIG)

    env.set_agents([RandomAgent(num_actions=env.num_actions) for _ in range(4)])

    print(env.num_actions)
    print(env.num_players)
    print(env.state_shape)
    print(env.action_shape)

    trajectories, payoffs = env.run()
    print(trajectories)
    print(payoffs)


if __name__ == '__main__':
    test_integration()
