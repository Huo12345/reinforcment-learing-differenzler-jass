import random
import os

import torch

from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import get_device, Logger, reorganize, tournament, plot_curve

from diff.env import DiffEnv

TRAIN_CONFIG = {
    'players': 4,
    'rounds': 1,
    # 'prediction_strategy': FixedPredictionStrategy(157 // 4),
    'reward_strategy': 'default',
    'state_representation': 'enhanced_large',
    'first_player_strategy': 'random',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}

TEST_CONFIG = {
    'players': 4,
    'rounds': 9,
    # 'prediction_strategy': FixedPredictionStrategy(157 // 4),
    'reward_strategy': 'winner_takes_all',
    'state_representation': 'enhanced_large',
    'first_player_strategy': 'random',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}

TEST_CONFIG_2 = {
    'players': 4,
    'rounds': 9,
    # 'prediction_strategy': FixedPredictionStrategy(157 // 4),
    'reward_strategy': 'default',
    'state_representation': 'enhanced_large',
    'first_player_strategy': 'random',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}


def train_agent(
        log_dir='log',
        save_every=1000,
        evaluate_every=1000,
        num_episodes=1000,
        num_eval_games=100
):
    device = get_device()

    train_env = DiffEnv(TRAIN_CONFIG)
    test_env = DiffEnv(TEST_CONFIG)
    test_env2 = DiffEnv(TEST_CONFIG_2)

    agent = DQNAgent(
        num_actions=train_env.num_actions,
        state_shape=train_env.state_shape[0],
        # mlp_layers=[256, 128, 64],
        mlp_layers=[128, 64, 64],
        device=device,
        save_path=log_dir,
        save_every=save_every,
        replay_memory_init_size=900,
        batch_size=256,
        epsilon_decay_steps=250000
    )

    performance_1_log = os.path.join(log_dir, 'winrate')
    performance_2_log = os.path.join(log_dir, 'pts')

    agents = [agent] + [RandomAgent(num_actions=train_env.num_actions) for _ in range(1, train_env.num_players)]
    train_env.set_agents(agents)
    test_env.set_agents(agents)
    test_env2.set_agents(agents)

    with Logger(performance_1_log) as logger, Logger(performance_2_log) as logger_2:
        for episode in range(num_episodes):

            trajectories, payoffs = train_env.run(is_training=True)

            trajectories = reorganize(trajectories, payoffs)

            for ts in trajectories[0]:
                agent.feed(ts)

            if episode % evaluate_every == 0:
                results_win = tournament(test_env, num_eval_games)
                logger.log_performance(episode, results_win[0])
                print(results_win)
                results_pts = tournament(test_env2, num_eval_games)
                logger_2.log_performance(episode, results_pts[0])
                print(results_pts)

        logger.log_performance(episode, tournament(test_env, num_eval_games)[0])
        logger_2.log_performance(episode, tournament(test_env2, num_eval_games)[0])

    csv_path, fig_path = logger.csv_path, logger.fig_path
    plot_curve(csv_path, fig_path, 'dqn')

    csv_path, fig_path = logger_2.csv_path, logger_2.fig_path
    plot_curve(csv_path, fig_path, 'dqn')

    # Save model
    save_path = os.path.join(log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    train_agent()
