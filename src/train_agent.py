import random
import os
import argparse

import torch

from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import get_device, Logger, reorganize, tournament, plot_curve

from diff import FixedPredictionStrategy, DiffEnv

TRAIN_CONFIG = {
    'players': 4,
    'rounds': 1,
    'reward_strategy': 'default',
    'first_player_strategy': 'random',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}

TEST_CONFIG_WINRATE = {
    'players': 4,
    'rounds': 9,
    'reward_strategy': 'winner_takes_all',
    'first_player_strategy': 'random',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}

TEST_CONFIG_POINTS = {
    'players': 4,
    'rounds': 9,
    'reward_strategy': 'default',
    'first_player_strategy': 'random',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}


def train_agent(args: argparse.Namespace) -> None:
    """
    Trains an agent.

    :param args: Command line arguments
    """
    # Gets torch device
    device = get_device()

    # Prepares the train and test configs
    train_cfg = TRAIN_CONFIG.copy()
    test_cfg_winrate = TEST_CONFIG_WINRATE.copy()
    test_cfg_points = TEST_CONFIG_POINTS.copy()

    for cfg in [train_cfg, test_cfg_winrate, test_cfg_points]:
        if args.prediction != 'rand':
            if args.prediction == 'max':
                points = 157
            elif args.prediction == 'min':
                points = 0
            else:
                points = 157 // 4
            cfg['prediction_strategy'] = FixedPredictionStrategy(points)

        cfg['state_representation'] = args.state_rep

    # Prepares the environments
    train_env = DiffEnv(train_cfg)
    test_env_winrate = DiffEnv(test_cfg_winrate)
    test_env_points = DiffEnv(test_cfg_points)

    # Prepares the dqn agent
    agent = DQNAgent(
        num_actions=train_env.num_actions,
        state_shape=train_env.state_shape[0],
        mlp_layers=[256, 128, 64] if args.state_rep == 'default' else [128, 64, 64],
        device=device,
        save_path=args.out_dir,
        save_every=args.save_every,
        replay_memory_init_size=args.init_memory_size,
        batch_size=args.batch_size,
        epsilon_decay_steps=args.epsilon_decay_steps * 9,  # Each game in training has 9 steps
        train_every=args.train_every,
    )

    # Create loggers
    winrate_log = os.path.join(args.out_dir, 'winrate')
    point_log = os.path.join(args.out_dir, 'pts')

    # Initializing random agents
    agents = [agent] + [RandomAgent(num_actions=train_env.num_actions) for _ in range(1, train_env.num_players)]
    train_env.set_agents(agents)
    test_env_winrate.set_agents(agents)
    test_env_points.set_agents(agents)

    # Starting loggers
    with Logger(winrate_log) as winrate_logger, Logger(point_log) as point_logger:
        for episode in range(args.episodes):

            # Collecting trajectories
            trajectories, payoffs = train_env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)

            # Feeding trajectories to agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluating the performance
            if episode % args.evaluate_every == 0:
                print()
                print("Evaluation winrate:")
                winrate = tournament(test_env_winrate, args.evaluate_games)[0]
                winrate_logger.log_performance(episode, winrate)
                print("Evaluation points:")
                points = tournament(test_env_points, args.evaluate_games)[0]
                point_logger.log_performance(episode, points)

        # Evaluating final performance
        print()
        print("Evaluation winrate:")
        winrate_logger.log_performance(episode, tournament(test_env_winrate, args.evaluate_games)[0])
        print("Evaluation points:")
        point_logger.log_performance(episode, tournament(test_env_points, args.evaluate_games)[0])

    # Plotting progress curves
    for logger in [winrate_logger, point_logger]:
        csv_path, fig_path = logger.csv_path, logger.fig_path
        plot_curve(csv_path, fig_path, 'dqn')

    # Save model
    save_path = os.path.join(args.out_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Differenzler trainer")

    parser.add_argument(
        '--out-dir', '-o',
        type=str,
        required=True,
        help='Where to save the output of the training.'
    )

    parser.add_argument(
        "--state-rep", '-s',
        type=str,
        choices=['default', 'compressed', 'enhanced_small', 'enhanced_large'],
        default='compressed',
        help='How should the state be parsed for the DQN.'
    )

    parser.add_argument(
        "--prediction", '-p',
        type=str,
        choices=['max', 'min', 'avg', 'rand'],
        default='rand',
        help='Which prediction strategy to use.'
    )

    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=50000,
        help='How many episodes to train the model for.'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=265,
        help='Batch size for the training.'
    )

    parser.add_argument(
        '--epsilon-decay-steps',
        type=int,
        default=25000,
        help='Over how many epochs to decay epsilon.'
    )

    parser.add_argument(
        '--train-every',
        type=int,
        default=1,
        help='How often to train the model.'
    )

    parser.add_argument(
        '--save-every',
        type=int,
        default=1000,
        help='After how many episodes to create a checkpoint'
    )

    parser.add_argument(
        '--init-memory-size',
        type=int,
        default=900,
        help='Initial size of the replay memory.'
    )

    parser.add_argument(
        '--evaluate-every',
        type=int,
        default=1000,
        help='After how many episodes to evaluate the model'
    )

    parser.add_argument(
        '--evaluate-games',
        type=int,
        default=100,
        help='How many games to run to evaluate the model'
    )

    parser.add_argument(
        '-m',
        type=str,
        required=True,
        help='Allows to register modules. Please pass src/diff as argument here',
    )

    cli_args = parser.parse_args()
    train_agent(cli_args)
