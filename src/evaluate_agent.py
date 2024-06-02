import sys
import random
import argparse

import torch

from rlcard.agents import RandomAgent
from rlcard.utils import get_device, tournament

from diff import DiffEnv, FixedPredictionStrategy

from agent.greedy_agent import GreedyAgent
from agent.trained_dqn_agent import DqnAgent
from agent.look_ahead_dqn_agent import LookaheadDqnAgent

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


def evaluate_agent(args: argparse.Namespace) -> None:
    """
    Evaluates the performance of an agent
    :param args: Command line arguments
    """
    # Gets torch device
    device = torch.device(args.device) if args.device is not None else get_device()

    # Prepares the test configs
    test_cfg_winrate = TEST_CONFIG_WINRATE.copy()
    test_cfg_points = TEST_CONFIG_POINTS.copy()

    for cfg in [test_cfg_winrate, test_cfg_points]:
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
    test_env_winrate = DiffEnv(test_cfg_winrate)
    test_env_points = DiffEnv(test_cfg_points)

    # Prepares the agent
    if args.type == 'greedy':
        agent = GreedyAgent(semi_greedy=False)
    elif args.type == 'semi-greedy':
        agent = GreedyAgent(semi_greedy=False)
    elif args.type == 'dqn':
        agent = DqnAgent(args.weights, device)
    elif args.type == 'lookahead':
        agent = LookaheadDqnAgent(args.weights, device, args.lookahead_steps, args.lookahead_samples)
    else:
        raise Exception('Unknown agent type %s' % args.type)

    agents = ([agent] +
              [RandomAgent(num_actions=test_env_winrate.num_actions) for _ in range(1, test_env_winrate.num_players)])

    test_env_winrate.set_agents(agents)
    test_env_points.set_agents(agents)

    # Evaluating the agents performance
    result = tournament(test_env_winrate, args.eval_games)
    print("Agent wins %f%% of the games" % (result[0] * 100))
    result = tournament(test_env_points, args.eval_games)
    print("Agent is off by %f pts on avg" % (result[0] * -157))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Differenzler evaluation")

    parser.add_argument(
        '--type', '-t',
        type=str,
        required=True,
        choices=['greedy', 'semi-greedy', 'dqn', 'lookahead'],
        help='Type of agent to evaluate. Note that lookahead only works with compressed state dqn models.'
    )

    parser.add_argument(
        '--weights', '-w',
        type=str,
        default=None,
        help='Path to model weights to evaluate. Required for type dqn and lookahead.'
    )

    parser.add_argument(
        '--eval-games', '-e',
        type=int,
        default=1000,
        help='Number of games to evaluate.'
    )

    parser.add_argument(
        "--prediction", '-p',
        type=str,
        choices=['max', 'min', 'avg', 'rand'],
        default='rand',
        help='Which prediction strategy to use.'
    )

    parser.add_argument(
        "--state-rep", '-s',
        type=str,
        choices=['default', 'compressed', 'enhanced_small', 'enhanced_large'],
        default='compressed',
        help='How should the state be parsed for the DQN. Must match with the mode that was used to train the model'
    )

    parser.add_argument(
        '--lookahead-steps',
        type=int,
        default=1,
        help='How many steps a lookahead agents should look ahead'
    )

    parser.add_argument(
        '--lookahead-samples',
        type=int,
        default=10,
        help='How many samples the lookahead agent should take at each step'
    )

    parser.add_argument(
        '--device', '-d',
        type=str,
        help='Torch device.'
    )

    parser.add_argument(
        '-m',
        type=str,
        required=True,
        help='Allows to register modules. Please pass src/diff as argument here',
    )

    cli_args = parser.parse_args()

    if cli_args.weights is None and (cli_args.type == 'dqn' or cli_args.type == 'lookahead'):
        print("Agents of type dqn and lookahead require model weights passed. Use param -h for help.", sys.stderr)
        exit(1)

    if cli_args.type == 'lookahead' and cli_args.state_rep != 'compressed':
        print('Cannot use %s state representation with lookahead. Please choose a different model' % cli_args.state_rep,
              sys.stderr)
        exit(1)

    evaluate_agent(cli_args)
