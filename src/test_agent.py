import random

from rlcard.agents import RandomAgent
from rlcard.utils import tournament, get_device

from diff.env import DiffEnv
from agent.greedy_agent import GreedyAgent
from agent.trained_dqn_agent import DqnAgent
from agent.look_ahead_dqn_agent import LookaheadDqnAgent

TEST_CONFIG_1 = {
    'players': 4,
    'rounds': 9,
    # 'prediction_strategy': FixedPredictionStrategy(157 // 4),
    'reward_strategy': 'winner_takes_all',
    'state_representation': 'compressed',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}

TEST_CONFIG_2 = {
    'players': 4,
    'rounds': 9,
    # 'prediction_strategy': FixedPredictionStrategy(157 // 4),
    'reward_strategy': 'default',
    'state_representation': 'compressed',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}

# Run script: python -m src/diff test_agent.py


def evaluate_performance(
        semi_greedy=True,
        rounds=10
):
    device = get_device()
    print("Testing %s agent for %d rounds: " % ("semi-greedy" if semi_greedy else "greedy", rounds))
    env = DiffEnv(TEST_CONFIG_1)
    env2 = DiffEnv(TEST_CONFIG_2)

    # agent = GreedyAgent(semi_greedy=semi_greedy)
    # agent = DqnAgent(path="./runs/compressed/log_rand/model.pth", device=device)
    agent = LookaheadDqnAgent(path="./runs/compressed/log_rand/model.pth", device=device)
    agents = [agent] + [RandomAgent(num_actions=env.num_actions) for _ in range(1, env.num_players)]
    env.set_agents(agents)
    env2.set_agents(agents)
    result = tournament(env, rounds)
    print("Agent wins %f%% of the games" % (result[0] * 100))
    result = tournament(env2, rounds)
    print("Agent is off by %f pts on avg" % (result[0] * -157))


if __name__ == '__main__':
    evaluate_performance()
