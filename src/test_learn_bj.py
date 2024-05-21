import os

import torch

import rlcard
from rlcard.agents import RandomAgent,  DQNAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)


def train(
        seed=0,
        log_dir='./logs_uno',
        save_every=1000,
        evaluate_every=1000,
        num_episodes=20000,
        num_eval_games=100
):
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(seed)

    # Make the environment with seed
    env = rlcard.make(
        'uno',
        config={
            'seed': seed,
        }
    )

    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[128, 64, 64],
        device=device,
        save_path=log_dir,
        save_every=save_every,
        replay_memory_init_size=200,
        batch_size=128,
        epsilon_decay_steps=50000
    )

    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    with Logger(log_dir) as logger:
        for episode in range(num_episodes):

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        num_eval_games,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, "DQN")

    # Save model
    save_path = os.path.join(log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    train()
