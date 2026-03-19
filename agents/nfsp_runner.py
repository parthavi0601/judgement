"""
NFSP Training Runner for Judgement card game.
Uses rlcard.agents.nfsp_agent.NFSPAgent with the custom JudgementEnv.
"""

import os
import torch
import numpy as np

from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import reorganize


def create_nfsp_agents(env, hidden_layers=None, device=None):
    """Create NFSP agents for all players."""
    if hidden_layers is None:
        hidden_layers = [128, 128]
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    agents = []
    for _ in range(env.num_players):
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=hidden_layers,
            reservoir_buffer_capacity=50000,
            anticipatory_param=0.1,
            batch_size=128,
            train_every=64,
            rl_learning_rate=0.01,
            sl_learning_rate=0.005,
            min_buffer_size_to_learn=256,
            q_replay_memory_size=50000,
            q_replay_memory_init_size=256,
            q_update_target_estimator_every=1000,
            q_discount_factor=0.99,
            q_epsilon_start=0.08,
            q_epsilon_end=0.003,
            q_epsilon_decay_steps=int(1e5),
            q_batch_size=128,
            q_train_every=64,
            q_mlp_layers=[128, 128],
            evaluate_with='average_policy',
            device=device,
        )
        agents.append(agent)
    return agents


def train_nfsp(env, num_episodes=10000, evaluate_every=500, save_dir=None, verbose=True):
    """
    Train NFSP agents on the Judgement environment.
    
    Args:
        env: JudgementEnv instance
        num_episodes: Total training episodes
        evaluate_every: Evaluate performance every N episodes
        save_dir: Directory to save checkpoints
        verbose: Print progress
    
    Returns:
        agents: Trained NFSP agents
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    agents = create_nfsp_agents(env, device=device)
    env.set_agents(agents)

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards_log = []

    for episode in range(1, num_episodes + 1):
        # Sample episode policy for each agent
        for agent in agents:
            agent.sample_episode_policy()

        # Run one episode
        trajectories, payoffs = env.run(is_training=True)

        # Use rlcard's reorganize() to convert raw trajectories into
        # per-player lists of (state, action, reward, next_state, done) tuples.
        # Each state/next_state is a full dict with 'obs' and 'legal_actions'
        # keys, which is exactly what DQNAgent.feed() expects.
        trajectories = reorganize(trajectories, payoffs)

        for pid in range(env.num_players):
            for ts in trajectories[pid]:
                agents[pid].feed(ts)

        rewards_log.append(payoffs)

        # Evaluate
        if verbose and episode % evaluate_every == 0:
            avg_payoffs = np.mean(rewards_log[-evaluate_every:], axis=0)
            print(f'\n\nEpisode {episode}/{num_episodes}')
            for pid in range(env.num_players):
                print(f'  Player {pid}: avg payoff = {avg_payoffs[pid]:.4f}')

        # Save checkpoint
        if save_dir and episode % (evaluate_every * 4) == 0:
            for pid, agent in enumerate(agents):
                agent.save_checkpoint(save_dir, filename=f'nfsp_agent_{pid}_ep{episode}.pt')
            if verbose:
                print(f'  Checkpoints saved at episode {episode}')

    return agents


def evaluate_agents(env, agents, num_episodes=100):
    """Evaluate trained agents via random games."""
    env.set_agents(agents)
    payoffs_sum = np.zeros(env.num_players)

    for _ in range(num_episodes):
        _, payoffs = env.run(is_training=False)
        payoffs_sum += payoffs

    avg = payoffs_sum / num_episodes
    return avg
