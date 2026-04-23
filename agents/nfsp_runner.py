"""
NFSP Training Runner for Judgement card game.
Uses rlcard.agents.nfsp_agent.NFSPAgent with the custom JudgementEnv.
"""

import os
import csv
import torch
import numpy as np

from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import reorganize


def reorganize_dense(trajectories, payoffs):
    """
    Reorganize the trajectory to provide dense trick-by-trick rewards.

    Each player transition gets a reward that is the change in accumulated
    dense rewards between their consecutive states.  On the terminal step
    the normalized game payoff is added so the Q-network also learns from
    the final score (bid correctness bonus / penalty).

    Transition format fed to agent.feed():
        [state, action, reward (float), next_state, done (bool)]
    """
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]

    for player in range(num_players):
        for i in range(0, len(trajectories[player]) - 2, 2):
            state = trajectories[player][i]
            next_state = trajectories[player][i + 2]

            done = (i == len(trajectories[player]) - 3)

            # Per-step dense reward: change in accumulated dense rewards
            curr_dense = state.get('dense_rewards', [0.0] * num_players)[player]
            next_dense = next_state.get('dense_rewards', [0.0] * num_players)[player]
            reward = next_dense - curr_dense

            # On the terminal step, add the final game payoff so the
            # Q-network also receives the end-of-round score signal
            if done:
                reward += payoffs[player]

            transition = trajectories[player][i:i + 3].copy()
            transition.insert(2, reward)
            transition.append(done)

            new_trajectories[player].append(transition)
    return new_trajectories


def patch_agent_losses(agent):
    """Monkey patch to track latest losses and PER beta for logging."""
    agent.latest_sl_loss = 0.0
    agent.latest_rl_loss = 0.0
    agent.latest_per_beta = 0.0

    original_train_sl = agent.train_sl
    def tracked_train_sl(*args, orig_fn=original_train_sl, agent_ref=agent, **kwargs):
        sl_loss = orig_fn(*args, **kwargs)
        if sl_loss is not None:
            agent_ref.latest_sl_loss = sl_loss
        return sl_loss
    agent.train_sl = tracked_train_sl

    original_update = agent._rl_agent.q_estimator.update
    def tracked_update(*args, orig_fn=original_update, agent_ref=agent, **kwargs):
        rl_loss = orig_fn(*args, **kwargs)
        agent_ref.latest_rl_loss = rl_loss
        # Capture PER beta from memory if using PER
        if hasattr(agent_ref._rl_agent.memory, 'current_beta'):
            agent_ref.latest_per_beta = agent_ref._rl_agent.memory.current_beta
        return rl_loss
    agent._rl_agent.q_estimator.update = tracked_update
    return agent


def create_nfsp_agents(env, hidden_layers=None, device=None, rl_learning_rate=0.001, sl_learning_rate=0.005, use_per=True, anticipatory_param=0.15, q_epsilon_decay_steps=1400000):
    """Create NFSP agents for all players."""
    if hidden_layers is None:
        hidden_layers = [1024, 512, 256]
    if device is None:
        device = torch.device('cpu')

    agents = []
    for _ in range(env.num_players):
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=hidden_layers,
            reservoir_buffer_capacity=350000, # Scaled down to prevent OOM
            anticipatory_param=anticipatory_param,
            batch_size=512,
            train_every=32,
            rl_learning_rate=rl_learning_rate,
            sl_learning_rate=sl_learning_rate,
            min_buffer_size_to_learn=256,
            q_replay_memory_size=350000, # Scaled down to prevent OOM
            q_replay_memory_init_size=256,
            q_update_target_estimator_every=500,
            q_discount_factor=0.995,  # Critical for terminal bid reward credit assignment in Judgement
            q_epsilon_start=1.0,
            q_epsilon_end=0.05,
            q_epsilon_decay_steps=q_epsilon_decay_steps,
            q_train_every=32,
            q_mlp_layers=hidden_layers,
            evaluate_with='average_policy',
            device=device,
            use_per=use_per,
        )

        patch_agent_losses(agent)
        agents.append(agent)
    return agents


def train_nfsp(env, num_episodes=10000, evaluate_every=500, checkpoint_every=None, save_dir=None, verbose=True, agents=None, start_episode=0, rl_learning_rate=0.01, sl_learning_rate=0.005, use_per=True):
    """
    Train NFSP agents on the Judgement environment.

    Args:
        env: JudgementEnv instance
        num_episodes: Total training episodes
        evaluate_every: Evaluate performance every N episodes
        checkpoint_every: Save checkpoints every N episodes
        save_dir: Directory to save checkpoints
        verbose: Print progress
        agents: Optional existing agents to resume training
        start_episode: Episode number to start/resume from
        use_per: If True, use Prioritized Experience Replay for the DQN buffer.

    Returns:
        agents: Trained NFSP agents
    """
    if checkpoint_every is None:
        checkpoint_every = evaluate_every * 4

    if agents is None:
        device = torch.device('cpu')
        decay_steps = int(num_episodes * 14 * 0.8) # Decay epsilon over 80% of the training run
        agents = create_nfsp_agents(env, device=device, rl_learning_rate=rl_learning_rate, sl_learning_rate=sl_learning_rate, use_per=use_per, q_epsilon_decay_steps=decay_steps)
        
    env.set_agents(agents)

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    csv_path = os.path.join(save_dir, 'training_metrics.csv') if save_dir else None
    if csv_path is not None:
        mode = 'a' if (os.path.exists(csv_path) and start_episode > 0) else 'w'
        with open(csv_path, mode, newline='') as f:
            writer = csv.writer(f)
            if mode == 'w':
                header = ['episode']
                for pid in range(env.num_players):
                    header.extend([
                        f'player_{pid}_avg_payoff', f'player_{pid}_rl_loss', f'player_{pid}_sl_loss',
                        f'player_{pid}_won_pct', f'player_{pid}_under_pct', f'player_{pid}_over_pct',
                        f'player_{pid}_per_beta',
                    ])
                writer.writerow(header)

    rewards_log = []
    # Track bid outcomes: list of dicts per episode [{pid: 'won'/'under'/'over'}, ...]
    outcomes_log = []

    for episode in range(start_episode + 1, num_episodes + 1):
        # Sample episode policy for each agent
        for agent in agents:
            agent.sample_episode_policy()

        # Run one episode
        trajectories, payoffs = env.run(is_training=True)

        # Track bid outcomes before reorganizing trajectories
        ep_outcomes = {}
        for pid in range(env.num_players):
            p = env.game.players[pid]
            if p.bid is not None:
                if p.tricks_won == p.bid:
                    ep_outcomes[pid] = 'won'
                elif p.tricks_won < p.bid:
                    ep_outcomes[pid] = 'under'
                else:
                    ep_outcomes[pid] = 'over'
            else:
                ep_outcomes[pid] = 'under'  # shouldn't happen
        outcomes_log.append(ep_outcomes)

        # Use our custom reorganize_dense() to convert raw trajectories into
        # per-player lists of (state, action, reward, next_state, done) tuples.
        # This preserves dense trick-level rewards for the Q-Network.
        trajectories = reorganize_dense(trajectories, payoffs)

        for pid in range(env.num_players):
            for ts in trajectories[pid]:
                agents[pid].feed(ts)

        rewards_log.append(payoffs)

        # Evaluate
        if verbose and episode % evaluate_every == 0:
            avg_payoffs = np.mean(rewards_log[-evaluate_every:], axis=0)
            if start_episode > 0:
                print(f'\n\nEpisode {episode}/{num_episodes} (resumed from {start_episode}, {num_episodes - episode} remaining)')
            else:
                print(f'\n\nEpisode {episode}/{num_episodes}')
            
            # Compute outcome percentages over the evaluation window
            window = outcomes_log[-evaluate_every:]
            outcome_pcts = {}
            for pid in range(env.num_players):
                counts = {'won': 0, 'under': 0, 'over': 0}
                for ep_out in window:
                    counts[ep_out[pid]] += 1
                total = len(window)
                outcome_pcts[pid] = {
                    'won': counts['won'] / total * 100,
                    'under': counts['under'] / total * 100,
                    'over': counts['over'] / total * 100,
                }

            csv_row = [episode]
            for pid in range(env.num_players):
                pcts = outcome_pcts[pid]
                per_beta = agents[pid].latest_per_beta
                print(f'  Player {pid}: avg payoff = {avg_payoffs[pid]:.4f} | RL loss = {agents[pid].latest_rl_loss:.4f} | SL loss = {agents[pid].latest_sl_loss:.4f} | PER beta = {per_beta:.3f} | Won {pcts["won"]:.1f}% Under {pcts["under"]:.1f}% Over {pcts["over"]:.1f}%')
                csv_row.extend([avg_payoffs[pid], agents[pid].latest_rl_loss, agents[pid].latest_sl_loss,
                                round(pcts['won'], 2), round(pcts['under'], 2), round(pcts['over'], 2),
                                round(per_beta, 4)])
            
            if csv_path is not None:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_row)

        # Save checkpoint
        if save_dir and episode % checkpoint_every == 0:
            for pid, agent in enumerate(agents):
                agent.save_checkpoint(save_dir, filename=f'nfsp_agent_{pid}_ep{episode}.pt')
            if verbose:
                print(f'  Checkpoints saved at episode {episode}')

    return agents


def evaluate_agents(env, agents, num_episodes=100, logger=None):
    """Evaluate trained agents via random games."""
    # Force greedy evaluation! NFSP defaults to average_policy which 
    # throws cards probabilistically. We want strict Deterministic argmax.
    for agent in agents:
        if hasattr(agent, 'evaluate_with'):
            agent.evaluate_with = 'best_response'

    env.set_agents(agents)
    payoffs_sum = np.zeros(env.num_players)
    counts = [{'won': 0, 'under': 0, 'over': 0} for _ in range(env.num_players)]

    for _ in range(num_episodes):
        _, payoffs = env.run(is_training=False)
        if logger:
            logger.log_eval_game('pure_all', env.game.players, payoffs, env.num_players)
        payoffs_sum += payoffs
        
        for pid in range(env.num_players):
            p = env.game.players[pid]
            if p.bid is not None:
                if p.tricks_won == p.bid:
                    counts[pid]['won'] += 1
                elif p.tricks_won < p.bid:
                    counts[pid]['under'] += 1
                else:
                    counts[pid]['over'] += 1

    if num_episodes == 0:
        return payoffs_sum, counts

    avg_payoffs = payoffs_sum / num_episodes
    pcts = []
    for pid in range(env.num_players):
        pcts.append({
            'won': counts[pid]['won'] / num_episodes * 100,
            'under': counts[pid]['under'] / num_episodes * 100,
            'over': counts[pid]['over'] / num_episodes * 100,
        })
    
    return avg_payoffs, pcts


