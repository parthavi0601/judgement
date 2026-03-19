"""
Main entry point: MCTS warmup → NFSP training pipeline for Judgement card game.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlcard.envs.registration import register as rlcard_register, make as rlcard_make


def register_judgement_env():
    """Register the Judgement environment with RLCard."""
    try:
        rlcard_register(
            env_id='judgement',
            entry_point='judgement.env:JudgementEnv',
        )
    except ValueError:
        pass  # Already registered


def run_mcts_warmup(env, num_games, mcts_depth, mcts_simulations):
    """Phase 1: Run MCTS self-play games for warmup/data generation."""
    from agents.mcts_agent import JudgementMCTSAgent

    print(f'\n=== Phase 1: MCTS Warmup ({num_games} games) ===')
    print(f'  Depth (own moves): {mcts_depth}, Simulations: {mcts_simulations}')

    agents = [
        JudgementMCTSAgent(
            num_simulations=mcts_simulations,
            max_depth=mcts_depth,
        )
        for _ in range(env.num_players)
    ]
    env.set_agents(agents)

    total_payoffs = [0.0] * env.num_players
    for game_idx in range(1, num_games + 1):
        trajectories, payoffs = env.run(is_training=False)
        for pid in range(env.num_players):
            total_payoffs[pid] += payoffs[pid]

        if game_idx % max(1, num_games // 5) == 0:
            avg = [t / game_idx for t in total_payoffs]
            print(f'  Game {game_idx}/{num_games} — avg payoffs: {[f"{a:.3f}" for a in avg]}')

    print('  MCTS warmup complete.\n')
    return total_payoffs


def run_nfsp_training(env, num_episodes, save_dir):
    """Phase 2: Train NFSP agents."""
    from agents.nfsp_runner import train_nfsp

    print(f'=== Phase 2: NFSP Training ({num_episodes} episodes) ===')
    agents = train_nfsp(
        env,
        num_episodes=num_episodes,
        evaluate_every=max(1, num_episodes // 10),
        save_dir=save_dir,
        verbose=True,
    )
    print('\nNFSP training complete.\n')
    return agents


def main():
    parser = argparse.ArgumentParser(description='Judgement Card Game RL Training')
    parser.add_argument('--mcts-depth', type=int, default=2,
                        help='MCTS max depth (counts only agent own moves)')
    parser.add_argument('--mcts-simulations', type=int, default=50,
                        help='MCTS simulations per decision')
    parser.add_argument('--mcts-games', type=int, default=10,
                        help='Number of MCTS warmup games')
    parser.add_argument('--nfsp-episodes', type=int, default=100,
                        help='NFSP training episodes')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Register and create environment
    register_judgement_env()
    env = rlcard_make('judgement', config={
        'seed': args.seed,
        'allow_step_back': False,
        'game_num_players': 4,
    })

    print(f'Judgement Environment Created')
    print(f'  Players: {env.num_players}')
    print(f'  Actions: {env.num_actions}')
    print(f'  State shape: {env.state_shape}')

    # Phase 1: MCTS warmup
    run_mcts_warmup(env, args.mcts_games, args.mcts_depth, args.mcts_simulations)

    # Phase 2: NFSP training
    run_nfsp_training(env, args.nfsp_episodes, args.save_dir)

    print('=== Done ===')


if __name__ == '__main__':
    main()
