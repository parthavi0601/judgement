"""
Main entry point: NFSP training → Hybrid MC-NFSP evaluation pipeline
for Judgement card game.
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


def run_nfsp_training(env, num_episodes, save_dir, evaluate_every=None, checkpoint_every=None, agents=None, start_episode=0, rl_lr=0.01, sl_lr=0.005):
    """Phase 1: Train NFSP agents."""
    from agents.nfsp_runner import train_nfsp

    eval_freq = evaluate_every if evaluate_every else max(1, num_episodes // 10)
    ckpt_freq = checkpoint_every if checkpoint_every else eval_freq * 4

    print(f'\n=== Phase 1: NFSP Training ({num_episodes} episodes) ===')
    agents = train_nfsp(
        env,
        num_episodes=num_episodes,
        evaluate_every=eval_freq,
        checkpoint_every=ckpt_freq,
        save_dir=save_dir,
        verbose=True,
        agents=agents,
        start_episode=start_episode,
        rl_learning_rate=rl_lr,
        sl_learning_rate=sl_lr,
    )
    print('\nNFSP training complete.\n')
    return agents


def run_hybrid_evaluation(env, nfsp_agents, num_games, mcts_depth, mcts_simulations):
    """Phase 2: Evaluate using Hybrid MC-NFSP agents."""
    from agents.hybrid_agent import HybridMCNFSPAgent

    print(f'=== Phase 2: Hybrid MC-NFSP Evaluation ({num_games} games) ===')
    print(f'  MCTS depth (own moves): {mcts_depth}, Simulations: {mcts_simulations}')
    print(f'  Leaf evaluation: NFSP average-policy network\n')

    # Create hybrid agents: each uses MCTS + that player's trained NFSP policy
    hybrid_agents = []
    for pid in range(env.num_players):
        agent = HybridMCNFSPAgent(
            env=env,
            agent_player_id=pid,
            nfsp_agent=nfsp_agents[pid],
            num_simulations=mcts_simulations,
            max_depth=mcts_depth,
        )
        hybrid_agents.append(agent)

    env.set_agents(hybrid_agents)

    total_payoffs = [0.0] * env.num_players
    for game_idx in range(1, num_games + 1):
        trajectories, payoffs = env.run(is_training=False)
        for pid in range(env.num_players):
            total_payoffs[pid] += payoffs[pid]

        if game_idx % max(1, num_games // 5) == 0:
            avg = [t / game_idx for t in total_payoffs]
            print(f'  Game {game_idx}/{num_games} — avg payoffs: {[f"{a:.4f}" for a in avg]}')

    final_avg = [t / num_games for t in total_payoffs]
    print(f'\n  Final avg payoffs: {[f"{a:.4f}" for a in final_avg]}')
    print('  Hybrid evaluation complete.\n')
    return final_avg


def run_pure_nfsp_evaluation(env, nfsp_agents, num_games):
    """Phase 3: Evaluate pure NFSP for comparison."""
    from agents.nfsp_runner import evaluate_agents

    print(f'=== Phase 3: Pure NFSP Evaluation ({num_games} games, for comparison) ===')
    env.set_agents(nfsp_agents)
    avg = evaluate_agents(env, nfsp_agents, num_episodes=num_games)
    print(f'  Pure NFSP avg payoffs: {[f"{a:.4f}" for a in avg]}')
    print()
    return avg


def load_nfsp_agents(env, checkpoint_dir, episode_tag, new_rl_lr=None, new_sl_lr=None):
    """Load pre-trained NFSP agents from checkpoints."""
    import torch
    from rlcard.agents.nfsp_agent import NFSPAgent
    from agents.nfsp_runner import patch_agent_losses

    print(f'\n=== Loading NFSP agents from {checkpoint_dir} (episode {episode_tag}) ===')
    if new_rl_lr or new_sl_lr:
        print(f'  Updating learning rates -> RL: {new_rl_lr}, SL: {new_sl_lr}')
    agents = []
    for pid in range(env.num_players):
        filename = f'nfsp_agent_{pid}_ep{episode_tag}.pt'
        filepath = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Checkpoint not found: {filepath}')
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        agent = NFSPAgent.from_checkpoint(checkpoint)
        
        if new_sl_lr is not None:
            agent._sl_learning_rate = new_sl_lr
            for param_group in agent.policy_network_optimizer.param_groups:
                param_group['lr'] = new_sl_lr

        if new_rl_lr is not None:
            agent._rl_agent.q_estimator.learning_rate = new_rl_lr
            for param_group in agent._rl_agent.q_estimator.optimizer.param_groups:
                param_group['lr'] = new_rl_lr

        patch_agent_losses(agent)
        agents.append(agent)
        print(f'  Loaded Player {pid} from {filename}')
    print('  All agents loaded.\n')
    return agents


def main():
    parser = argparse.ArgumentParser(description='Judgement Card Game — Hybrid MC-NFSP Pipeline')
    parser.add_argument('--nfsp-episodes', type=int, default=500,
                        help='NFSP training episodes (Phase 1, skipped if --load-checkpoint without --resume-training)')
    parser.add_argument('--resume-training', action='store_true',
                        help='Resume training from the loaded checkpoint instead of skipping training')
    parser.add_argument('--rl-learning-rate', type=float, default=None,
                        help='Override RL learning rate (Q-network) when resuming training')
    parser.add_argument('--sl-learning-rate', type=float, default=None,
                        help='Override SL learning rate (Average Policy) when resuming training')
    parser.add_argument('--load-checkpoint', type=int, default=None, metavar='EPISODE',
                        help='Load pre-trained NFSP from checkpoints at this episode number '
                             '(e.g. --load-checkpoint 4000). Skips training unless --resume-training is specified.')
    parser.add_argument('--hybrid-games', type=int, default=10,
                        help='Hybrid MC-NFSP evaluation games (Phase 2)')
    parser.add_argument('--mcts-depth', type=int, default=2,
                        help='MCTS max depth (counts only agent own moves)')
    parser.add_argument('--mcts-simulations', type=int, default=50,
                        help='MCTS simulations per decision')
    parser.add_argument('--eval-games', type=int, default=20,
                        help='Pure NFSP evaluation games (Phase 3, for comparison)')
    parser.add_argument('--evaluate-every', type=int, default=None,
                        help='Evaluate every N episodes')
    parser.add_argument('--checkpoint-every', type=int, default=None,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save/load model checkpoints')
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

    # Phase 1: Train or Load NFSP agents
    if args.load_checkpoint is not None:
        nfsp_agents = load_nfsp_agents(
            env, 
            args.save_dir, 
            args.load_checkpoint,
            new_rl_lr=args.rl_learning_rate,
            new_sl_lr=args.sl_learning_rate
        )
        if args.resume_training:
            nfsp_agents = run_nfsp_training(env, args.nfsp_episodes, args.save_dir, args.evaluate_every, args.checkpoint_every, agents=nfsp_agents, start_episode=args.load_checkpoint)
    else:
        rl_lr = args.rl_learning_rate if args.rl_learning_rate is not None else 0.01
        sl_lr = args.sl_learning_rate if args.sl_learning_rate is not None else 0.005
        nfsp_agents = run_nfsp_training(env, args.nfsp_episodes, args.save_dir, args.evaluate_every, args.checkpoint_every, rl_lr=rl_lr, sl_lr=sl_lr)

    # Phase 2: Hybrid MC-NFSP evaluation
    hybrid_avg = run_hybrid_evaluation(
        env, nfsp_agents, args.hybrid_games,
        args.mcts_depth, args.mcts_simulations
    )

    # Phase 3: Pure NFSP comparison
    nfsp_avg = run_pure_nfsp_evaluation(env, nfsp_agents, args.eval_games)

    # Summary
    print('═' * 60)
    print('  RESULTS COMPARISON')
    print('═' * 60)
    print(f'  {"Player":<10} {"Hybrid MC-NFSP":>15} {"Pure NFSP":>15}')
    print(f'  {"─"*10} {"─"*15} {"─"*15}')
    for pid in range(env.num_players):
        print(f'  Player {pid:<3} {hybrid_avg[pid]:>15.4f} {nfsp_avg[pid]:>15.4f}')
    print('═' * 60)
    print('\n=== Done ===')


if __name__ == '__main__':
    main()

