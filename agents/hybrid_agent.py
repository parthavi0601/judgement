"""
Hybrid MC-NFSP Agent for Judgement (Oh Hell) card game.

Combines MCTS tree search with NFSP policy evaluation:
- MCTS explores up to max_depth of the agent's OWN moves
- Opponent moves are stochastically sampled from the trained Average Policy (SL Network)
- Leaf nodes are evaluated via a blend of DQN Q-values and bid-alignment heuristic
- Bid actions use the same reward heuristic from round.py for shaped rewards
- Terminal states use compute_round_scores for exact +1/-1 scoring
"""

import copy
import math
import numpy as np
import torch
from typing import List, Optional, Dict


class _MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(self, parent=None, action=None, player_id=None):
        self.parent: Optional['_MCTSNode'] = parent
        self.action: Optional[int] = action
        self.player_id: Optional[int] = player_id
        self.children: Dict[int, '_MCTSNode'] = {}
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.is_terminal: bool = False

    @property
    def q_value(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        return self.q_value + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c: float = 1.414) -> '_MCTSNode':
        return max(self.children.values(), key=lambda n: n.ucb1(c))

    def is_fully_expanded(self, legal_actions: List[int]) -> bool:
        return all(a in self.children for a in legal_actions)


class HybridMCNFSPAgent:
    def __init__(self, env, agent_player_id: int, all_nfsp_agents=None,
                 num_simulations: int = 200, max_depth: int = 2,
                 exploration_constant: float = 1.414):
        self.env = env
        self.agent_player_id = agent_player_id
        self.all_nfsp_agents = all_nfsp_agents
        self.nfsp_agent = all_nfsp_agents[agent_player_id] if all_nfsp_agents else None
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.use_raw = True

    def step(self, state) -> int:
        return self._run_mcts(state)

    def eval_step(self, state):
        action = self._run_mcts(state)
        return action, {'agent': 'hybrid_mc_nfsp'}

    def _run_mcts(self, state) -> int:
        legal_actions = list(state['legal_actions'].keys())
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else 0

        root = _MCTSNode(player_id=self.agent_player_id)

        # Single deep copy + checkpoint: N lightweight restores instead of N deep copies
        game_clone = copy.deepcopy(self.env.game)
        checkpoint = game_clone.save_checkpoint()

        # Dynamic simulation scaling: if tactical choices are narrow (late seats), 
        # Monte Carlo rollout variance dominates. Increase sims to stabilize.
        sims_to_run = self.num_simulations
        if len(legal_actions) <= 3:
            sims_to_run *= 4
        elif len(legal_actions) <= 5:
            sims_to_run *= 2

        for _ in range(sims_to_run):
            self._determinize(game_clone)
            self._simulate(root, game_clone, legal_actions)
            game_clone.restore_checkpoint(checkpoint)

        if not root.children:
            return int(np.random.choice(legal_actions))

        return max(root.children.keys(), key=lambda a: root.children[a].visits)

    def _sample_opponent_action(self, game, acting_player_id, legal_actions) -> int:
        """Sample an action using the specific opponent's NFSP Average Policy."""
        if not self.all_nfsp_agents or acting_player_id >= len(self.all_nfsp_agents):
            return int(np.random.choice(legal_actions))

        opp_agent = self.all_nfsp_agents[acting_player_id]
        if opp_agent is None or not hasattr(opp_agent, 'policy_network'):
            return int(np.random.choice(legal_actions))

        raw_state = game.get_state(acting_player_id)
        extracted = self.env._extract_state(raw_state)
        obs = extracted['obs']

        obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
        device = next(opp_agent.policy_network.parameters()).device
        obs_tensor = obs_tensor.to(device)

        with torch.no_grad():
            log_probs = opp_agent.policy_network(obs_tensor).cpu().numpy()[0]

        probs = np.exp(log_probs)
        legal_probs = probs[legal_actions]
        sum_probs = legal_probs.sum()

        if sum_probs < 1e-8:
            return int(np.random.choice(legal_actions))

        legal_probs /= sum_probs
        return int(np.random.choice(legal_actions, p=legal_probs))

    def _determinize(self, game):
        """
        Information Set MCTS: Strictly build 'true unknown cards' from the full
        deck minus known cards (our hand, played cards, revealed trump).
        Shuffle and deal sizes back to opponents, respecting voids.
        """
        if not game.current_round:
            return

        from judgement.card import JudgementCard

        # 1. Collect all KNOWN cards
        known_card_ids = set()
        
        for c in game.players[self.agent_player_id].hand:
            known_card_ids.add(c.card_id)
            
        if game.current_round.get_trump_card():
            known_card_ids.add(game.current_round.get_trump_card().card_id)
            
        for trick in game.current_round.trick_history:
            for pid, c in trick:
                known_card_ids.add(c.card_id)
                
        for pid, c in game.current_round.current_trick:
            known_card_ids.add(c.card_id)

        # 2. Build True Unknown Cards from the 52-card deck
        true_unknown_cards = []
        for c in JudgementCard.get_deck():
            if c.card_id not in known_card_ids:
                true_unknown_cards.append(c)

        np.random.shuffle(true_unknown_cards)

        # 3. Clear opponent hands and record sizes exactly
        opp_hand_sizes = {}
        for i, p in enumerate(game.players):
            if i != self.agent_player_id:
                opp_hand_sizes[i] = len(p.hand)
                p.hand = []

        # 4. Deduce voids based on trick history
        voids = {i: set() for i in range(game.num_players)}
        for trick in game.current_round.trick_history:
            if not trick: continue
            lead_suit = trick[0][1].suit
            for pid, c in trick:
                if c.suit != lead_suit:
                    voids[pid].add(lead_suit)

        if game.current_round.current_trick:
            lead_suit = game.current_round.current_trick[0][1].suit
            for pid, c in game.current_round.current_trick:
                if c.suit != lead_suit:
                    voids[pid].add(lead_suit)

        # 5. Greedy deal using MRV (Most Constrained First) to prevent void violations
        sorted_opps = sorted(opp_hand_sizes.keys(), key=lambda p: len(voids[p]), reverse=True)
        
        for pid in sorted_opps:
            needed = opp_hand_sizes[pid]
            while needed > 0 and true_unknown_cards:
                valid_idx = -1
                for i, c in enumerate(true_unknown_cards):
                    if c.suit not in voids[pid]:
                        valid_idx = i
                        break
                
                if valid_idx != -1:
                    card = true_unknown_cards.pop(valid_idx)
                    game.players[pid].hand.append(card)
                    needed -= 1
                else:
                    # Fallback constraint violation if strictly necessary
                    card = true_unknown_cards.pop(0)
                    game.players[pid].hand.append(card)
                    needed -= 1

    def _simulate(self, root: _MCTSNode, game, legal_actions: List[int]):
        node = root
        depth = 0
        path = [node]
        bid_reward_bonus = 0.0  # Track bid heuristic reward accumulated in this sim

        # ── Selection ──
        current_legal = legal_actions
        while (node.children
               and node.is_fully_expanded(current_legal)
               and not node.is_terminal):

            acting_player = game.get_player_id()
            if acting_player == self.agent_player_id:
                node = node.best_child(self.exploration_constant)
                action = node.action
                depth += 1
            else:
                action = self._sample_opponent_action(game, acting_player, current_legal)
                if action not in node.children:
                    break
                node = node.children[action]

            # Track bid heuristic if our agent is bidding
            if (acting_player == self.agent_player_id
                    and game.current_round and game.current_round.is_bidding):
                bid_reward_bonus += self._bid_reward_heuristic(game, action)

            game.step(action)
            path.append(node)

            if game.is_over():
                node.is_terminal = True
                break
            current_legal = game._get_legal_actions()

        # ── Expansion ──
        if not node.is_terminal and not game.is_over() and depth < self.max_depth:
            current_legal = game._get_legal_actions()
            acting_player = game.get_player_id()

            if acting_player == self.agent_player_id:
                unexplored = [a for a in current_legal if a not in node.children]
                if unexplored:
                    action = int(np.random.choice(unexplored))
                else:
                    action = None
            else:
                action = self._sample_opponent_action(game, acting_player, current_legal)

            if action is not None and action not in node.children:
                child = _MCTSNode(parent=node, action=action, player_id=acting_player)
                node.children[action] = child
                node = child
                path.append(node)

                # Track bid heuristic for expansion action
                if (acting_player == self.agent_player_id
                        and game.current_round and game.current_round.is_bidding):
                    bid_reward_bonus += self._bid_reward_heuristic(game, action)

                if not game.is_over():
                    game.step(action)
                    if acting_player == self.agent_player_id:
                        depth += 1

        # ── Fast-forward opponents to reach our next turn ──
        while not game.is_over() and game.get_player_id() != self.agent_player_id:
            acting_player = game.get_player_id()
            current_legal = game._get_legal_actions()
            if not current_legal:
                break
            action = self._sample_opponent_action(game, acting_player, current_legal)
            game.step(action)

        # ── Leaf Evaluation ──
        if game.is_over():
            reward = self._score_terminal(game)
        elif self.nfsp_agent is not None:
            reward = self._nfsp_evaluate(game)
        else:
            reward = self._heuristic_evaluate(game)

        # Blend in bid heuristic bonus (weighted down so it guides but doesn't dominate)
        reward += 0.15 * bid_reward_bonus

        reward = float(np.clip(reward, -1.0, 1.0))

        # ── Backpropagation ──
        for n in path:
            n.visits += 1
            n.total_reward += reward

    def _bid_reward_heuristic(self, game, bid_action: int) -> float:
        """
        Evaluate bid quality using the same hand-strength heuristic from round.py.
        Returns a shaped reward in roughly [-1.0, +0.5].
        """
        if bid_action >= 14:
            return 0.0  # Not a bid action

        bid_value = bid_action
        player = game.players[self.agent_player_id]
        trump_suit = game.current_round.trump_suit if game.current_round else None

        expected_tricks = self._estimate_tricks(player.hand, trump_suit)

        diff = abs(bid_value - expected_tricks)
        return max(-1.0, 0.5 - (0.3 * diff))

    def _estimate_tricks(self, hand, trump_suit) -> float:
        """Estimate expected tricks based on hand strength (matches round.py heuristic)."""
        expected = 0.0
        for c in hand:
            if c.suit == trump_suit:
                if c.rank_index >= 12:
                    expected += 1.0      # Ace of trump
                elif c.rank_index >= 11:
                    expected += 0.8      # King of trump
                elif c.rank_index >= 9:
                    expected += 0.5      # 10, J, Q of trump
                else:
                    expected += 0.2      # low trumps
            else:
                if c.rank_index >= 12:
                    expected += 0.5      # Ace off-suit
                elif c.rank_index >= 10:
                    expected += 0.2      # Q, K off-suit
        return expected

    def _score_terminal(self, game) -> float:
        """Use the game's official scoring: +1.0 for exact bid, -1.0 for miss."""
        from judgement.judger import JudgementJudger
        scores = JudgementJudger.compute_round_scores(game.players)
        return scores[self.agent_player_id]

    def _nfsp_evaluate(self, game) -> float:
        """
        Evaluate leaf using NFSP DQN Q-values blended with bid alignment.
        The Q-value captures learned strategic value; bid alignment adds
        explicit progress tracking that raw Q-values may underweight.
        """
        pid = game.get_player_id()
        if pid != self.agent_player_id:
            # If it's not our turn (shouldn't happen after fast-forward), use heuristic
            return self._heuristic_evaluate(game)

        raw_state = game.get_state(pid)
        extracted = self.env._extract_state(raw_state)
        obs = extracted['obs']
        legal_actions = list(extracted['legal_actions'].keys())

        if not legal_actions:
            return self._heuristic_evaluate(game)

        obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()

        q_value = None
        if hasattr(self.nfsp_agent, '_rl_agent') and hasattr(self.nfsp_agent._rl_agent, 'q_estimator'):
            q_estimator = self.nfsp_agent._rl_agent.q_estimator
            device = next(q_estimator.qnet.parameters()).device
            obs_tensor = obs_tensor.to(device)
            with torch.no_grad():
                q_values = q_estimator.qnet(obs_tensor).cpu().numpy()[0]

            legal_q_values = q_values[legal_actions]
            q_value = float(np.max(legal_q_values))
            q_value = np.clip(q_value, -1.0, 1.0)

        if q_value is not None:
            # Blend Q-value (70%) with bid-alignment heuristic (30%)
            heuristic = self._heuristic_evaluate(game)
            return 0.7 * q_value + 0.3 * heuristic
        else:
            return self._heuristic_evaluate(game)

    def _heuristic_evaluate(self, game) -> float:
        """
        Non-terminal evaluation using bid alignment.
        Primary signal: how well is the agent tracking toward its bid?
        """
        p = game.players[self.agent_player_id]

        if p.bid is None:
            return 0.0

        tricks_remaining = 0
        if game.current_round:
            tricks_remaining = game.current_round.num_cards - game.current_round.tricks_played

        needed = p.bid - p.tricks_won

        if needed == 0:
            # Already met bid — good position, reward proportional to
            # how few tricks remain (fewer = safer)
            total = game.current_round.num_cards if game.current_round else 1
            safety = 1.0 - (tricks_remaining / max(total, 1))
            return 0.3 + 0.4 * safety
        elif needed > 0:
            if tricks_remaining >= needed:
                # Still achievable — mild optimism
                achievability = needed / max(tricks_remaining, 1)
                return 0.1 * (1.0 - achievability)
            else:
                # Impossible to make bid — penalize
                return -0.5
        else:
            # Over bid — penalize based on how much over
            return float(np.clip(-0.3 * abs(needed), -1.0, 0.0))
