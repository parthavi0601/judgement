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
                 num_simulations: int = 200, max_depth: int = 100,
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
        """
        Split evaluation into Bidding (Supervised) and Play (MCTS).
        Bidding is performed by the expert SL model to avoid DQN bias.
        Play is performed by MCTS + Leaf Evaluation.
        """
        current_legal = list(state['legal_actions'].keys())
        if all(a <= 13 for a in current_legal):
            # Bidding Phase
            return self.nfsp_agent.eval_step(state)
            
        # Play Phase: Search to Depth 4 with Leaf Evaluation
        action = self._run_mcts(state)
        return action, {'agent': 'hybrid_mc_nfsp'}

    def _run_mcts(self, state) -> int:
        legal_actions = list(state['legal_actions'].keys())
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else 0

        root = _MCTSNode(player_id=self.agent_player_id)

        # Dynamic budget scaling: invest more sims when branching is narrow
        # to cut through determinization noise with fewer legal options
        n_legal = len(legal_actions)
        if n_legal <= 3:
            scaled_sims = self.num_simulations * 4
        elif n_legal <= 5:
            scaled_sims = self.num_simulations * 2
        else:
            scaled_sims = self.num_simulations

        # Single deep copy + checkpoint: N lightweight restores instead of N deep copies
        game_clone = copy.deepcopy(self.env.game)
        checkpoint = game_clone.save_checkpoint()

        for _ in range(scaled_sims):
            self._determinize(game_clone)
            self._simulate(root, game_clone, legal_actions)
            game_clone.restore_checkpoint(checkpoint)

        if not root.children:
            return int(np.random.choice(legal_actions))

        return max(root.children.keys(), key=lambda a: root.children[a].visits)

    def _sample_opponent_action(self, game, acting_player_id, legal_actions) -> int:
        """Sample an action using the specific opponent's NFSP Average Policy or eval_step."""
        if not self.all_nfsp_agents or acting_player_id >= len(self.all_nfsp_agents):
            return int(np.random.choice(legal_actions))

        opp_agent = self.all_nfsp_agents[acting_player_id]
        if opp_agent is None:
            return int(np.random.choice(legal_actions))

        if hasattr(opp_agent, 'policy_network'):
            # NFSP Agent: Sample from SL Network (Average Policy)
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
        elif hasattr(opp_agent, 'eval_step'):
            # Generic/Rule-Based Agent logic
            raw_state = game.get_state(acting_player_id)
            action, _ = opp_agent.eval_step(raw_state)
            if action in legal_actions:
                return action
            return int(np.random.choice(legal_actions))
        else:
            return int(np.random.choice(legal_actions))

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
                # IMPORTANT: In ISMCTS, we must only pick actions that are LEGAL 
                # in the current determinization. Filter children by current legality.
                valid_children_actions = [a for a in node.children.keys() if a in current_legal]
                if not valid_children_actions:
                    # If no explored child is legal in this determinization, 
                    # we must break selection and expand/rollout.
                    break
                
                # For opponents, select a child that is legal in this state
                action = int(np.random.choice(valid_children_actions))
                node = node.children[action]

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
                # Expand opponents randomly to explore all plausible counter-moves
                action = int(np.random.choice(current_legal))

            if action is not None and action not in node.children:
                child = _MCTSNode(parent=node, action=action, player_id=acting_player)
                node.children[action] = child
                node = child
                path.append(node)

                if not game.is_over():
                    game.step(action)
                    if acting_player == self.agent_player_id:
                        depth += 1

        # ── Leaf Evaluation ──
        # Instead of a full rollout, we use the NFSP agent's Q-values to estimate 
        # the future expected return of the leaf state. This fulfills the user's
        # goal of "predicting how good" a state is without simulating to the end.
        if game.is_over():
            reward = self._score_terminal(game)
        else:
            reward = self._nfsp_evaluate(game)
        
        reward = float(np.clip(reward, -1.0, 1.0))
        reward = float(np.clip(reward, -1.0, 1.0))

        # ── Backpropagation ──
        for n in path:
            n.visits += 1
            n.total_reward += reward

    def _score_terminal(self, game) -> float:
        """Use the game's official scoring: +1.0 for exact bid, -1.0 for miss."""
        from judgement.judger import JudgementJudger
        scores = JudgementJudger.compute_round_scores(game.players)
        return scores[self.agent_player_id]

    def _nfsp_evaluate(self, game) -> float:
        """
        Evaluate leaf using NFSP DQN Q-values blended with hand-power heuristics.
        At early depths (bidding), we rely on a smooth 'Bid Accuracy' gradient
        to avoid the DQN's pessimistic Zero-Bid bias.
        """
        p = game.players[self.agent_player_id]
        hand_power = self._calculate_hand_strength(game)
        
        # Determine if we are still in the strategic bidding phase
        in_bidding = False
        if any(pl.bid is None for pl in game.players):
            in_bidding = True
            
        if in_bidding:
            if p.bid is None:
                # Evaluating a state before we have chosen our own bid
                return float(np.clip(hand_power / 13.0, 0.0, 1.0))
            else:
                # Evaluating a potential bid we have just made.
                # Reward based on how well the bid matches our hand strength.
                diff = abs(p.bid - hand_power)
                accuracy = 1.0 - (diff / 13.0)
                return float(np.clip(accuracy, 0.0, 1.0))

        # --- Play Phase ---
        # Get DQN strategic value
        q_value = 0.0
        raw_state = game.get_state(self.agent_player_id)
        extracted = self.env._extract_state(raw_state)
        obs = extracted['obs']
        legal_actions = list(extracted['legal_actions'].keys())

        if legal_actions and hasattr(self.nfsp_agent, '_rl_agent'):
            q_estimator = self.nfsp_agent._rl_agent.q_estimator
            obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
            device = next(q_estimator.qnet.parameters()).device
            obs_tensor = obs_tensor.to(device)
            with torch.no_grad():
                # Q-values are in range [-1, 1] for terminal rewards
                q_values = q_estimator.qnet(obs_tensor).cpu().numpy()[0]
            
            # Use max legal Q-value
            q_value = float(np.max(q_values[legal_actions]))

        # Heuristic bid-alignment (tracking progress during play)
        alignment = self._heuristic_evaluate(game)

        # Blend strategic Q-value with local trick alignment
        # This prevents the bot from making 'correct' plays that don't match his bid
        return 0.7 * q_value + 0.3 * alignment

    def _calculate_hand_strength(self, game) -> float:
        """Estimate hand power conservatively for high cards/trumps."""
        p = game.players[self.agent_player_id]
        if not p.hand:
            return 0.0
            
        trump_suit = None
        if game.current_round:
            trump_suit = game.current_round.trump_suit
            
        strength = 0.0
        for card in p.hand:
            rank_idx = card.rank_index # 2=0, A=12
            
            # Trump bonus: even low trumps are strong
            if card.suit == trump_suit:
                strength += 0.7 + 0.3 * (rank_idx / 12.0)
            else:
                # Suit winners (Ages, Kings)
                if rank_idx >= 11: # A, K
                    strength += 0.8 * (rank_idx / 12.0)
                elif rank_idx >= 9: # Q, J
                    strength += 0.3 * (rank_idx / 12.0)
                else:
                    # Garbage cards might still win if others are short
                    strength += 0.05 * (rank_idx / 12.0)
                    
        return strength

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


# class ISMCTSAgent(HybridMCNFSPAgent):
#     """
#     Classic ISMCTS Agent using random rollouts until round-end.
#     - No NFSP guidance (sampled actions are random)
#     - No depth limit (searches to terminal state)
#     - Random opponent modeling
#     - Terminal reward scoring only
#     """
#     def __init__(self, env, agent_player_id: int, num_simulations: int = 200, exploration_constant: float = 1.414):
#         super().__init__(env, agent_player_id, 
#                          all_nfsp_agents=None, 
#                          num_simulations=num_simulations,
#                          max_depth=100,           # Full rollout
#                          exploration_constant=exploration_constant)

#     def eval_step(self, state):
#         action = self._run_mcts(state)
#         return action, {'agent': 'pure_ismcts'}

#     def _sample_opponent_action(self, game, acting_player_id, legal_actions) -> int:
#         """Strictly random opponent modeling."""
#         return int(np.random.choice(legal_actions))

#     def _simulate(self, root: _MCTSNode, game, legal_actions: List[int]):
#         """Classic ISMCTS rollout until game over."""
#         node = root
#         path = [node]

#         # ── Selection ──
#         current_legal = legal_actions
#         while (node.children
#                and node.is_fully_expanded(current_legal)
#                and not node.is_terminal):

#             acting_player = game.get_player_id()
#             if acting_player == self.agent_player_id:
#                 node = node.best_child(self.exploration_constant)
#                 action = node.action
#             else:
#                 # Filter children by current legality for this determinization
#                 valid_children_actions = [a for a in node.children.keys() if a in current_legal]
#                 if not valid_children_actions:
#                     break
                
#                 # Pick a legal action from the explored children
#                 action = int(np.random.choice(valid_children_actions))
#                 node = node.children[action]

#             game.step(action)
#             path.append(node)

#             if game.is_over():
#                 node.is_terminal = True
#                 break
#             current_legal = game._get_legal_actions()

#         # ── Expansion ──
#         if not node.is_terminal and not game.is_over() and (self.max_depth is None or len(path) - 1 < self.max_depth):
#             current_legal = game._get_legal_actions()
#             acting_player = game.get_player_id()

#             if acting_player == self.agent_player_id:
#                 unexplored = [a for a in current_legal if a not in node.children]
#                 if unexplored:
#                     action = int(np.random.choice(unexplored))
#                 else:
#                     action = None
#             else:
#                 action = self._sample_opponent_action(game, acting_player, current_legal)

#             if action is not None and action not in node.children:
#                 child = _MCTSNode(parent=node, action=action, player_id=acting_player)
#                 node.children[action] = child
#                 node = child
#                 path.append(node)

#                 if not game.is_over():
#                     game.step(action)

#         # ── Full Rollout (Random Play) ──
#         while not game.is_over():
#             current_legal = game._get_legal_actions()
#             if not current_legal:
#                 break
#             action = int(np.random.choice(current_legal))
#             game.step(action)

#         # ── Terminal Reward ──
#         reward = self._score_terminal(game)
#         reward = float(np.clip(reward, -1.0, 1.0))

#         # ── Backpropagation ──
#         for n in path:
#             n.visits += 1
#             n.total_reward += reward
