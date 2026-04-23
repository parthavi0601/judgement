"""
Smoke test for PER (Prioritized Experience Replay) implementation.
Tests SumTree, PERMemory, and verifies DQNAgent + NFSPAgent accept use_per.
Run with: uv run python test_per_smoke.py
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== Testing SumTree ===")
from rlcard.agents.dqn_agent import SumTree, PERMemory, DQNAgent, Transition

# --- SumTree unit tests ---
tree = SumTree(capacity=8)
assert tree.total_priority == 0.0, "Empty tree should have zero priority"

priorities = [0.5, 0.3, 0.7, 0.1, 0.9, 0.2, 0.6, 0.4]
for i, p in enumerate(priorities):
    tree.add(p, data=f"exp_{i}")

assert abs(tree.total_priority - sum(priorities)) < 1e-5, \
    f"Root should equal sum of all priorities: {tree.total_priority} vs {sum(priorities)}"
assert tree.num_stored == 8
print(f"  SumTree total_priority = {tree.total_priority:.4f} (expected {sum(priorities):.4f}) ✓")

# Verify sample returns valid data
tree_idx, priority, data = tree.get(tree.total_priority * 0.5)
assert data is not None, "Sample should return non-None data"
print(f"  SumTree.get() returned priority={priority:.4f}, data={data!r} ✓")

# Verify update propagates correctly
old_total = tree.total_priority
leaf_idx = 0
tree.update(tree.capacity - 1 + leaf_idx, 2.0)  # Update first leaf to 2.0
new_total = tree.total_priority
expected_new_total = old_total - priorities[0] + 2.0
assert abs(new_total - expected_new_total) < 1e-4, \
    f"After update: {new_total:.4f} vs expected {expected_new_total:.4f}"
print(f"  SumTree.update() propagated correctly: {old_total:.4f} → {new_total:.4f} ✓")


print("\n=== Testing PERMemory ===")
per_mem = PERMemory(memory_size=100, batch_size=32)

# Insert fake transitions
dummy_state = np.zeros(10)
dummy_next = np.ones(10)
for i in range(100):
    per_mem.save(dummy_state, i % 10, float(i) * 0.01, dummy_next, [0, 1, 2], False)

assert per_mem._sum_tree.num_stored == 100
print(f"  Inserted 100 transitions ✓")

# Check initial beta
beta0 = per_mem.current_beta
assert abs(beta0 - 0.4) < 1e-6, f"Initial beta should be 0.4, got {beta0}"
print(f"  Initial beta = {beta0:.4f} ✓")

# Sample a batch
result = per_mem.sample()
assert len(result) == 8, f"Expected 8-tuple from PERMemory.sample(), got {len(result)}"
state_b, action_b, reward_b, next_b, done_b, legal_b, tree_idxs, is_weights = result

assert state_b.shape == (32, 10), f"state_batch shape: {state_b.shape}"
assert len(tree_idxs) == 32
assert is_weights.shape == (32,)
assert is_weights.max() <= 1.0 + 1e-6  # normalized
assert is_weights.min() >= 0.0
print(f"  PERMemory.sample() returned batch of 32, IS-weights range: [{is_weights.min():.4f}, {is_weights.max():.4f}] ✓")

# Update priorities
fake_td_errors = np.abs(np.random.randn(32)) + 0.01
per_mem.update_priorities(tree_idxs, fake_td_errors)
assert per_mem._train_step == 1
print(f"  update_priorities() completed, _train_step=1, _max_priority={per_mem._max_priority:.4f} ✓")

# Beta advances after train_step
per_mem._train_step = per_mem.beta_anneal_steps
beta_end = per_mem.current_beta
assert abs(beta_end - 1.0) < 1e-6, f"Final beta should be 1.0, got {beta_end}"
print(f"  Beta at end of annealing = {beta_end:.4f} ✓")


print("\n=== Testing DQNAgent with use_per=True ===")
agent = DQNAgent(
    replay_memory_size=200,
    replay_memory_init_size=64,
    batch_size=32,
    num_actions=5,
    state_shape=[10],
    mlp_layers=[64, 32],
    use_per=True,
)
assert isinstance(agent.memory, PERMemory), f"Expected PERMemory, got {type(agent.memory)}"
assert agent.use_per is True
print(f"  DQNAgent(use_per=True) uses PERMemory ✓")

# Fill buffer past init size and train
for _ in range(130):
    s = np.random.randn(10).astype(np.float32)
    ns = np.random.randn(10).astype(np.float32)
    agent.feed_memory(s, 0, 1.0, ns, [0, 1, 2], False)

assert agent.memory._sum_tree.num_stored == 130
agent.train()  # Should NOT crash
print(f"  DQNAgent.train() with PER ran successfully ✓")


print("\n=== Testing DQNAgent with use_per=False ===")
from rlcard.agents.dqn_agent import Memory
agent_uniform = DQNAgent(
    replay_memory_size=200,
    replay_memory_init_size=64,
    batch_size=32,
    num_actions=5,
    state_shape=[10],
    mlp_layers=[64, 32],
    use_per=False,
)
assert isinstance(agent_uniform.memory, Memory), f"Expected Memory, got {type(agent_uniform.memory)}"
print(f"  DQNAgent(use_per=False) uses uniform Memory ✓")


print("\n=== Testing checkpoint save/restore with PER ===")
ckpt = agent.checkpoint_attributes()
assert ckpt['use_per'] is True
assert 'per_alpha' in ckpt
assert 'sum_tree' in ckpt['memory']  # PERMemory checkpoint structure
print(f"  checkpoint_attributes() includes PER fields ✓")

agent2 = DQNAgent.from_checkpoint(ckpt)
assert isinstance(agent2.memory, PERMemory)
assert abs(agent2.memory._max_priority - agent.memory._max_priority) < 1e-6
print(f"  from_checkpoint() restored PERMemory correctly ✓")


print("\n=== Testing NFSPAgent use_per forwarding ===")
from rlcard.agents.nfsp_agent import NFSPAgent
nfsp = NFSPAgent(
    num_actions=5,
    state_shape=[10],
    hidden_layers_sizes=[64, 32],
    use_per=True,
    q_replay_memory_size=200,
    q_replay_memory_init_size=64,
    q_batch_size=32,
    q_mlp_layers=[64, 32],
)
assert isinstance(nfsp._rl_agent.memory, PERMemory), \
    f"NFSPAgent._rl_agent should use PERMemory, got {type(nfsp._rl_agent.memory)}"
print(f"  NFSPAgent(use_per=True)._rl_agent uses PERMemory ✓")


print("\n✅ All PER smoke tests passed!")
