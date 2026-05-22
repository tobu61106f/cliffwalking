"""
Cliff Walking — main entry point
=================================
Run this file to:
  1. Verify the environment is set up correctly
  2. Run value iteration to find the optimal policy
  3. Simulate the agent following that policy
"""

from environment import (
    NUM_STATES, NUM_ACTIONS, REWARDS, TRANSITIONS,
    START_STATE, GOAL_STATE, CLIFF_STATES, state_to_rc
)
from value_iteration import value_iteration, print_value_grid, print_policy_grid
from simulation import run_episode, run_multiple_episodes


def main():
    print("=" * 55)
    print("  CLIFF WALKING — Reinforcement Learning Demo")
    print("=" * 55)

    # ── Part 1: Environment overview ────────────────────────
    print(f"\n[Environment]")
    print(f"  Grid       : 4 rows × 12 cols  ({NUM_STATES} states)")
    print(f"  Actions    : {NUM_ACTIONS}  (Up / Down / Left / Right)")
    print(f"  Start      : state {START_STATE}  → cell {state_to_rc(START_STATE)}")
    print(f"  Goal       : state {GOAL_STATE}  → cell {state_to_rc(GOAL_STATE)}")
    print(f"  Cliff      : {len(CLIFF_STATES)} cells along the bottom row")
    print(f"  Reward     : −1 per step, −100 if cliff, 0 at goal")

    # ── Part 2: Value Iteration ──────────────────────────────
    print("\n[Value Iteration]")
    gamma = 0.99
    print(f"  γ = {gamma}")
    V, policy, iters = value_iteration(gamma=gamma)
    print_value_grid(V)
    print_policy_grid(policy)

    # ── Part 3: Simulation ───────────────────────────────────
    print("[Simulation]  One verbose episode:")
    run_episode(policy, max_steps=200, verbose=True)

    print("\n[Simulation]  10 silent episodes:")
    run_multiple_episodes(policy, n=10, max_steps=200)


if __name__ == "__main__":
    main()

