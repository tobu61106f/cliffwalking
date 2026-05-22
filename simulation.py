import numpy as np
from environment import (
    START_STATE, GOAL_STATE, CLIFF_STATES, TRANSITIONS, REWARDS,
    state_to_rc, ACTION_NAMES, is_terminal
)

# ============================================================
# Simulation
#
# Each episode:
#   1. Start at START_STATE
#   2. At each step: observe s, pick a = π(s), observe r, move to s'
#   3. End when s == GOAL_STATE or max_steps reached
# ============================================================


def step(state: int, action: int) -> tuple[int, float]:
    """
    Execute one transition in the environment.

    Returns
    -------
    next_state : int
    reward     : float
    """
    reward = REWARDS[state, action]
    # Deterministic: argmax of one-hot transition row
    next_state = int(np.argmax(TRANSITIONS[state, action]))
    return next_state, reward


def run_episode(
    policy: np.ndarray,
    max_steps: int = 200,
    verbose: bool = True,
) -> tuple[list[int], list[int], list[float], float]:
    """
    Run one episode following `policy`.

    Returns
    -------
    states  : list of states visited
    actions : list of actions taken
    rewards : list of rewards received
    total_r : total (undiscounted) return
    """
    state = START_STATE
    states, actions, rewards = [state], [], []
    total_r = 0.0

    if verbose:
        print(f"\n{'─'*55}")
        print(f"{'Step':>4}  {'State(r,c)':>12}  {'Action':>6}  "
              f"{'Reward':>7}  {'→ Next(r,c)':>12}")
        print(f"{'─'*55}")

    for t in range(max_steps):
        if is_terminal(state):
            break

        action = policy[state]
        next_state, reward = step(state, action)
        total_r += reward

        actions.append(action)
        rewards.append(reward)

        if verbose:
            r0, c0 = state_to_rc(state)
            r1, c1 = state_to_rc(next_state)
            cliff_flag = " ← CLIFF!" if state in CLIFF_STATES or next_state in CLIFF_STATES else ""
            print(f"{t:>4}  ({r0},{c0:>2})      "
                  f"{ACTION_NAMES[action]:>6}  "
                  f"{reward:>7.0f}  "
                  f"  ({r1},{c1:>2}){cliff_flag}")

        state = next_state
        states.append(state)

    if verbose:
        outcome = "✓ GOAL reached" if is_terminal(state) else "✗ max steps"
        print(f"{'─'*55}")
        print(f"Episode end: {outcome}  |  steps={len(actions)}  "
              f"|  total reward={total_r:.0f}")

    return states, actions, rewards, total_r


def run_multiple_episodes(
    policy: np.ndarray,
    n: int = 10,
    max_steps: int = 200,
) -> None:
    """
    Run n silent episodes and print a summary table.
    """
    print(f"\n{'═'*45}")
    print(f"  Running {n} episodes (silent)")
    print(f"{'═'*45}")
    print(f"{'Ep':>4}  {'Steps':>6}  {'Total reward':>14}  {'Outcome':>14}")
    print(f"{'─'*45}")

    total_rewards = []
    for ep in range(1, n + 1):
        states, actions, rewards, total_r = run_episode(
            policy, max_steps=max_steps, verbose=False
        )
        outcome = "GOAL" if is_terminal(states[-1]) else "timeout"
        total_rewards.append(total_r)
        print(f"{ep:>4}  {len(actions):>6}  {total_r:>14.0f}  {outcome:>14}")

    print(f"{'─'*45}")
    print(f"  Mean reward: {np.mean(total_rewards):.1f}  "
          f"Std: {np.std(total_rewards):.1f}")
    print(f"{'═'*45}\n")


# ------------------------------------------------------------------ #
#  Quick self-test                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Build a dummy "always go right then up" policy for a quick test
    dummy = np.zeros(48, dtype=int)  # all Up
    # Use actual optimal policy from value iteration instead
    from value_iteration import value_iteration
    _, policy, _ = value_iteration(gamma=0.99)

    # One verbose episode
    run_episode(policy, verbose=True)

    # 5 silent episodes
    run_multiple_episodes(policy, n=5)