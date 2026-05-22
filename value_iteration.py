import numpy as np
from environment import (
    NUM_STATES, NUM_ACTIONS, REWARDS, TRANSITIONS,
    is_terminal, state_to_rc, ACTION_NAMES
)

# ============================================================
# Value Iteration
#
# Bellman optimality update (deterministic):
#   V(s) ← max_a [ R(s,a) + γ · Σ_{s'} T(s,a,s') · V(s') ]
#
# We repeat until the value function converges (max change < θ).
# ============================================================

def value_iteration(
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iters: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Run value iteration.

    Parameters
    ----------
    gamma     : discount factor (0 < γ ≤ 1)
    theta     : convergence threshold (stop when max |ΔV| < θ)
    max_iters : safety cap on iterations

    Returns
    -------
    V        : optimal value function, shape (NUM_STATES,)
    policy   : greedy policy derived from V, shape (NUM_STATES,)
               policy[s] ∈ {0,1,2,3} (Up/Down/Left/Right)
    iters    : number of sweeps performed
    """
    V = np.zeros(NUM_STATES)   # initialise all values to 0

    for i in range(1, max_iters + 1):
        delta = 0.0
        V_new = np.zeros(NUM_STATES)

        for s in range(NUM_STATES):
            if is_terminal(s):
                V_new[s] = 0.0
                continue

            # Q(s, a) = R(s,a) + γ · Σ_{s'} T(s,a,s') · V(s')
            Q = REWARDS[s] + gamma * (TRANSITIONS[s] @ V)
            # shape note: TRANSITIONS[s] is (NUM_ACTIONS, NUM_STATES)
            #             TRANSITIONS[s] @ V  gives (NUM_ACTIONS,)

            V_new[s] = Q.max()
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < theta:
            print(f"Value iteration converged after {i} iterations "
                  f"(max |ΔV| = {delta:.2e})")
            break
    else:
        print(f"Value iteration reached max iterations ({max_iters}).")

    policy = extract_policy(V, gamma)
    return V, policy, i


def extract_policy(V: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """
    Derive the greedy policy from a value function.

    policy[s] = argmax_a [ R(s,a) + γ · Σ_{s'} T(s,a,s') · V(s') ]
    """
    policy = np.zeros(NUM_STATES, dtype=int)
    for s in range(NUM_STATES):
        if is_terminal(s):
            policy[s] = 0  # arbitrary for terminal
            continue
        Q = REWARDS[s] + gamma * (TRANSITIONS[s] @ V)
        policy[s] = int(Q.argmax())
    return policy


def print_value_grid(V: np.ndarray) -> None:
    """Pretty-print the value function as a 4×12 grid."""
    print("\n── Value Function V(s) ──")
    from environment import NUM_ROWS, NUM_COLS, CLIFF_STATES, GOAL_STATE, START_STATE
    for r in range(NUM_ROWS):
        row_str = ""
        for c in range(NUM_COLS):
            s = r * NUM_COLS + c
            if s in CLIFF_STATES:
                row_str += "  CLIFF "
            elif s == GOAL_STATE:
                row_str += "   GOAL "
            else:
                row_str += f"{V[s]:7.1f} "
        print(row_str)
    print()


def print_policy_grid(policy: np.ndarray) -> None:
    """Pretty-print the policy as arrows on a 4×12 grid."""
    ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    print("── Optimal Policy π(s) ──")
    from environment import NUM_ROWS, NUM_COLS, CLIFF_STATES, GOAL_STATE, START_STATE
    for r in range(NUM_ROWS):
        row_str = ""
        for c in range(NUM_COLS):
            s = r * NUM_COLS + c
            if s in CLIFF_STATES:
                row_str += " ▓ "
            elif s == GOAL_STATE:
                row_str += " G "
            elif s == START_STATE:
                row_str += f" {ARROWS[policy[s]]} "   # show arrow at S too
            else:
                row_str += f" {ARROWS[policy[s]]} "
        print(row_str)
    print()


# ------------------------------------------------------------------ #
#  Quick self-test                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    V, policy, iters = value_iteration(gamma=0.99)
    print_value_grid(V)
    print_policy_grid(policy)

    from environment import START_STATE, GOAL_STATE
    print(f"V(start) = {V[START_STATE]:.2f}")
    print(f"V(goal)  = {V[GOAL_STATE]:.2f}")
    print(f"Action at start: {ACTION_NAMES[policy[START_STATE]]}")