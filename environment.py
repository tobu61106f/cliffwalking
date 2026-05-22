import numpy as np

# ============================================================
# Cliff Walking Environment
# Grid: 4 rows x 12 cols  (row 0 = top, row 3 = bottom)
# Start S  = (3, 0)   →  state index 36
# Goal  G  = (3, 11)  →  state index 47
# Cliff    = (3, 1..10)
#
# State encoding:  s = row * NUM_COLS + col
# Actions: 0=Up, 1=Down, 2=Left, 3=Right
# ============================================================

NUM_ROWS   = 4
NUM_COLS   = 12
NUM_STATES = NUM_ROWS * NUM_COLS   # 48
NUM_ACTIONS = 4                    # up / down / left / right

# Action index → (row_delta, col_delta)
ACTION_DELTAS = {
    0: (-1,  0),   # Up
    1: ( 1,  0),   # Down
    2: ( 0, -1),   # Left
    3: ( 0,  1),   # Right
}
ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

# Special state indices
START_STATE = 3 * NUM_COLS + 0    # 36
GOAL_STATE  = 3 * NUM_COLS + 11   # 47

# Cliff state indices (row 3, cols 1–10)
CLIFF_STATES = set(3 * NUM_COLS + c for c in range(1, 11))


def rc_to_state(row: int, col: int) -> int:
    """Convert (row, col) to a flat state index."""
    return row * NUM_COLS + col


def state_to_rc(s: int):
    """Convert a flat state index back to (row, col)."""
    return divmod(s, NUM_COLS)


def is_terminal(s: int) -> bool:
    """Goal is the only true terminal state."""
    return s == GOAL_STATE


def build_rewards() -> np.ndarray:
    """
    R[s, a]  – reward received when taking action a in state s.

    Rules:
      - Entering a cliff cell  → -100
      - Reaching the goal      →   0  (episode ends)
      - Any other transition   →  -1
    """
    R = np.full((NUM_STATES, NUM_ACTIONS), -1.0)

    for s in range(NUM_STATES):
        row, col = state_to_rc(s)
        for a in range(NUM_ACTIONS):
            dr, dc = ACTION_DELTAS[a]
            nr = max(0, min(NUM_ROWS - 1, row + dr))
            nc = max(0, min(NUM_COLS - 1, col + dc))
            ns = rc_to_state(nr, nc)

            if ns in CLIFF_STATES:
                R[s, a] = -100.0
            elif ns == GOAL_STATE:
                R[s, a] = 0.0   # reaching goal: no extra penalty
            # else stays -1.0

    return R


def build_transitions() -> np.ndarray:
    """
    T[s, a, s']  – probability of reaching s' after taking a in s.

    Deterministic dynamics:
      - Hitting a wall → stay in place.
      - Stepping onto a cliff cell → teleport back to START_STATE.
      - Terminal state (GOAL) → self-loop with probability 1.
    """
    T = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))

    for s in range(NUM_STATES):
        # Terminal absorbing state
        if is_terminal(s):
            T[s, :, s] = 1.0
            continue

        row, col = state_to_rc(s)
        for a in range(NUM_ACTIONS):
            dr, dc = ACTION_DELTAS[a]
            nr = max(0, min(NUM_ROWS - 1, row + dr))
            nc = max(0, min(NUM_COLS - 1, col + dc))
            ns = rc_to_state(nr, nc)

            # Cliff → reset to start
            if ns in CLIFF_STATES:
                ns = START_STATE

            T[s, a, ns] = 1.0

    return T


# Pre-build the arrays once; other modules import them directly.
REWARDS     = build_rewards()       # shape (48, 4)
TRANSITIONS = build_transitions()   # shape (48, 4, 48)


# ------------------------------------------------------------------ #
#  Quick self-test                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print(f"States: {NUM_STATES}, Actions: {NUM_ACTIONS}")
    print(f"Start={START_STATE} {state_to_rc(START_STATE)}, "
          f"Goal={GOAL_STATE} {state_to_rc(GOAL_STATE)}")
    print(f"Cliff states: {sorted(CLIFF_STATES)}")

    # Sanity checks
    assert TRANSITIONS.sum(axis=2).round(6).min() == 1.0, "Rows must sum to 1"
    print("Transition matrix sanity check passed ✓")

    # Example: from start, go right → should hit cliff → back to start
    ns_idx = np.argmax(TRANSITIONS[START_STATE, 3])  # action 3 = Right
    print(f"From START go Right → land on state {ns_idx} "
          f"(expected {START_STATE} because cliff)")
    print(f"Reward for that move: {REWARDS[START_STATE, 3]:.0f} (expected -100)")