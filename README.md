```
cliff_walking/
├── environment.py       # 环境（状态、动作、奖励、转移概率）
├── value_iteration.py   # 值迭代算法
├── simulation.py        # 模拟运行
└── main.py              # 主入口
```
<img width="1440" height="520" alt="image" src="https://github.com/user-attachments/assets/bd30f76c-28e6-4ab3-9a5e-cf250814311c" />

output:

```
[Environment]
  Grid       : 4 rows × 12 cols  (48 states)
  Actions    : 4  (Up / Down / Left / Right)
  Start      : state 36  → cell (3, 0)
  Goal       : state 47  → cell (3, 11)
  Cliff      : 10 cells along the bottom row
  Reward     : −1 per step, −100 if cliff, 0 at goal

[Value Iteration]
  γ = 0.99
Value iteration converged after 14 iterations (max |ΔV| = 0.00e+00)

── Value Function V(s) ──
  -12.2   -11.4   -10.5    -9.6    -8.6    -7.7    -6.8    -5.9    -4.9    -3.9    -3.0    -2.0 
  -11.4   -10.5    -9.6    -8.6    -7.7    -6.8    -5.9    -4.9    -3.9    -3.0    -2.0    -1.0 
  -10.5    -9.6    -8.6    -7.7    -6.8    -5.9    -4.9    -3.9    -3.0    -2.0    -1.0     0.0 
  -11.4   CLIFF   CLIFF   CLIFF   CLIFF   CLIFF   CLIFF   CLIFF   CLIFF   CLIFF   CLIFF    GOAL 

── Optimal Policy π(s) ──
 ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓ 
 ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓ 
 →  →  →  →  →  →  →  →  →  →  →  ↓ 
 ↑  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  G 

[Simulation]  One verbose episode:

───────────────────────────────────────────────────────
Step    State(r,c)  Action   Reward   → Next(r,c)
───────────────────────────────────────────────────────
   0  (3, 0)          Up       -1    (2, 0)
   1  (2, 0)       Right       -1    (2, 1)
   2  (2, 1)       Right       -1    (2, 2)
   3  (2, 2)       Right       -1    (2, 3)
   4  (2, 3)       Right       -1    (2, 4)
   5  (2, 4)       Right       -1    (2, 5)
   6  (2, 5)       Right       -1    (2, 6)
   7  (2, 6)       Right       -1    (2, 7)
   8  (2, 7)       Right       -1    (2, 8)
   9  (2, 8)       Right       -1    (2, 9)
  10  (2, 9)       Right       -1    (2,10)
  11  (2,10)       Right       -1    (2,11)
  12  (2,11)        Down        0    (3,11)
───────────────────────────────────────────────────────
Episode end: ✓ GOAL reached  |  steps=13  |  total reward=-12

[Simulation]  10 silent episodes:

═════════════════════════════════════════════
  Running 10 episodes (silent)
═════════════════════════════════════════════
  Ep   Steps    Total reward         Outcome
─────────────────────────────────────────────
   1      13             -12            GOAL
   2      13             -12            GOAL
   3      13             -12            GOAL
   4      13             -12            GOAL
   5      13             -12            GOAL
   6      13             -12            GOAL
   7      13             -12            GOAL
   8      13             -12            GOAL
   9      13             -12            GOAL
  10      13             -12            GOAL
─────────────────────────────────────────────
  Mean reward: -12.0  Std: 0.0
═════════════════════════════════════════════
```
