# Soft Actor-Critic (SAC) Algorithm

## Overview

**SAC** (Soft Actor-Critic) is an off-policy reinforcement learning algorithm that combines:

- **Actor-Critic Architecture**: Policy (actor) and value function (critic) networks
- **Entropy Regularization**: Encourages exploration by maximizing policy entropy alongside reward
- **Clipped Double Q-learning**: Two critics reduce overestimation of Q-values
- **Automatic Entropy Tuning** (optional): Adaptive temperature coefficient for entropy bonus

SAC is particularly effective for **discrete action spaces** and provides stable training with good sample efficiency.

---

## Algorithm Components

### 1. Networks

The SAC implementation uses three main components:

#### **Actor Network (Policy)**

- Maps states → action probabilities (for discrete actions)
- Uses categorical distribution over discrete actions
- Architecture: `state_dim → hidden → hidden → action_dim`
- Output: logits that generate softmax probabilities

```python
class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, action_dim)
```

#### **Critic Networks (Q-Functions)**

- Two independent Q-networks estimate action values
- Both map states → Q-values for all actions
- Same architecture: `state_dim → hidden → hidden → action_dim`
- Dual critics: Q1 and Q2 (reduces overestimation bias)

```python
class DiscreteCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q = nn.Linear(hidden, action_dim)
```

#### **Target Critic Networks**

- Delayed copies of Q1 and Q2 for stability
- Updated via soft-update (polyak averaging): `target ← τ·source + (1-τ)·target`
- Prevents moving-target problem

### 2. Replay Buffer

- Off-policy learning: stores transitions `(s, a, r, s', done)` in a deque
- Fixed max size (e.g., 50,000)
- Enables batch sampling for mini-batch training

```python
self.replay = deque(maxlen=config["replay_memory_size"])
```

### 3. Entropy Temperature (α)

- Controls exploration-exploitation tradeoff
- Can be fixed or learned automatically
- **Fixed**: Set `alpha` in config (e.g., 0.01)
- **Automatic**: Learn α to maintain target entropy level

---

## Training Algorithm

### Phase 1: Experience Collection (ε-Greedy)

```
For each step:
  1. With probability ε: take random action (exploration)
  2. Otherwise: sample action from policy π(a|s)
  3. Execute action, receive reward r, observe s'
  4. Store (s, a, r, s', done) in replay buffer
  5. Decay ε: ε ← max(ε · decay_rate, ε_min)
```

### Phase 2: Network Update (SAC Core)

When `|replay_buffer| >= batch_size`:

#### **1. Compute Target Q-value**

```
For next states s':
  - Get actor policy: π(·|s') = softmax(actor(s'))
  - Evaluate both critics: Q1(s'), Q2(s')
  - Min operator (double Q): Q_min = min(Q1, Q2)

  - Soft Bellman backup:
    V(s') = Σ_a π(a|s') · [Q_min(s',a) - α·log(π(a|s'))]
    target_Q = r + γ·(1-done)·V(s')
```

#### **2. Update Critic Networks**

```
For both Q1 and Q2:
  - Predict: Q(s,a) = critic(s).gather(actions)
  - Loss: L = (Q(s,a) - target_Q)²
  - Backprop and update critic optimizer
```

#### **3. Update Actor (Policy)**

```
- Get policy from actor: π(·|s) = softmax(actor(s))
- Get Q-values: Q1(s), Q2(s)
- Min operator: Q_min = min(Q1, Q2)

- Actor loss (maximize expected Q - entropy):
  L = -Σ_a π(a|s) · [Q_min(s,a) - α·log(π(a|s))]

- Backprop and update actor optimizer
```

#### **4. Update Entropy Temperature (if automatic)**

```
- Measure current entropy: H = -Σ_a π(a|s)·log(π(a|s))
- Target entropy: H_target = -log(1/action_dim) · 0.98

- Temperature loss: L_α = -α · (H - H_target)
- Backprop and update α via alpha optimizer

- Update α: α ← exp(log_α)
```

#### **5. Soft Update Target Networks**

```
target ← τ·critic + (1-τ)·target  (for both target1, target2)
```

---

## Key Hyperparameters

| Parameter                  | Default    | Purpose                                |
| -------------------------- | ---------- | -------------------------------------- |
| `learning_rate`            | 0.0003     | Optimizer step size (actor & critic)   |
| `discount_factor` (γ)      | 0.99       | Reward discounting                     |
| `replay_memory_size`       | 50,000     | Size of experience buffer              |
| `batch_size`               | 64         | Mini-batch size for updates            |
| `tau` (τ)                  | 0.005      | Soft-update rate for targets           |
| `alpha`                    | 0.01       | Entropy temperature (if not automatic) |
| `automatic_entropy_tuning` | true/false | Learn α dynamically                    |
| `epsilon_start`            | 1.0        | Initial ε-greedy exploration           |
| `epsilon_decay`            | 0.995      | ε decay per episode                    |
| `epsilon_min`              | 0.05       | Minimum ε (don't stop exploring)       |
| `episodes`                 | 800        | Total training episodes                |

---

## Implementation Details (from `src/algorithms/sac.py`)

### Network Sizes

```python
hidden_size = 128  # Hidden layer dimension for all networks
```

### Device Management

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Automatically uses GPU if available for faster training.

### Action Selection

```python
def select_action(self, state, deterministic=False):
    # ε-greedy during training:
    if random.random() < self.epsilon:
        return random.randint(0, action_dim-1)  # Random action

    # Otherwise use policy:
    logits = actor(state)
    probs = softmax(logits)
    if deterministic:
        return argmax(probs)  # Greedy
    else:
        return sample(Categorical(probs))  # Stochastic
```

### Batch Sampling

```python
def sample_batch(self):
    batch = random.sample(self.replay, self.batch_size)
    # Convert to tensors and move to device
    return states, actions, rewards, next_states, dones
```

### Soft Update

```python
def soft_update(self, target, source):
    # Polyak averaging: slowly blend source into target
    for t, s in zip(target.parameters(), source.parameters()):
        t.data = τ·s.data + (1-τ)·t.data
```

---

## Hyperparameter Justification by Environment

### CartPole-v1 (Discrete, Easy)

CartPole is a simple balance task with dense rewards (reward per step) and fast convergence. The agent receives positive reward for each step it keeps the pole balanced.

| Parameter               | Value  | Justification                                                                             |
| ----------------------- | ------ | ----------------------------------------------------------------------------------------- |
| **Discount Factor (γ)** | 0.99   | Standard choice for most tasks; CartPole doesn't require extreme long-horizon planning    |
| **Learning Rate**       | 0.0003 | Standard actor-critic rate; CartPole converges quickly so doesn't need aggressive updates |
| **Replay Memory**       | 20,000 | Smaller buffer suffices — CartPole has simple, uniform state distribution                 |
| **Batch Size**          | 64     | Medium size provides stable gradients without excessive memory                            |
| **Tau (τ)**             | 0.005  | Slow target updates prevent oscillation early in learning                                 |
| **Entropy (α)**         | 0.05   | Moderate entropy helps exploration but CartPole doesn't need excessive randomness         |
| **Auto Entropy Tuning** | true   | Allows adaptive exploration as learning progresses                                        |
| **Epsilon Decay**       | 0.995  | Slow decay shifts from exploration to exploitation smoothly                               |
| **Episodes**            | 1000   | Sufficient to reach convergence (solve threshold ≥ 195 over 100 episodes)                 |

---

### Acrobot-v1 (Discrete, Harder with Sparse Rewards)

Acrobot requires swinging the lower link to reach a goal. Rewards are sparse: **-1 per step** until success, then goal-reaching reward. This demands longer, exploratory sequences.

| Parameter                | Value  | Justification                                                                                                                                         |
| ------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Discount Factor (γ)**  | 0.99   | High γ is **critical**: rewards appear only after long swing sequences → must value future heavily                                                    |
| **Actor Learning Rate**  | 0.0005 | **Slower** than CartPole: avoid instability while critics assign delayed rewards                                                                      |
| **Critic Learning Rate** | 0.001  | **Faster** than actor: critics must quickly learn value of success for long-term guidance                                                             |
| **Replay Memory**        | 30,000 | Smaller than MountainCar: Acrobot produces more varied state transitions during swing-up; 30k provides diversity without diluting rare success events |
| **Batch Size**           | 256    | **Large batches** reduce variance in Q-estimates (critical under sparse rewards) → stable gradients                                                   |
| **Tau (τ)**              | 0.005  | Slightly slower than CartPole: conservative updates provide stability on sparse-reward tasks                                                          |
| **Entropy (α)**          | 0.2    | **Higher entropy**: many actions lead to equally poor outcomes initially; high α forces discovery of swing-up strategies                              |
| **Epsilon Decay**        | 0.997  | **Very slow decay**: solving requires precise sequences; maintain exploration longer                                                                  |
| **Episodes**             | 500    | Fewer than MountainCar/Pendulum but more than CartPole; 500 ensures full swing-up coordination learning                                               |

**Key Insight**: Acrobot needs **strong exploration** (high α, slow ε decay) + **fast critic** learning to propagate delayed rewards backward through swing sequences.

---

### MountainCar-v0 (Discrete, Extremely Sparse Rewards)

MountainCar has **no intermediate reward** — agent only receives reward (+1) at goal. The car must build momentum by going left first (negative reward accumulates), then right to reach the peak.

| Parameter                | Value   | Justification                                                                                                                                 |
| ------------------------ | ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Discount Factor (γ)**  | 0.99    | **Highest importance**: the only reward is at the goal; γ must be very high to make delayed goal worth pursuing                               |
| **Actor Learning Rate**  | 0.0007  | Slightly higher than Acrobot: learning is inherently slow with no shaping; needs aggressive updates                                           |
| **Critic Learning Rate** | 0.001   | Even higher than actor: sparse reward signal requires fast Q-value propagation through the trajectory                                         |
| **Replay Memory**        | 100,000 | **Largest buffer**: useful transitions are rare; large storage increases probability of training on successful trajectories discovered later  |
| **Batch Size**           | 256     | **Large batches** critical: smooth gradient estimates under extreme sparsity                                                                  |
| **Tau (τ)**              | 0.005   | Balanced: stability needed but updates can't be too slow (rare reward signals degrade if delayed)                                             |
| **Entropy (α)**          | 0.0     | **No entropy**: random actions destroy momentum building → entropy _hurts_ performance; policy must be deterministic once strategy discovered |
| **Epsilon Decay**        | 0.999   | **Slowest decay**: exploration is mandatory to discover hill-climbing strategy; maintain ε-greedy phase very long                             |
| **Episodes**             | 2000    | **Longest training**: sparse rewards = slow learning; 2000 episodes ensure consistent solution discovery                                      |

**Key Insight**: MountainCar requires **maximum exploration** (zero entropy, slowest ε decay) + **no random noise** (entropy = 0) because exploration must be structured (building momentum).

---

### Pendulum-v1 (Continuous-Control → Discrete with ~11 actions)

Pendulum is where SAC excels. It's a continuous-control problem (convert to discrete via `--num-discrete-actions`). Reward is **dense** and proportional to angle and angular velocity: `-(angle² + 0.1·vel² + 0.001·action²)`.

| Parameter                    | Value   | Justification                                                                                                       |
| ---------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------- |
| **Discount Factor (γ)**      | 0.99    | Dense reward doesn't require extreme long-horizon focus; standard γ works well                                      |
| **Actor Learning Rate**      | 0.0003  | Fine control required → actor updates must be smooth and gradual                                                    |
| **Critic Learning Rate**     | 0.0005  | Slightly faster for accurate Q-targets in continuous action space                                                   |
| **Replay Memory**            | 100,000 | Large buffer captures full variety of swing dynamics, torque ranges, and velocities                                 |
| **Batch Size**               | 256     | Large batches stabilize learning in high-variance continuous action spaces                                          |
| **Tau (τ)**                  | 0.005   | Good trade-off: fast enough for steady progress, slow enough to prevent target instability                          |
| **Entropy (α)**              | 0.2     | **Moderate entropy** beneficial: exploration helps escape local equilibrium; too much = instability in fine control |
| **Automatic Entropy Tuning** | true    | Recommended: adapts α from exploration (initial swing-up) to exploitation (fine stabilization)                      |
| **Episodes**                 | 1000    | Longer training ensures smooth convergence of continuous actor-critic networks                                      |

**Key Insight**: Pendulum benefits from **moderate exploration + adaptive entropy** because initial random actions help swing-up, but fine control requires determinism once moving.

---

## Environment Comparison Summary

| Aspect                     | CartPole        | Acrobot          | MountainCar                  | Pendulum           |
| -------------------------- | --------------- | ---------------- | ---------------------------- | ------------------ |
| **Reward Density**         | Dense (+1/step) | Sparse (-1/step) | Extremely Sparse (goal only) | Dense (continuous) |
| **Exploration Difficulty** | Easy            | Hard             | Very Hard                    | Medium             |
| **Required Entropy**       | Low-Medium      | High             | Zero                         | Medium             |
| **Critic Learning Speed**  | Medium          | Fast             | Very Fast                    | Medium-Fast        |
| **Replay Buffer Size**     | 20K             | 30K              | 100K                         | 100K               |
| **Batch Size**             | 64              | 256              | 256                          | 256                |
| **Epsilon Decay**          | 0.995           | 0.997            | 0.999                        | 0.995              |
| **Typical Episodes**       | 100–200         | 500              | 2000                         | 500–1000           |

---

## Advantages of SAC

1. **Stability**: Dual Q-networks and target networks reduce overestimation
2. **Exploration**: Entropy bonus naturally encourages diverse behaviors
3. **Sample Efficiency**: Off-policy learning reuses past experiences
4. **Automatic Tuning**: Optional adaptive entropy temperature
5. **Discrete & Continuous**: Works with both action spaces (with modifications)
6. **Sparse Reward Friendly**: Large batch sizes and dual critics handle delayed signals

---

## Tips for Tuning

### If Training is Unstable

- Lower `learning_rate` (e.g., 0.0001)
- Increase `tau` (e.g., 0.01) for faster target tracking
- Disable `automatic_entropy_tuning` and set fixed `alpha`

### If Learning is Too Slow

- Increase `batch_size` (e.g., 128)
- Lower `replay_memory_size` for faster sampling
- Faster `epsilon_decay` (e.g., 0.99) to shift towards policy quickly

### If Agent Doesn't Explore Enough

- Increase `alpha` (if fixed) or `target_entropy` (if automatic)
- Slow down `epsilon_decay` for longer ε-greedy phase
- Reduce `epsilon_min` to keep random exploration longer

---
