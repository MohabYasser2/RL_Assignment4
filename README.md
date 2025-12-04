# Reinforcement Learning Algorithms: A2C, SAC, and PPO

This project implements three popular reinforcement learning algorithms on classical Gymnasium environments, with a **fully functional PPO implementation**.

## Algorithms

- **A2C** (Advantage Actor-Critic) - _Template provided_
- **SAC** (Soft Actor-Critic) - _Template provided_
- **PPO** (Proximal Policy Optimization) - ✅ **Fully Implemented**

## Environments

- CartPole-v1 (Discrete, Easy)
- Acrobot-v1 (Discrete, Moderate)
- MountainCar-v0 (Discrete, Sparse Rewards)
- Pendulum-v1 (Continuous, Dense Rewards)

## PPO Implementation Features

### Core Algorithm Components

- **Actor-Critic Architecture**: Shared feature extraction with separate policy and value heads
- **GAE (Generalized Advantage Estimation)**: For computing advantages with λ = 0.95
- **Clipped Surrogate Objective**: PPO's signature clipping mechanism (ε = 0.2)
- **Value Function Loss**: MSE loss for critic training
- **Entropy Bonus**: For maintaining exploration
- **Gradient Clipping**: Prevents unstable updates

### Key Features

- ✅ Support for both discrete and continuous action spaces
- ✅ Automatic device selection (CPU/CUDA)
- ✅ Progress bars with tqdm
- ✅ Weights & Biases integration
- ✅ Model checkpointing (save/load)
- ✅ Comprehensive testing and evaluation
- ✅ CSV export of test results
- ✅ Statistical analysis and visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training PPO

Train PPO on CartPole (easiest environment):

```bash
python src/train.py --algorithm ppo --environment CartPole-v1
```

Train on other environments:

```bash
python src/train.py --algorithm ppo --environment Acrobot-v1
python src/train.py --algorithm ppo --environment MountainCar-v0
python src/train.py --algorithm ppo --environment Pendulum-v1
```

Disable W&B logging:

```bash
python src/train.py --algorithm ppo --environment CartPole-v1 --no-wandb
```

### Testing Trained Models

Test a trained PPO agent (runs 100 episodes):

```bash
python src/test.py --algorithm ppo --environment CartPole-v1 --save-results
```

The `--save-results` flag will:

- Save statistics to a text file
- Export episode data to CSV
- Generate performance distribution plots

### Recording Videos

Record videos of your trained agent:

```bash
python src/record.py --algorithm ppo --environment CartPole-v1 --episodes 5
```

## Project Structure

```
.
├── configs/              # Configuration files for each algorithm
│   ├── a2c_config.yaml
│   ├── sac_config.yaml
│   └── ppo_config.yaml  # ✅ Optimized PPO hyperparameters
├── src/
│   ├── algorithms/       # Algorithm implementations
│   │   ├── a2c.py       # Template
│   │   ├── sac.py       # Template
│   │   └── ppo.py       # ✅ Full implementation
│   ├── environments.py   # Environment wrapper
│   ├── train.py         # ✅ Training script (PPO integrated)
│   ├── test.py          # ✅ Testing script (PPO integrated)
│   └── record.py        # ✅ Video recording (PPO integrated)
├── utils/               # Utility functions
│   ├── logger.py        # W&B integration
│   └── plotting.py      # Visualization functions
├── models/              # Saved models
├── videos/              # Recorded videos
└── results/             # Test results and plots
```

## PPO Hyperparameters

The configuration file (`configs/ppo_config.yaml`) includes optimized hyperparameters for each environment:

| Hyperparameter    | CartPole | Acrobot | MountainCar | Pendulum |
| ----------------- | -------- | ------- | ----------- | -------- |
| Learning Rate     | 3e-4     | 3e-4    | 3e-4        | 3e-4     |
| Trajectory Length | 2048     | 2048    | 2048        | 2048     |
| Batch Size        | 64       | 64      | 64          | 64       |
| Epochs per Update | 10       | 10      | 10          | 10       |
| Clip Range (ε)    | 0.2      | 0.2     | 0.2         | 0.2      |
| GAE Lambda (λ)    | 0.95     | 0.95    | 0.95        | 0.95     |
| Entropy Coef      | 0.01     | 0.01    | 0.05        | 0.0      |
| Max Episodes      | 500      | 1000    | 2000        | 500      |

## Expected Performance

### CartPole-v1

- **Solved**: Average reward ≥ 195 over 100 episodes
- **Training Time**: ~500 episodes
- **Expected**: Agent should learn to balance the pole consistently

### Acrobot-v1

- **Solved**: Average reward ≥ -100 over 100 episodes
- **Training Time**: ~1000 episodes
- **Expected**: Agent should swing up efficiently

### MountainCar-v0

- **Solved**: Average reward ≥ -110 over 100 episodes
- **Training Time**: ~2000 episodes (sparse rewards make this challenging)
- **Expected**: Agent should learn momentum-based strategy

### Pendulum-v1

- **Solved**: Average reward ≥ -200 over 100 episodes
- **Training Time**: ~500 episodes
- **Expected**: Smooth continuous control of pendulum

## Implementation Details

### PPO Algorithm Flow

1. **Rollout Phase**: Collect `trajectory_length` steps of experience
2. **Advantage Computation**: Calculate GAE advantages
3. **Policy Update**:
   - Sample minibatches from collected data
   - Compute clipped surrogate objective
   - Update policy for `n_epochs` epochs
4. **Value Update**: Update critic to predict returns
5. **Repeat**: Continue until training complete

### Network Architecture

```python
Actor-Critic Network:
  Shared Layers:
    Linear(state_dim → 64) + Tanh
    Linear(64 → 64) + Tanh

  Actor Head (Discrete):
    Linear(64 → action_dim)

  Actor Head (Continuous):
    Linear(64 → action_dim)  # mean
    Parameter(action_dim)     # log_std

  Critic Head:
    Linear(64 → 1)
```

## Logging

The project uses Weights & Biases (W&B) for experiment tracking. Login before training:

```bash
wandb login
```

Tracked metrics:

- Episode reward
- Episode length
- Actor loss
- Critic loss
- Entropy
- Total loss
- Average reward (rolling mean over 100 episodes)

## Testing the Implementation

Quick test to verify PPO works:

```bash
# Train for a few episodes on CartPole (fastest environment)
python src/train.py --algorithm ppo --environment CartPole-v1 --no-wandb
python src/train.py --algorithm sac --environment MountainCar-v0 --no-wandb

# Test the trained model
python src/test.py --algorithm sac --environment CartPole-v1 --save-results
python src/test.py --algorithm sac --environment Pendulum-v1 --save-results
python src/test.py --algorithm sac --environment Acrobot-v1 --save-results

# Record a video
python src/record.py --algorithm sac --environment CartPole-v1 --episodes 3
```

## Code Quality Features

- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ PEP 8 compliant
- ✅ Proper error handling
- ✅ Progress tracking with tqdm
- ✅ Device management (CPU/CUDA)
- ✅ Gradient clipping for stability
- ✅ Orthogonal weight initialization
- ✅ Clean separation of concerns

## License

MIT
