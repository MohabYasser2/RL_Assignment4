"""
Main training script for RL algorithms.

This script handles training of A2C, SAC, and PPO algorithms on various
Gymnasium environments with Weights & Biases logging.
Supports both single and parallel vectorized environments.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import gymnasium as gym

# Add src to path
sys.path.append(str(Path(__file__).parent))

from environments import EnvironmentWrapper
from algorithms import A2C, SAC, PPO
from utils.logger import WandBLogger


def load_config(algorithm: str, environment: str) -> dict:
    """
    Load configuration for the specified algorithm and environment.
    
    Args:
        algorithm: Algorithm name (a2c, sac, ppo)
        environment: Environment name
        
    Returns:
        Configuration dictionary for the environment
    """
    config_path = Path(__file__).parent.parent / "configs" / f"{algorithm}_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    if environment not in configs:
        raise ValueError(
            f"Environment {environment} not found in {algorithm} config. "
            f"Available: {list(configs.keys())}"
        )
    
    return configs[environment]


def get_algorithm_class(algorithm: str):
    """
    Get the algorithm class based on the algorithm name.
    
    Args:
        algorithm: Algorithm name (a2c, sac, ppo)
        
    Returns:
        Algorithm class
    """
    algorithms = {
        'a2c': A2C,
        'sac': SAC,
        'ppo': PPO
    }
    
    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: {list(algorithms.keys())}"
        )
    
    return algorithms[algorithm]


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train RL algorithms on Gymnasium environments"
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        required=True,
        choices=['a2c', 'sac', 'ppo'],
        help='Algorithm to use for training'
    )
    parser.add_argument(
        '--environment',
        type=str,
        required=True,
        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'Pendulum-v1'],
        help='Gymnasium environment to train on'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='rl-algorithms',
        help='W&B project name'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training (cpu/cuda)'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    
    # Load configuration
    print(f"Loading configuration for {args.algorithm} on {args.environment}...")
    config = load_config(args.algorithm, args.environment)
    print(f"Configuration: {config}")
    
    # Initialize environment (single or parallel)
    print(f"Initializing environment: {args.environment}")
    num_envs = config.get('num_envs', 1)
    
    if num_envs > 1:
        # Create parallel vectorized environments
        print(f"Creating {num_envs} parallel environments...")
        
        def make_env(env_id, seed, idx):
            """Create a single environment with unique seed."""
            def thunk():
                env = gym.make(env_id)
                env.action_space.seed(seed + idx)
                env.observation_space.seed(seed + idx)
                return env
            return thunk
        
        envs = gym.vector.SyncVectorEnv([
            make_env(args.environment, args.seed, i) for i in range(num_envs)
        ])
        
        # Wrap in a simple interface for compatibility
        class VectorEnvWrapper:
            def __init__(self, envs):
                self.envs = envs
                self.is_discrete_action = isinstance(envs.single_action_space, gym.spaces.Discrete)
                self.num_envs = envs.num_envs
                
            def reset(self):
                obs, info = self.envs.reset()
                return obs, info
                
            def step(self, action):
                return self.envs.step(action)
                
            def close(self):
                self.envs.close()
                
            def get_state_dim(self):
                return self.envs.single_observation_space.shape[0]
                
            def get_action_dim(self):
                if self.is_discrete_action:
                    return self.envs.single_action_space.n
                else:
                    return self.envs.single_action_space.shape[0]
        
        env = VectorEnvWrapper(envs)
        print(f"Using {num_envs} parallel environments")
    else:
        # Single environment
        env = EnvironmentWrapper(args.environment)
        env.num_envs = 1  # Add for consistency
        print("Using single environment")
    
    print(env)
    
    # Get state and action dimensions
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    is_continuous = not env.is_discrete_action
    
    # Initialize algorithm
    print(f"Initializing {args.algorithm.upper()} algorithm...")
    AlgorithmClass = get_algorithm_class(args.algorithm)
    
    # For PPO, pass continuous flag
    if args.algorithm == 'ppo':
        agent = AlgorithmClass(state_dim, action_dim, config, continuous=is_continuous)
    else:
        agent = AlgorithmClass(state_dim, action_dim, config)
    
    # Initialize W&B logger
    logger = None
    if not args.no_wandb:
        print("Initializing Weights & Biases...")
        logger = WandBLogger(
            project=args.wandb_project,
            config={
                'algorithm': args.algorithm,
                'environment': args.environment,
                'seed': args.seed,
                **config
            },
            name=f"{args.algorithm}_{args.environment}"
        )
    
    # Train the agent
    if 'convergence_threshold' in config:
        print(f"Training until convergence (target: {config['convergence_threshold']})...")
    else:
        print(f"Starting training for {config.get('episodes', 1000)} episodes...")
    try:
        training_stats = agent.train(env, config=config, logger=logger)
        
        # Log final statistics
        if logger:
            logger.log(training_stats)
            logger.finish()
        
        print(f"Training completed!")
        print(f"Training statistics: {training_stats}")
        
    except NotImplementedError as e:
        print(f"\nError: {e}")
        print(f"Please implement the train() method in {args.algorithm}.py")
        if logger:
            logger.finish()
        env.close()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        if logger:
            logger.finish()
        env.close()
        sys.exit(0)
    
    # Save the trained model
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{args.algorithm}_{args.environment}.pth"
    
    try:
        agent.save(str(model_path))
        print(f"Model saved to: {model_path}")
    except NotImplementedError:
        print(f"Warning: save() method not implemented for {args.algorithm}")
    
    # Close environment
    env.close()
    print("Training session finished!")


if __name__ == "__main__":
    main()
