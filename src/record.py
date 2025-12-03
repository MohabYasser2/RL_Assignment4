"""
Recording script for creating videos of trained RL agents.

This script uses Gymnasium's RecordVideo wrapper to record trained agents
and generates performance plots.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent))

from environments import EnvironmentWrapper
from algorithms import A2C, SAC, PPO
from utils.plotting import save_statistics_plot, plot_episode_durations


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


def record_episodes(
    agent,
    env,
    num_episodes: int,
    video_folder: Path,
    name_prefix: str
) -> List[int]:
    """
    Record episodes using the trained agent.
    
    Args:
        agent: Trained RL agent
        env: Gymnasium environment
        num_episodes: Number of episodes to record
        video_folder: Directory to save videos
        name_prefix: Prefix for video filenames
        
    Returns:
        List of episode durations
    """
    episode_durations = []
    
    # Wrap environment with RecordVideo
    env_wrapped = RecordVideo(
        env,
        video_folder=str(video_folder),
        name_prefix=name_prefix,
        episode_trigger=lambda x: True  # Record all episodes
    )
    
    for episode in range(num_episodes):
        state, _ = env_wrapped.reset()
        done = False
        truncated = False
        steps = 0
        total_reward = 0
        
        while not (done or truncated):
            # Select action using trained policy
            try:
                action, log_prob, value = agent.select_action(state, deterministic=True)
            except NotImplementedError:
                print(f"Error: select_action() not implemented")
                env_wrapped.close()
                raise
            
            # Take action
            state, reward, done, truncated, _ = env_wrapped.step(action)
            steps += 1
            total_reward += reward
        
        episode_durations.append(steps)
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Steps = {steps}, Reward = {total_reward:.2f}")
    
    env_wrapped.close()
    return episode_durations


def main():
    """Main recording function."""
    parser = argparse.ArgumentParser(
        description="Record videos of trained RL agents"
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        required=True,
        choices=['a2c', 'sac', 'ppo'],
        help='Algorithm used for training'
    )
    parser.add_argument(
        '--environment',
        type=str,
        required=True,
        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'Pendulum-v1'],
        help='Gymnasium environment to record'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the trained model (default: models/{algorithm}_{environment}.pth)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes to record (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--video-folder',
        type=str,
        default=None,
        help='Folder to save videos (default: videos/)'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine model path
    if args.model_path is None:
        model_path = Path(__file__).parent.parent / "models" / f"{args.algorithm}_{args.environment}.pth"
    else:
        model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    
    # Determine video folder
    if args.video_folder is None:
        video_folder = Path(__file__).parent.parent / "videos"
    else:
        video_folder = Path(args.video_folder)
    
    video_folder.mkdir(exist_ok=True, parents=True)
    
    # Load configuration
    config = load_config(args.algorithm, args.environment)
    
    # Initialize environment for recording (rgb_array mode for video)
    print(f"Initializing environment: {args.environment}")
    base_env_wrapper = EnvironmentWrapper(args.environment, render_mode='rgb_array')
    base_env = base_env_wrapper.env  # Get the wrapped environment
    
    # Get state and action dimensions
    state_dim = base_env_wrapper.get_state_dim()
    action_dim = base_env_wrapper.get_action_dim()
    is_continuous = not base_env_wrapper.is_discrete_action
    
    # Initialize algorithm
    print(f"Initializing {args.algorithm.upper()} algorithm...")
    AlgorithmClass = get_algorithm_class(args.algorithm)
    
    # For PPO, pass continuous flag
    if args.algorithm == 'ppo':
        agent = AlgorithmClass(state_dim, action_dim, config, continuous=is_continuous)
    else:
        agent = AlgorithmClass(state_dim, action_dim, config)
    
    # Load trained model
    try:
        agent.load(str(model_path))
        print("Model loaded successfully!")
    except NotImplementedError:
        print(f"Error: load() method not implemented for {args.algorithm}")
        sys.exit(1)
    
    # Record episodes
    print(f"\nRecording {args.episodes} episodes...")
    print(f"Videos will be saved to: {video_folder}")
    
    try:
        name_prefix = f"{args.algorithm}_{args.environment}"
        episode_durations = record_episodes(
            agent,
            base_env,
            args.episodes,
            video_folder,
            name_prefix
        )
        
        # Display statistics
        print("\n" + "="*50)
        print("RECORDING STATISTICS")
        print("="*50)
        print(f"Mean duration: {np.mean(episode_durations):.2f} steps")
        print(f"Std duration: {np.std(episode_durations):.2f} steps")
        print(f"Min duration: {np.min(episode_durations)} steps")
        print(f"Max duration: {np.max(episode_durations)} steps")
        print("="*50)
        
        # Generate and save duration plot
        plot_path = video_folder / f"{name_prefix}_durations.png"
        plot_episode_durations(
            episode_durations,
            title=f"{args.algorithm.upper()} on {args.environment} - Episode Durations",
            save_path=str(plot_path)
        )
        print(f"\nDuration plot saved to: {plot_path}")
        
    except NotImplementedError as e:
        print(f"\nError: {e}")
        print(f"Please implement the required methods in {args.algorithm}.py")
        sys.exit(1)
    
    print("\nRecording session finished!")
    print(f"Check {video_folder} for recorded videos")


if __name__ == "__main__":
    main()
