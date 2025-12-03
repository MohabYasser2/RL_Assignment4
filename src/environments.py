"""
Environment wrapper module for Gymnasium environments.

This module provides a unified interface for different Gymnasium environments
used in the RL project (CartPole-v1, Acrobot-v1, MountainCar-v0, Pendulum-v1).
"""

import gymnasium as gym
from typing import Tuple, Any, Dict
import numpy as np
import sys
from pathlib import Path

# Add environments directory to path for discrete_pendulum import
sys.path.append(str(Path(__file__).parent / "environments"))
from discrete_pendulum import make_pendulum


class EnvironmentWrapper:
    """
    Unified wrapper for Gymnasium environments.
    
    Provides a consistent interface across different environments for easier
    algorithm implementation and testing.
    """
    
    SUPPORTED_ENVIRONMENTS = [
        "CartPole-v1",
        "Acrobot-v1", 
        "MountainCar-v0",
        "Pendulum-v1"
    ]
    
    def __init__(self, env_name: str, render_mode: str = None, num_discrete_actions: int = 5, use_discrete_pendulum: bool = False):
        """
        Initialize the environment wrapper.
        
        Args:
            env_name: Name of the Gymnasium environment
            render_mode: Rendering mode ('human', 'rgb_array', None)
            num_discrete_actions: Number of discrete actions for Pendulum (default: 5)
            use_discrete_pendulum: Whether to use discrete wrapper for Pendulum (default: False for continuous)
            
        Raises:
            ValueError: If environment name is not supported
        """
        if env_name not in self.SUPPORTED_ENVIRONMENTS:
            raise ValueError(
                f"Environment {env_name} not supported. "
                f"Supported environments: {self.SUPPORTED_ENVIRONMENTS}"
            )
        
        self.env_name = env_name
        
        # Use discrete wrapper for Pendulum only if specified
        if env_name == "Pendulum-v1" and use_discrete_pendulum:
            self.env = make_pendulum(num_discrete_actions=num_discrete_actions, render_mode=render_mode)
        else:
            self.env = gym.make(env_name, render_mode=render_mode)
        
        # Store environment properties
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.is_discrete_action = isinstance(
            self.action_space, gym.spaces.Discrete
        )
        
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for environment
            
        Returns:
            Tuple of (observation, info)
        """
        return self.env.reset(seed=seed)
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return self.env.step(action)
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()
    
    def render(self) -> Any:
        """Render the environment."""
        return self.env.render()
    
    def get_state_dim(self) -> int:
        """
        Get the dimension of the observation space.
        
        Returns:
            Dimension of observation space
        """
        if isinstance(self.observation_space, gym.spaces.Box):
            return self.observation_space.shape[0]
        return self.observation_space.n
    
    def get_action_dim(self) -> int:
        """
        Get the dimension of the action space.
        
        Returns:
            Dimension of action space
        """
        if isinstance(self.action_space, gym.spaces.Box):
            return self.action_space.shape[0]
        return self.action_space.n
    
    def sample_action(self) -> Any:
        """
        Sample a random action from the action space.
        
        Returns:
            Random action
        """
        return self.action_space.sample()
    
    def __str__(self) -> str:
        """String representation of the environment."""
        return (
            f"EnvironmentWrapper({self.env_name})\n"
            f"  Observation space: {self.observation_space}\n"
            f"  Action space: {self.action_space}\n"
            f"  Discrete actions: {self.is_discrete_action}"
        )
