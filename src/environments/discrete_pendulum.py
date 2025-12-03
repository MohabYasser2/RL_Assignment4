"""
Discrete Action Wrapper for Pendulum Environment.

Pendulum-v1 has a continuous action space by default. This module provides
a wrapper that discretizes the action space into a fixed number of actions,
allowing DQN/DDQN agents to work with this environment.
"""

import gymnasium as gym
import numpy as np

# ============================================================================
# DISCRETE PENDULUM WRAPPER
# ============================================================================

class DiscretePendulum(gym.ActionWrapper):
    """
    Wrapper to discretize Pendulum-v1's continuous action space.
    
    The continuous action space [-2, 2] is divided into num_actions
    discrete bins, allowing discrete control algorithms to be applied.
    
    Args:
        env (gym.Env): The base Pendulum environment
        num_actions (int): Number of discrete actions to create
    """
    
    def __init__(self, env, num_actions=5):
        super().__init__(env)
        self.num_actions = num_actions
        # Convert action space from continuous to discrete
        self.action_space = gym.spaces.Discrete(num_actions)
        # Create mapping from discrete actions to continuous values
        self.action_map = np.linspace(-2, 2, num_actions)

    def action(self, action):
        """
        Convert discrete action to continuous action for the environment.
        
        Args:
            action (int): Discrete action index [0, num_actions-1]
            
        Returns:
            np.ndarray: Continuous action value
        """
        # Ensure action is an integer index
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        continuous_action = self.action_map[action]
        return np.array([continuous_action], dtype=np.float32)

    def reverse_action(self, action):
        """
        Convert continuous action back to discrete action index.
        
        Args:
            action (np.ndarray): Continuous action value
            
        Returns:
            int: Closest discrete action index
        """
        idx = np.argmin(np.abs(self.action_map - action[0]))
        return idx

# ============================================================================
# ENVIRONMENT FACTORY
# ============================================================================

def make_pendulum(num_discrete_actions=5, render_mode=None):
    """
    Create a Pendulum environment with discretized actions.
    
    Args:
        num_discrete_actions (int): Number of discrete actions
        render_mode (str, optional): Rendering mode ('human', 'rgb_array', None)
        
    Returns:
        gym.Env: Wrapped Pendulum environment with discrete actions
    """
    env = gym.make('Pendulum-v1', render_mode=render_mode)
    env = DiscretePendulum(env, num_actions=num_discrete_actions)
    return env
