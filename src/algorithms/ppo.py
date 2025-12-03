"""
PPO (Proximal Policy Optimization) Algorithm Implementation.

This module contains the PPO algorithm for reinforcement learning.
Based on the Spinning Up in Deep RL implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, Any, Tuple, List
import numpy as np
from tqdm import tqdm
import gymnasium as gym


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    The actor outputs a policy distribution over actions.
    The critic outputs a value estimate for the current state.
    """
    
    def __init__(self, state_dim: int, action_dim: int, continuous: bool = False, hidden_dim: int = 64):
        """
        Initialize Actor-Critic networks.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            continuous: Whether action space is continuous
            hidden_dim: Hidden layer dimension (default: 64, use 128 for Pendulum)
        """
        super(ActorCritic, self).__init__()
        
        self.continuous = continuous
        
        # SEPARATE networks for actor and critic (CleanRL style)
        # Actor network
        if continuous:
            self.actor = nn.Sequential(
                self._layer_init(nn.Linear(state_dim, hidden_dim)),
                nn.Tanh(),
                self._layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                self._layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
            )
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # Discrete action space
            self.actor = nn.Sequential(
                self._layer_init(nn.Linear(state_dim, hidden_dim)),
                nn.Tanh(),
                self._layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                self._layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
            )
        
        # Critic network (separate from actor)
        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize layer with orthogonal weights (CleanRL style)."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass through both actor and critic.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action_distribution, value_estimate)
        """
        # Actor: get action distribution
        if self.continuous:
            action_mean = self.actor(state)
            action_std = torch.exp(self.actor_log_std)
            action_dist = Normal(action_mean, action_std)
        else:
            action_logits = self.actor(state)
            action_dist = Categorical(logits=action_logits)
        
        # Critic: get value estimate
        value = self.critic(state)
        
        return action_dist, value
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate for a state."""
        return self.critic(state)
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions taken in states.
        
        Args:
            states: State tensor
            actions: Action tensor
            
        Returns:
            Tuple of (log_probs, state_values, entropy)
        """
        action_dist, values = self.forward(states)
        
        log_probs = action_dist.log_prob(actions)
        if self.continuous:
            # Sum log probs for multi-dimensional continuous actions
            log_probs = log_probs.sum(dim=-1)
        
        entropy = action_dist.entropy()
        if self.continuous:
            entropy = entropy.sum(dim=-1)
        
        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """
    Buffer for storing trajectory data during rollout.
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def store(self, state, action, log_prob, reward, value, done):
        """Store a single transition."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self) -> Tuple[torch.Tensor, ...]:
        """Get all stored data as tensors."""
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(np.array(self.rewards)),
            torch.FloatTensor(np.array(self.values)),
            torch.FloatTensor(np.array(self.dones))
        )
    
    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
    
    PPO is an on-policy algorithm that uses a clipped surrogate objective
    to prevent large policy updates while maintaining sample efficiency.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any], continuous: bool = False):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary with hyperparameters
            continuous: Whether the action space is continuous
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.continuous = continuous
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.lr = config.get('learning_rate', 3e-4)
        self.gamma = config.get('discount_factor', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.epsilon_clip = config.get('clip_range', 0.2)
        self.n_epochs = config.get('n_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.trajectory_length = config.get('replay_memory_size', 2048)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.entropy_coef_final = config.get('entropy_coef_final', self.entropy_coef * 0.1)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.clip_vloss = config.get('clip_vloss', True)  # Value function clipping
        self.hidden_dim = config.get('hidden_dim', 64)  # Hidden layer size
        
        # Initialize networks
        self.actor_critic = ActorCritic(state_dim, action_dim, continuous, hidden_dim=self.hidden_dim).to(self.device)
        
        # Initialize optimizer (eps=1e-5 like CleanRL for better stability)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=1e-5)
        
        # Learning rate annealing
        self.total_updates = config.get('total_updates', 1000)
        self.update_step = 0
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select an action given the current state.
        
        Args:
            state: Current state
            deterministic: If True, select action deterministically (use mean for continuous, argmax for discrete)
            
        Returns:
            Tuple of (action, log_probability, value_estimate)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_dist, value = self.actor_critic(state_tensor)
            
            if deterministic:
                if self.continuous:
                    action = action_dist.mean
                else:
                    action = torch.argmax(action_dist.probs, dim=-1)
            else:
                action = action_dist.sample()
            
            log_prob = action_dist.log_prob(action)
            if self.continuous:
                log_prob = log_prob.sum(dim=-1)
        
        action_np = action.cpu().numpy().flatten()
        if not self.continuous:
            action_np = action_np[0]
        
        return action_np, log_prob.item(), value.item()
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Tensor of rewards
            values: Tensor of value estimates
            dones: Tensor of done flags
            next_value: Value estimate for the next state after trajectory
            
        Returns:
            Tuple of (advantages, returns)
        """
        # Move tensors to device
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae = 0
        
        # Append next_value for bootstrap
        values_extended = torch.cat([values, torch.tensor([next_value], device=self.device)])
        
        # Compute GAE backwards through trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values_extended[t + 1]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            
            # GAE: A_t = δ_t + γ * λ * A_{t+1}
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        # Returns = Advantages + Values
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, 
               old_values: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor) -> Dict[str, float]:
        """
        Perform PPO update using collected trajectory data.
        
        Args:
            states: State tensor
            actions: Action tensor
            old_log_probs: Log probabilities from old policy
            old_values: Value estimates from old critic
            returns: Computed returns
            advantages: Computed advantages
            
        Returns:
            Dictionary of loss components
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        old_values = old_values.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        # Training metrics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_loss = 0
        n_updates = 0
        approx_kl = 0.0  # Initialize outside loops
        
        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            # Create random indices for minibatch sampling
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # Track KL for this epoch
            epoch_approx_kl = 0.0
            epoch_updates = 0
            
            # Minibatch updates
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions with current policy
                log_probs, values, entropy = self.actor_critic.evaluate_actions(batch_states, batch_actions)
                
                # Track approximate KL divergence (don't break yet)
                batch_approx_kl = (batch_old_log_probs - log_probs).mean().item()
                epoch_approx_kl += batch_approx_kl
                epoch_updates += 1
                approx_kl = batch_approx_kl  # Store for logging
                
                # Compute ratio: π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss with optional clipping (CleanRL style)
                if self.clip_vloss:
                    # Clipped value loss
                    values_pred = values
                    values_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values,
                        -self.epsilon_clip,
                        self.epsilon_clip
                    )
                    v_loss_unclipped = F.mse_loss(values_pred, batch_returns)
                    v_loss_clipped = F.mse_loss(values_clipped, batch_returns)
                    critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
                else:
                    # Standard MSE loss
                    critic_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss (use current entropy coefficient from annealing)
                # Note: entropy_coef will be computed after the loop, so use self.entropy_coef for now
                loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_loss += loss.item()
                n_updates += 1
            
            # Check KL divergence after each epoch (not per minibatch)
            if epoch_updates > 0:
                avg_epoch_kl = epoch_approx_kl / epoch_updates
                if avg_epoch_kl > 0.05:  # Higher threshold: 0.05 instead of 0.03
                    # Policy changed too much, stop further epochs
                    break
        
        # Learning rate and entropy annealing
        self.update_step += 1
        frac = 1.0 - (self.update_step / self.total_updates)
        new_lr = self.lr * max(frac, 0.0)  # Don't go below 0
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr
        
        # Anneal entropy coefficient (high early for exploration, low late for convergence)
        current_entropy_coef = self.entropy_coef_final + (self.entropy_coef - self.entropy_coef_final) * frac
        
        # Average losses
        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'total_loss': total_loss / n_updates,
            'learning_rate': new_lr,
            'entropy_coef': current_entropy_coef,
            'approx_kl': approx_kl if n_updates > 0 else 0.0
        }
    
    def train(self, env, config: Dict[str, Any], logger=None) -> Dict[str, Any]:
        """
        Train the PPO agent until convergence or max episodes.
        Supports both single and vectorized environments.
        
        Args:
            env: Environment to train on (single or vectorized)
            config: Configuration dictionary containing convergence criteria or num_episodes
            logger: Optional W&B logger
            
        Returns:
            Dictionary containing training metrics and statistics
        """
        # Check if using vectorized environment
        num_envs = getattr(env, 'num_envs', 1)
        is_vectorized = num_envs > 1
        
        # Check if using convergence-based training
        use_convergence = 'convergence_threshold' in config
        
        if use_convergence:
            convergence_threshold = config['convergence_threshold']
            convergence_window = config.get('convergence_window', 100)
            min_episodes = config.get('min_episodes', 100)
            max_episodes = config.get('max_episodes', 10000)
            print(f"Training until convergence (threshold: {convergence_threshold}, window: {convergence_window})")
            print(f"Min episodes: {min_episodes}, Max episodes: {max_episodes}")
        else:
            max_episodes = config.get('episodes', config.get('num_episodes', 1000))
            print(f"Training for {max_episodes} episodes...")
        
        if is_vectorized:
            print(f"Using {num_envs} parallel environments")
            # For vectorized envs, track episodes per environment
            states, _ = env.reset()
            episode_rewards = np.zeros(num_envs)
            episode_lengths = np.zeros(num_envs, dtype=int)
        else:
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        
        episode_count = 0
        total_steps = 0
        
        # Progress bar (will update dynamically)
        pbar = tqdm(desc="Training PPO")
        
        best_avg_reward = -float('inf')
        recent_rewards = []
        converged = False
        
        while episode_count < max_episodes and not converged:
            # Collect trajectory
            for _ in range(self.trajectory_length):
                if is_vectorized:
                    # Vectorized environment handling
                    actions = []
                    log_probs = []
                    values = []
                    
                    for i in range(num_envs):
                        action, log_prob, value = self.select_action(states[i], deterministic=False)
                        actions.append(action)
                        log_probs.append(log_prob)
                        values.append(value)
                    
                    actions = np.array(actions)
                    log_probs = np.array(log_probs)
                    values = np.array(values)
                    
                    # Take steps in all environments
                    next_states, rewards, terminateds, truncateds, _ = env.step(actions)
                    dones = terminateds | truncateds
                    
                    # Store transitions for each environment
                    for i in range(num_envs):
                        self.buffer.store(states[i], actions[i], log_probs[i], 
                                        rewards[i], values[i], dones[i])
                        
                        episode_rewards[i] += rewards[i]
                        episode_lengths[i] += 1
                        total_steps += 1
                        
                        if dones[i]:
                            # Episode finished in environment i
                            self.episode_rewards.append(episode_rewards[i])
                            self.episode_lengths.append(episode_lengths[i])
                            recent_rewards.append(episode_rewards[i])
                            if len(recent_rewards) > 100:
                                recent_rewards.pop(0)
                            
                            avg_reward = np.mean(recent_rewards)
                            
                            # Log to W&B
                            if logger:
                                logger.log_episode(
                                    episode=episode_count,
                                    reward=episode_rewards[i],
                                    duration=episode_lengths[i],
                                    average_reward=avg_reward
                                )
                            
                            episode_count += 1
                            pbar.update(1)
                            pbar.set_postfix({
                                'episode': episode_count,
                                'reward': f'{episode_rewards[i]:.2f}',
                                'avg_reward': f'{avg_reward:.2f}',
                                'length': episode_lengths[i]
                            })
                            
                            # Check for convergence
                            if use_convergence and episode_count >= min_episodes:
                                if len(recent_rewards) >= convergence_window:
                                    window_avg = np.mean(recent_rewards[-convergence_window:])
                                    if window_avg >= convergence_threshold:
                                        converged = True
                                        print(f"\nConverged! Average reward {window_avg:.2f} >= {convergence_threshold:.2f}")
                            
                            # Reset this environment
                            episode_rewards[i] = 0
                            episode_lengths[i] = 0
                            
                            if episode_count >= max_episodes or converged:
                                break
                    
                    states = next_states
                    
                else:
                    # Single environment handling (original code)
                    action, log_prob, value = self.select_action(state, deterministic=False)
                    
                    # Take step in environment
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # Store transition
                    self.buffer.store(state, action, log_prob, reward, value, done)
                    
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    total_steps += 1
                    
                    if done:
                        # Episode finished
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        recent_rewards.append(episode_reward)
                        if len(recent_rewards) > 100:
                            recent_rewards.pop(0)
                        
                        avg_reward = np.mean(recent_rewards)
                        
                        # Log to W&B
                        if logger:
                            logger.log_episode(
                                episode=episode_count,
                                reward=episode_reward,
                                duration=episode_length,
                                average_reward=avg_reward
                            )
                        
                        episode_count += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            'episode': episode_count,
                            'reward': f'{episode_reward:.2f}',
                            'avg_reward': f'{avg_reward:.2f}',
                            'length': episode_length
                        })
                        
                        # Check for convergence
                        if use_convergence and episode_count >= min_episodes:
                            if len(recent_rewards) >= convergence_window:
                                window_avg = np.mean(recent_rewards[-convergence_window:])
                                if window_avg >= convergence_threshold:
                                    converged = True
                                    print(f"\nConverged! Average reward {window_avg:.2f} >= {convergence_threshold:.2f}")
                        
                        # Reset for next episode
                        state, _ = env.reset()
                        episode_reward = 0
                        episode_length = 0
                        
                        if episode_count >= max_episodes or converged:
                            break
                
                if episode_count >= max_episodes or converged:
                    break
            
            # Get value estimate for last state (for bootstrapping)
            if is_vectorized:
                # For vectorized envs, get values for all current states
                with torch.no_grad():
                    states_tensor = torch.FloatTensor(states).to(self.device)
                    next_values = self.actor_critic.get_value(states_tensor).cpu().numpy()
                # Use average for simplicity (could be more sophisticated)
                next_value = np.mean(next_values)
            else:
                # Single environment
                if done:
                    next_value = 0.0
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        next_value = self.actor_critic.get_value(state_tensor).item()
            
            # Get trajectory data
            states_batch, actions_batch, log_probs_batch, rewards_batch, values_batch, dones_batch = self.buffer.get()
            
            # Compute advantages and returns using GAE
            advantages, returns = self.compute_gae(rewards_batch, values_batch, dones_batch, next_value)
            
            # Perform PPO update (pass old values for value clipping)
            update_info = self.update(states_batch, actions_batch, log_probs_batch, values_batch, returns, advantages)
            
            # Log update metrics
            if logger and episode_count > 0:
                logger.log(update_info, step=episode_count)
            
            # Clear buffer
            self.buffer.clear()
        
        pbar.close()
        
        # Print final convergence status
        if use_convergence:
            if converged:
                print(f"\nTraining converged after {episode_count} episodes!")
            else:
                print(f"\nTraining stopped at max episodes ({max_episodes}) without full convergence.")
        else:
            print(f"\nTraining completed!")
        
        print(f"Mean reward (last 100 episodes): {np.mean(recent_rewards[-100:]) if len(recent_rewards) >= 100 else np.mean(recent_rewards):.2f}")
        print(f"Mean episode length: {np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths):.2f}")
        
        # Compute final statistics
        final_stats = {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths),
            'total_steps': total_steps,
            'converged': converged if use_convergence else None
        }
        
        print(f"\nTraining completed!")
        print(f"Mean reward (last 100 episodes): {final_stats['mean_reward']:.2f}")
        print(f"Mean episode length: {final_stats['mean_length']:.2f}")
        
        return final_stats
    
    def test(self, env, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Test the PPO agent.
        
        Args:
            env: Environment to test on
            num_episodes: Number of test episodes
            
        Returns:
            Dictionary containing test metrics and statistics
        """
        print(f"Testing PPO agent for {num_episodes} episodes...")
        
        test_rewards = []
        test_lengths = []
        
        for episode in tqdm(range(num_episodes), desc="Testing"):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Select action deterministically
                action, _, _ = self.select_action(state, deterministic=True)
                
                # Take step
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)
        
        # Compute statistics
        test_stats = {
            'mean_reward': np.mean(test_rewards),
            'std_reward': np.std(test_rewards),
            'min_reward': np.min(test_rewards),
            'max_reward': np.max(test_rewards),
            'mean_duration': np.mean(test_lengths),
            'std_duration': np.std(test_lengths),
            'min_duration': np.min(test_lengths),
            'max_duration': np.max(test_lengths),
            'episode_rewards': test_rewards,
            'episode_durations': test_lengths
        }
        
        print(f"\nTest Results:")
        print(f"Mean Reward: {test_stats['mean_reward']:.2f} ± {test_stats['std_reward']:.2f}")
        print(f"Mean Duration: {test_stats['mean_duration']:.2f} ± {test_stats['std_duration']:.2f}")
        
        return test_stats
    
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'episode_rewards' in checkpoint:
            self.episode_rewards = checkpoint['episode_rewards']
        if 'episode_lengths' in checkpoint:
            self.episode_lengths = checkpoint['episode_lengths']
        print(f"Model loaded from {path}")
