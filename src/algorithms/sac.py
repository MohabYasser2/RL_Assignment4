import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# ================================
# ACTOR NETWORK (Categorical Softmax)
# ================================
class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.logits(x)

    def sample(self, state):
        logits = self(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, probs


# ================================
# CRITIC NETWORK (Q-values)
# ================================
class DiscreteCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q = nn.Linear(hidden, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.q(x)


# ================================
# SOFT ACTOR-CRITIC (DISCRETE)
# ================================
class SAC:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = config

        # -------------------------
        # Hyperparameters
        # -------------------------
        self.gamma = config["discount_factor"]
        self.tau = config["tau"]
        self.batch_size = config["batch_size"]
        self.alpha = config["alpha"]
        self.automatic_entropy_tuning = config["automatic_entropy_tuning"]

        # Replay buffer
        self.replay = deque(maxlen=config["replay_memory_size"])

        # Optimizer LRs
        actor_lr = config.get("actor_lr", 0.0003)
        critic_lr = config.get("critic_lr", 0.0003)

        # ε-greedy parameters
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.epsilon_min = config.get("epsilon_min", 0.05)

        # -------------------------
        # Networks
        # -------------------------
        self.actor = DiscreteActor(state_dim, action_dim)
        self.critic1 = DiscreteCritic(state_dim, action_dim)
        self.critic2 = DiscreteCritic(state_dim, action_dim)
        self.target1 = DiscreteCritic(state_dim, action_dim)
        self.target2 = DiscreteCritic(state_dim, action_dim)

        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        # -------------------------
        # Optimizers
        # -------------------------
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # -------------------------
        # Device
        # -------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.move_to_device()

        # -------------------------
        # Automatic entropy tuning
        # -------------------------
        if self.automatic_entropy_tuning:
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=actor_lr)

    # -------------------------
    def move_to_device(self):
        self.actor.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)
        self.target1.to(self.device)
        self.target2.to(self.device)

    # -------------------------
    # ε–greedy action selection
    # -------------------------
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            logits = self.actor(state)
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                return int(probs.argmax(dim=-1).item())

            dist = torch.distributions.Categorical(probs)
            return int(dist.sample().item())

    # -------------------------
    def store(self, transition):
        self.replay.append(transition)

    def sample_batch(self):
        batch = random.sample(self.replay, self.batch_size)

        s, a, r, ns, d = zip(*batch)

        states = torch.FloatTensor(np.array(s)).to(self.device)
        actions = torch.LongTensor(a).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(ns)).to(self.device)
        dones = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    # -------------------------
    def soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    # -------------------------
    # SAC update step
    # -------------------------
    def update_networks(self,logger=None):
        states, actions, rewards, next_states, dones = self.sample_batch()

        # =============== Target Q =================
        with torch.no_grad():
            logits_next = self.actor(next_states)
            probs_next = F.softmax(logits_next, dim=-1)
            log_probs_next = F.log_softmax(logits_next, dim=-1)

            q1_next = self.target1(next_states)
            q2_next = self.target2(next_states)
            q_next = torch.min(q1_next, q2_next)

            v_next = (probs_next * (q_next - self.alpha * log_probs_next)).sum(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * v_next

        # =============== Critic updates =================
        q1 = self.critic1(states).gather(1, actions)
        q2 = self.critic2(states).gather(1, actions)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # =============== Actor update =================
        logits = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        q1_all = self.critic1(states)
        q2_all = self.critic2(states)
        q_min = torch.min(q1_all, q2_all)

        actor_loss = (probs * (self.alpha * log_probs - q_min)).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # =============== Temperature update =================
        if self.automatic_entropy_tuning:
            entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self.alpha = float(self.log_alpha.exp().detach().cpu().item())

        # W&B LOGGING
# ===============================
        if logger is not None:
            logger.log({
                    "loss/critic1": critic1_loss.item(),
                    "loss/critic2": critic2_loss.item(),
                    "loss/policy": actor_loss.item(),
                    "q1_mean": q1.mean().item(),
                    "q2_mean": q2.mean().item(),
                    "v_target_mean": target_q.mean().item(),
                    "entropy": float((-(probs * log_probs).sum(dim=1).mean()).item()),
                    "epsilon": self.epsilon,
                    "alpha": self.alpha,
                })
        # Soft update targets
        self.soft_update(self.target1, self.critic1)
        self.soft_update(self.target2, self.critic2)

    # ========================
    # TRAIN LOOP
    # ========================
    def train(self, env, config=None, logger=None):
        """
        Train SAC for the number of episodes defined in config.
        config and logger are optional to stay compatible with train.py.
        """
        if config is None:
            config = self.cfg

        episodes = config.get("episodes", 500)
        rewards = []

        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            total = 0
            steps = 0     

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # store transition
                self.store((state, action, reward, next_state, done))
                state = next_state
                total += reward
                steps += 1   

                if len(self.replay) >= self.batch_size:
                    self.update_networks(logger)

            # decay exploration
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            rewards.append(total)
            print(f"Episode {ep}: Reward = {total}")

            if logger:
                logger.log({"episode_reward": total,
                            "episode_length": steps})

        return {"rewards": rewards}

    # ========================
    # TEST LOOP
    # ========================
    def test(self, env, episodes=10, num_episodes=None):
        """
        Test SAC agent for a number of episodes.
        This version supports both 'episodes' and 'num_episodes'
        so it works with test.py.
        """
        
        # Use num_episodes if passed by test.py
        if num_episodes is not None:
            episodes = num_episodes

        scores = []
        durations = []  
        for ep in range(episodes):
            s, _ = env.reset()
            done = False
            total = 0
            steps = 0

            while not done:
                a = self.select_action(s, deterministic=True)
                s, r, term, trunc, _ = env.step(a)
                total += r
                steps += 1
                done = term or trunc

            scores.append(total)
            durations.append(steps)
            print(f"[TEST] Episode {ep}: Reward={total}")

        # Return detailed stats — test.py expects this dictionary:
        return {
            "episode_rewards": scores,
            "episode_durations": durations, 
            "mean_reward": float(np.mean(scores)),
            "min_reward": float(np.min(scores)),
            "max_reward": float(np.max(scores)),
            "std_reward": float(np.std(scores)),
            "mean_duration": float(np.mean(durations)), 
            "min_duration": int(np.min(durations)),      
            "max_duration": int(np.max(durations)),      
            "std_duration": float(np.std(durations)),
        }

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "alpha": self.alpha
        }, path)
        print(f"[SAC] Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.alpha = checkpoint.get("alpha", self.alpha)
        print(f"[SAC] Model loaded from {path}")

