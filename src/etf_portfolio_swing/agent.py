import os
import numpy as np
import torch
import torch.nn.functional as F


class PortfolioAgent:
    def __init__(self, policy_network, value_network, lr_policy=0.0002, lr_value=0.0005,
                 gamma=0.99, epsilon=0.2, entropy_coef=0.01, device='cpu'):
        self.device = device
        self.policy_net = policy_network.to(device)
        self.value_net = value_network.to(device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.optimizer_policy = torch.optim.Adam(policy_network.parameters(), lr=lr_policy, weight_decay=1e-5)
        self.optimizer_value = torch.optim.Adam(value_network.parameters(), lr=lr_value, weight_decay=1e-5)
        self.buffer = []

    def get_action(self, features, prev_weights):
        # features: [N, T, F] numpy, prev_weights: [N+1] numpy
        # returns: weights [N+1] numpy, log_prob scalar
        self.policy_net.eval()
        features_t = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        prev_w_t = torch.as_tensor(prev_weights, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net.get_logits(features_t, prev_w_t)
            concentration = torch.nn.functional.softplus(logits) + 1e-3
            dist = torch.distributions.Dirichlet(concentration)
            weights = dist.sample()
            # Re-normalize to ensure simplex constraint
            weights = weights.clamp(1e-6, 1.0)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            log_prob = dist.log_prob(weights).item()
        return weights.squeeze(0).cpu().numpy(), log_prob

    def get_value(self, features, prev_weights):
        self.value_net.eval()
        features_t = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        prev_w_t = torch.as_tensor(prev_weights, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            val = self.value_net(features_t, prev_w_t)
        return val.item()

    def store_transition(self, features, prev_weights, weights, log_prob, reward, next_features, next_prev_weights, done):
        self.buffer.append({
            'features': features, 'prev_weights': prev_weights, 'weights': weights,
            'log_prob': log_prob, 'reward': reward, 'next_features': next_features,
            'next_prev_weights': next_prev_weights, 'done': done,
        })

    def update(self, num_epochs=4, batch_size=32):
        if not self.buffer:
            return {}

        self.policy_net.train()
        self.value_net.train()

        # Convert to tensors
        features_arr = np.stack([t['features'] for t in self.buffer])
        prev_w_arr = np.stack([t['prev_weights'] for t in self.buffer])
        weights_arr = np.stack([t['weights'] for t in self.buffer])
        log_probs_arr = np.array([t['log_prob'] for t in self.buffer], dtype=np.float32)
        rewards_arr = np.array([t['reward'] for t in self.buffer], dtype=np.float32)
        next_features_arr = np.stack([t['next_features'] for t in self.buffer])
        next_prev_w_arr = np.stack([t['next_prev_weights'] for t in self.buffer])
        dones_arr = np.array([t['done'] for t in self.buffer], dtype=np.float32)

        features_t = torch.as_tensor(features_arr, dtype=torch.float32, device=self.device)
        prev_w_t = torch.as_tensor(prev_w_arr, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights_arr, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(log_probs_arr, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards_arr, dtype=torch.float32, device=self.device)
        next_features_t = torch.as_tensor(next_features_arr, dtype=torch.float32, device=self.device)
        next_prev_w_t = torch.as_tensor(next_prev_w_arr, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones_arr, dtype=torch.float32, device=self.device)

        # Compute GAE
        with torch.no_grad():
            values = self.value_net(features_t, prev_w_t).squeeze(-1)
            next_values = self.value_net(next_features_t, next_prev_w_t).squeeze(-1)

        advantages = torch.zeros_like(rewards_t)
        gae = 0.0
        for i in reversed(range(len(self.buffer))):
            delta = rewards_t[i] + self.gamma * next_values[i] * (1 - dones_t[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones_t[i]) * gae
            advantages[i] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(self.buffer)
        metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

        for _ in range(num_epochs):
            indices = np.random.permutation(T)
            for start in range(0, T, batch_size):
                batch_idx = indices[start:start + batch_size]
                if len(batch_idx) == 0:
                    continue

                b_features = features_t[batch_idx]
                b_prev_w = prev_w_t[batch_idx]
                b_weights = weights_t[batch_idx]
                b_old_lp = old_log_probs_t[batch_idx]
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]

                # Policy update
                log_prob, entropy = self.policy_net.get_log_prob_and_entropy(b_features, b_prev_w, b_weights)
                ratio = torch.exp(log_prob - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optimizer_policy.step()

                # Value update
                val_pred = self.value_net(b_features, b_prev_w).squeeze(-1)
                value_loss = F.mse_loss(val_pred, b_ret)

                self.optimizer_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.optimizer_value.step()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.mean().item()

        self.buffer.clear()
        return metrics

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({'policy': self.policy_net.state_dict(), 'value': self.value_net.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.value_net.load_state_dict(ckpt['value'])
