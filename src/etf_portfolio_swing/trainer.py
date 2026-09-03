import os
import json
import threading

from .environment import PortfolioTradingEnvironment
from .agent import PortfolioAgent
from ..etf_single_swing.train_monitor import generate_portfolio_dashboard


class PortfolioPPOTrainer:
    def __init__(self, env, agent, num_episodes=300, update_interval=128,
                 log_dir='logs/portfolio', output_dir='output/portfolio', viz_interval=10,
                 entropy_coef_start=0.10, entropy_coef_end=0.05, entropy_decay_episodes=400):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.update_interval = update_interval
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.viz_interval = viz_interval
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_decay_episodes = entropy_decay_episodes
        self._monitor_thread: threading.Thread = None
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def _update_monitor(self):
        """train_monitor.html을 백그라운드 스레드에서 갱신"""
        log_dir = self.log_dir

        def _run():
            try:
                generate_portfolio_dashboard(log_dir)
            except Exception:
                pass

        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._monitor_thread = threading.Thread(target=_run, daemon=True)
        self._monitor_thread.start()

    def train(self):
        best_return = -float('inf')
        log_path = os.path.join(self.log_dir, 'train_log.jsonl')

        for episode in range(self.num_episodes):
            ratio = min(episode / max(self.entropy_decay_episodes, 1), 1.0)
            self.agent.entropy_coef = self.entropy_coef_start + ratio * (self.entropy_coef_end - self.entropy_coef_start)

            obs = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_losses = {'policy_loss': [], 'value_loss': [], 'entropy': []}

            while not done:
                features = obs['features']
                prev_weights = obs['weights']
                weights, log_prob = self.agent.get_action(features, prev_weights)
                next_obs, reward, done, info = self.env.step(weights)
                self.agent.store_transition(
                    features, prev_weights, weights, log_prob, reward,
                    next_obs['features'], next_obs['weights'], done
                )

                if len(self.agent.buffer) >= self.update_interval:
                    loss_metrics = self.agent.update()
                    if loss_metrics:
                        for k in episode_losses:
                            episode_losses[k].append(loss_metrics.get(k, 0.0))

                obs = next_obs
                episode_reward += reward

            if self.agent.buffer:
                loss_metrics = self.agent.update()
                if loss_metrics:
                    for k in episode_losses:
                        episode_losses[k].append(loss_metrics.get(k, 0.0))

            metrics = self.env.get_metrics()
            metrics['episode'] = episode
            metrics['episode_reward'] = float(episode_reward)
            metrics['entropy_coef'] = round(self.agent.entropy_coef, 4)
            # 에피소드 평균 loss
            for k in episode_losses:
                vals = episode_losses[k]
                metrics[k] = round(sum(vals) / len(vals), 6) if vals else 0.0
            # numpy 타입 → Python float 변환
            metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}

            with open(log_path, 'a') as f:
                f.write(json.dumps(metrics) + '\n')

            cagr = metrics.get('cagr', 0.0)
            is_best = cagr > best_return
            if is_best:
                best_return = cagr
                metrics['best_episode'] = episode
                self.agent.save(os.path.join(self.output_dir, 'policy_best.pt'))

            if episode % 10 == 0:
                print(f"[Ep {episode:4d}] CAGR={cagr*100:.2f}% sharpe={metrics.get('sharpe', 0):.3f} mdd={metrics.get('mdd', 0)*100:.1f}% reward={episode_reward:.2f}")

            if episode % self.viz_interval == 0 or episode == self.num_episodes - 1 or is_best:
                self._update_monitor()

        self.agent.save(os.path.join(self.output_dir, 'policy_final.pt'))
