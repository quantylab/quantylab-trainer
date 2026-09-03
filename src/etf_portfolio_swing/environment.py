"""
포트폴리오 거래 환경 — EIIE 기반 멀티 ETF 포트폴리오 최적화
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import math


class PortfolioTradingEnvironment:
    """다중 ETF 포트폴리오 RL 환경

    관찰: {'features': [N, T, F], 'weights': [N+1]}
    행동: [N+1] 포트폴리오 비중 (softmax 정규화, 마지막=현금)
    """

    def __init__(
        self,
        asset_data: Dict[str, pd.DataFrame],
        asset_features: Dict[str, np.ndarray],
        lookback: int = 20,
        initial_balance: float = 10_000_000.0,
        trading_fee: float = 0.00015,
        reward_scale: float = 10.0,
        fee_penalty_scale: float = 5.0,
        drawdown_penalty_scale: float = 15.0,
        drawdown_penalty_threshold: float = 0.10,
        rolling_sharpe_window: int = 20,
        rolling_sharpe_scale: float = 2.0,
        reward_terminal_scale: float = 30.0,
        oos_start_date: Optional[str] = None,
    ):
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.reward_scale = reward_scale
        self.fee_penalty_scale = fee_penalty_scale
        self.drawdown_penalty_scale = drawdown_penalty_scale
        self.drawdown_penalty_threshold = drawdown_penalty_threshold
        self.rolling_sharpe_window = rolling_sharpe_window
        self.rolling_sharpe_scale = rolling_sharpe_scale
        self.reward_terminal_scale = reward_terminal_scale
        self.oos_start_date = pd.Timestamp(oos_start_date) if oos_start_date else None

        # Align to common date index
        self._asset_codes = list(asset_data.keys())
        self._align_data(asset_data, asset_features)

    def _align_data(
        self,
        asset_data: Dict[str, pd.DataFrame],
        asset_features: Dict[str, np.ndarray],
    ):
        """공통 날짜 인덱스로 정렬"""
        date_sets = []
        normalized = {}
        for code, df in asset_data.items():
            df = df.copy()
            # Normalize column names to lowercase
            df.columns = [c.lower() for c in df.columns]
            # Find date column
            date_col = None
            for col in df.columns:
                if 'date' in col or col == 'index':
                    date_col = col
                    break
            if date_col and date_col != 'date':
                df = df.rename(columns={date_col: 'date'})
            if 'date' in df.columns:
                # YYYYMMDD 정수 포맷 처리
                if pd.api.types.is_integer_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
                else:
                    df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            normalized[code] = df
            date_sets.append(set(df.index))

        common_dates = sorted(set.intersection(*date_sets))
        self._dates = pd.DatetimeIndex(common_dates)
        self._n_dates = len(self._dates)

        # Align price data
        self._close_prices: Dict[str, np.ndarray] = {}
        for code, df in normalized.items():
            df = df.loc[self._dates]
            close_col = 'close' if 'close' in df.columns else df.columns[3]
            self._close_prices[code] = df[close_col].values.astype(np.float32)

        # Align feature matrices
        self._features: Dict[str, np.ndarray] = {}
        for code in self._asset_codes:
            feat = asset_features[code]
            # feat shape: (n_dates_original, F) — already aligned by caller
            # We just take rows corresponding to common dates
            if feat.shape[0] == self._n_dates:
                self._features[code] = feat.astype(np.float32)
            else:
                # Attempt to slice by taking last n_dates rows that match
                self._features[code] = feat[-self._n_dates:].astype(np.float32)

        self._n_assets = len(self._asset_codes)
        self._n_features = next(iter(self._features.values())).shape[1]

    @property
    def n_assets(self) -> int:
        return self._n_assets

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def num_steps(self) -> int:
        return max(0, self._n_dates - self.lookback - 1)

    def reset(self) -> Dict:
        self._portfolio_value = self.initial_balance
        self._peak_value = self.initial_balance
        # Equal weight among assets + cash (last)
        self._weights = np.ones(self._n_assets + 1, dtype=np.float32) / (self._n_assets + 1)
        # OOS 모드: oos_start_date 이전 날짜는 스킵 (단, lookback 확보)
        if self.oos_start_date is not None:
            oos_indices = np.where(self._dates >= self.oos_start_date)[0]
            if len(oos_indices) == 0:
                raise ValueError(f"oos_start_date {self.oos_start_date} is beyond dataset range")
            start_idx = max(self.lookback, oos_indices[0])
        else:
            start_idx = self.lookback
        self._step_idx = start_idx
        self._log_returns: List[float] = []
        self._trade_log: List[dict] = []
        return self._get_obs()

    def _get_obs(self) -> Dict:
        # features: [N, T, F]
        features = np.stack([
            self._features[code][self._step_idx - self.lookback: self._step_idx]
            for code in self._asset_codes
        ], axis=0)
        return {
            'features': features,
            'weights': self._weights.copy(),
        }

    def step(self, weights: np.ndarray) -> Tuple[Dict, float, bool, dict]:
        weights = np.asarray(weights, dtype=np.float32)
        # Ensure valid weights (softmax normalize if needed)
        weights = np.clip(weights, 0, None)
        w_sum = weights.sum()
        if w_sum < 1e-8:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / w_sum

        # Asset returns: close-to-close
        asset_returns = np.zeros(self._n_assets, dtype=np.float32)
        for i, code in enumerate(self._asset_codes):
            prev_close = self._close_prices[code][self._step_idx - 1]
            curr_close = self._close_prices[code][self._step_idx]
            if prev_close > 0:
                asset_returns[i] = (curr_close - prev_close) / prev_close
            else:
                asset_returns[i] = 0.0

        # Portfolio return (cash has 0 return)
        portfolio_return = float(np.dot(weights[:-1], asset_returns))

        # Transaction cost
        transaction_cost = float(np.sum(np.abs(weights - self._weights))) * self.trading_fee * self.fee_penalty_scale

        # Log return
        log_return = math.log(1.0 + portfolio_return + 1e-9) - transaction_cost
        self._log_returns.append(log_return)

        # Base reward
        reward = log_return * self.reward_scale

        # Rolling Sharpe bonus
        if len(self._log_returns) >= self.rolling_sharpe_window:
            recent = self._log_returns[-self.rolling_sharpe_window:]
            mean_r = float(np.mean(recent))
            std_r = float(np.std(recent))
            if std_r > 1e-8:
                sharpe_bonus = self.rolling_sharpe_scale * (mean_r / std_r)
                reward += sharpe_bonus

        # Update portfolio value
        self._weights = weights.copy()
        self._portfolio_value *= (1.0 + portfolio_return - transaction_cost)

        # Drawdown penalty
        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value
        mdd = (self._peak_value - self._portfolio_value) / (self._peak_value + 1e-9)
        if mdd > self.drawdown_penalty_threshold:
            reward -= self.drawdown_penalty_scale * (mdd - self.drawdown_penalty_threshold)

        # Log this step
        self._trade_log.append({
            'date': self._dates[self._step_idx],
            'weights': weights.copy(),
            'asset_returns': asset_returns.copy(),
            'portfolio_return': portfolio_return,
            'portfolio_value': self._portfolio_value,
            'mdd': mdd,
        })

        self._step_idx += 1
        done = self._step_idx >= self._n_dates

        # Terminal reward: CAGR + Calmar + PLR
        if done:
            n_years = self.num_steps / 252.0
            if n_years > 0 and self._portfolio_value > 0:
                cagr = float((self._portfolio_value / self.initial_balance) ** (1.0 / n_years) - 1.0)
            else:
                cagr = -1.0
            cagr_reward = self.reward_terminal_scale * float(np.clip(cagr, -2.0, 5.0))

            # Calmar bonus
            if len(self._trade_log) > 0:
                peak = self.initial_balance
                max_dd = 0.0
                for t in self._trade_log:
                    v = t['portfolio_value']
                    if v > peak:
                        peak = v
                    dd = (peak - v) / (peak + 1e-9)
                    if dd > max_dd:
                        max_dd = dd
                if max_dd > 1e-6:
                    calmar = cagr / max_dd
                    calmar_bonus = float(np.clip(calmar * 0.3, -1.5, 1.5))
                else:
                    calmar_bonus = 0.0

                # PLR bonus
                daily_returns = [t['portfolio_return'] for t in self._trade_log]
                wins = [r for r in daily_returns if r > 0]
                losses = [r for r in daily_returns if r < 0]
                if wins and losses:
                    avg_win = float(np.mean(wins))
                    avg_loss = abs(float(np.mean(losses)))
                    plr_bonus = float(np.clip((avg_win / (avg_loss + 1e-9) - 1.0) * 0.5, -0.5, 0.5))
                else:
                    plr_bonus = 0.0
            else:
                calmar_bonus = 0.0
                plr_bonus = 0.0

            reward += cagr_reward + calmar_bonus + plr_bonus

        obs = self._get_obs() if not done else {
            'features': np.zeros((self._n_assets, self.lookback, self._n_features), dtype=np.float32),
            'weights': self._weights.copy(),
        }

        info = {
            'portfolio_value': self._portfolio_value,
            'portfolio_return': portfolio_return,
            'mdd': mdd,
            'step': self._step_idx,
        }

        return obs, float(reward), done, info

    def get_trade_log(self) -> List[dict]:
        return self._trade_log

    def get_metrics(self) -> dict:
        if not self._trade_log:
            return {}
        values = np.array([t['portfolio_value'] for t in self._trade_log], dtype=np.float32)
        total_return = float((values[-1] / self.initial_balance) - 1.0) if len(values) > 0 else 0.0
        n_years = len(values) / 252.0
        cagr = float((values[-1] / self.initial_balance) ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
        # MDD
        peak = self.initial_balance
        mdd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / (peak + 1e-9)
            if dd > mdd:
                mdd = dd
        # Sharpe
        log_rets = np.array(self._log_returns, dtype=np.float32)
        sharpe = 0.0
        if len(log_rets) > 1:
            std = float(np.std(log_rets))
            if std > 1e-8:
                sharpe = float(np.mean(log_rets) / std * math.sqrt(252))
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'mdd': mdd,
            'num_steps': len(self._trade_log),
        }
