#!/usr/bin/env python
"""Best model inference analysis"""
import sys, os
# Bypass src/__init__.py
sys.modules['src'] = type(sys)('src')
sys.modules['src'].__path__ = [os.path.join(os.path.dirname(__file__), 'src')]

import pandas as pd, numpy as np, torch
from src.environment import TradingEnvironment
from src.network import LSTMContinuousPolicyNetwork, LSTMValueNetwork

env_data_full = pd.read_csv('data/eth_20260228/environment.csv')
training_data_full = pd.read_csv('data/eth_20260228/training_scaled.csv').values

env_data = env_data_full.iloc[0:1440].copy()
training_data = training_data_full[0:1440]

env = TradingEnvironment(env_data, training_data,
    trade_cooldown=3, rebalance_threshold=0.03,
    action_smoothing=0.40, target_deadband=0.03,
    max_target_step_change=0.20,
    premium_alignment_coef=0.20, trade_quality_coef=0.15,
    price_position_coef=0.15, premium_lookback=60,
    action_scale=2.5)

device = torch.device('cuda')
policy_net = LSTMContinuousPolicyNetwork(env.num_features, hidden_dim=128, num_layers=2).to(device)
policy_net.load_state_dict(torch.load('models/policy_best.pt', map_location=device)['state_dict'])
policy_net.eval()

state = env.reset()
hidden = None
positions, raw_actions, alphas_list, betas_list = [], [], [], []

while True:
    s_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        alpha, beta, hidden = policy_net(s_tensor, hidden)
    action = (alpha / (alpha + beta)).item()
    raw_actions.append(action)
    alphas_list.append(alpha.item())
    betas_list.append(beta.item())
    state, reward, done, info = env.step(action)
    positions.append(info.get('position_ratio', 0))
    if done:
        break

positions = np.array(positions)
raw_actions = np.array(raw_actions)
alphas = np.array(alphas_list)
betas = np.array(betas_list)
conc = alphas + betas

print(f'Alpha: mean={alphas.mean():.2f}, std={alphas.std():.2f}')
print(f'Beta: mean={betas.mean():.2f}, std={betas.std():.2f}')
print(f'Concentration: mean={conc.mean():.2f}, std={conc.std():.2f}')
print(f'Raw action: mean={raw_actions.mean():.3f}, std={raw_actions.std():.3f}, min={raw_actions.min():.3f}, max={raw_actions.max():.3f}')
print(f'Position: mean={positions.mean():.3f}, std={positions.std():.3f}, min={positions.min():.3f}, max={positions.max():.3f}')
print(f'Position quartiles: Q1={np.percentile(positions,25):.3f}, Q2={np.percentile(positions,50):.3f}, Q3={np.percentile(positions,75):.3f}')
buys = env.num_buy
sells = env.num_sell
pv = env.portfolio_value
prf = (pv / 10_000_000 - 1) * 100
print(f'Trades: B={buys} S={sells}, Profit={prf:+.2f}%')

price_pctile = env.price_percentile[:len(positions)]
corr = np.corrcoef(positions, price_pctile)[0, 1]
print(f'Position-PricePercentile corr: {corr:.3f} (ideal: negative)')
