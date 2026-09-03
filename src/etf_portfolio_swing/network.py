"""
포트폴리오 신경망 — EIIE4QLT (GRU×2 + CrossAttn + Per-Asset Head)

핵심 개선 사항 (vs. 기존):
  1. Per-asset scoring head  — 자산별 독립 스코어 → 평균풀링 정보손실 없음
  2. 2-layer GRU             — temporal 표현력 향상
  3. LayerNorm + Residual    — CrossAttn 블록 학습 안정화
  4. prev_weight 자산별 주입 — 보유 비중을 각 자산 스코어에 반영
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _EIIE4QLT_Encoder(nn.Module):
    """공유 인코더: 2-layer GRU + LayerNorm → CrossAttn + Residual + LayerNorm

    입력:  features [B, N, T, F]
    출력:  context  [B, N, d_model]  — 자산별 컨텍스트 임베딩
    """

    def __init__(self, n_features: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.gru_norm = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N, T, F = features.shape

        # 2-layer GRU: 자산별 독립 시계열 인코딩 [B*N, T, F] → last hidden [B*N, d]
        gru_out, _ = self.gru(features.reshape(B * N, T, F))
        asset_embed = self.gru_norm(gru_out[:, -1, :]).reshape(B, N, self.d_model)

        # Cross-asset self-attention with residual + LN
        attn_out, _ = self.cross_attn(asset_embed, asset_embed, asset_embed)
        context = self.attn_norm(asset_embed + attn_out)  # [B, N, d_model]

        return context


class PortfolioPolicyNetwork(nn.Module):
    """EIIE4QLT 정책 신경망

    입력:
      features:     [B, N, T, F]
      prev_weights: [B, N+1]

    출력: [B, N+1] softmax 포트폴리오 비중
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model

        self.encoder = _EIIE4QLT_Encoder(n_features, d_model, n_heads, dropout)

        # Per-asset scorer: [context_i ‖ global_ctx ‖ prev_w_i] → scalar logit
        self.asset_scorer = nn.Sequential(
            nn.Linear(2 * d_model + 1, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Cash logit: 학습 가능한 바이어스 + 현재 현금 비중
        self.cash_bias = nn.Parameter(torch.zeros(1))

    def get_logits(self, features: torch.Tensor, prev_weights: torch.Tensor) -> torch.Tensor:
        """Raw logits [B, N+1] — cash first"""
        B, N = features.shape[0], features.shape[1]
        context = self.encoder(features)                          # [B, N, d]
        global_ctx = context.mean(dim=1, keepdim=True)           # [B, 1, d]
        global_ctx = global_ctx.expand(-1, N, -1)                # [B, N, d]

        prev_w_assets = prev_weights[:, 1:].unsqueeze(-1)        # [B, N, 1]
        asset_input = torch.cat([context, global_ctx, prev_w_assets], dim=-1)  # [B, N, 2d+1]
        asset_scores = self.asset_scorer(asset_input).squeeze(-1)              # [B, N]

        cash_score = self.cash_bias + prev_weights[:, :1]        # [B, 1]
        return torch.cat([cash_score, asset_scores], dim=-1)     # [B, N+1]

    def forward(self, features: torch.Tensor, prev_weights: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.get_logits(features, prev_weights), dim=-1)

    def get_log_prob_and_entropy(
        self,
        features: torch.Tensor,
        prev_weights: torch.Tensor,
        target_weights: torch.Tensor,
    ):
        """Dirichlet log-prob and entropy"""
        concentration = F.softplus(self.get_logits(features, prev_weights)) + 1e-3
        dist = torch.distributions.Dirichlet(concentration)
        clamped = target_weights.clamp(1e-6, 1.0)
        clamped = clamped / clamped.sum(dim=-1, keepdim=True)
        return dist.log_prob(clamped), dist.entropy()


class PortfolioValueNetwork(nn.Module):
    """EIIE4QLT 가치 신경망

    입력:
      features:     [B, N, T, F]
      prev_weights: [B, N+1]

    출력: [B, 1] scalar value
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = _EIIE4QLT_Encoder(n_features, d_model, n_heads, dropout)

        # Value head: [global_ctx ‖ prev_weights] → scalar
        self.value_head = nn.Sequential(
            nn.Linear(d_model + n_assets + 1, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, features: torch.Tensor, prev_weights: torch.Tensor) -> torch.Tensor:
        context = self.encoder(features)
        global_ctx = context.mean(dim=1)                         # [B, d_model]
        x = torch.cat([global_ctx, prev_weights], dim=-1)        # [B, d+N+1]
        return self.value_head(x)                                 # [B, 1]
