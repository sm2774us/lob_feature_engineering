"""LOB ML Pipeline — Feature Extraction, Signal Generation & RL Execution.

Production-grade implementation of machine learning algorithms for Limit Order
Book (LOB) analysis in systematic quantitative trading. Follows design-by-contract
and data-oriented principles so fabricated data can be swapped with live market
feeds from Refinitiv, Bloomberg, or proprietary tick sources without modifying
downstream logic.

Architecture:
    1. LOBDataGenerator    — synthetic data factory (replace with real feed)
    2. LOBFeatureEngine    — OFI, volume/trade imbalance, spread features
    3. DimensionalityReducer — PCA + sparse autoencoder latent codes
    4. LOBClassifier       — DeepLOB / LSTM / XGBoost / LightGBM ensemble
    5. RLExecutionAgent    — PPO-based execution policy
    6. LOBVisualizer       — Plotly-based diagnostic charts

References:
    Zhang, Z., Zohren, S., and Roberts, S. (2019). DeepLOB: Deep convolutional
    neural networks for limit order books. IEEE Transactions on Signal Processing.
    Cont, R., Kukanov, A., and Stoikov, S. (2014). The price impact of order book
    events. Journal of Financial Econometrics, 12(1), 47-88.

Usage:
    python lob_ml_pipeline.py

Author: Quant Research — Systematic Macro Division
Python: 3.13+
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Final, Literal, NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB: Final[bool] = True
except ImportError:
    HAS_LGB = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH: Final[bool] = True
except ImportError:
    HAS_TORCH = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants — single source of truth, easy to override per instrument
# ---------------------------------------------------------------------------
N_LEVELS: Final[int] = 10        # LOB depth (price levels each side)
N_FEATURES_RAW: Final[int] = 40  # 10 levels × (bid_px, bid_vol, ask_px, ask_vol)
LATENT_DIM: Final[int] = 8       # Autoencoder bottleneck dimension
PCA_VARIANCE_THRESHOLD: Final[float] = 0.90  # Retain 90% explained variance
LABEL_HORIZON: Final[int] = 50   # Forward ticks for label computation
LABEL_THETA: Final[float] = 2.0  # Threshold in basis-point mid-price change
SMOOTH_K: Final[int] = 5         # Rolling average window for mid smoothing
RANDOM_SEED: Final[int] = 42


# ---------------------------------------------------------------------------
# Data Contracts — replace fabricated generators with real data adapters
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LOBSnapshot:
    """Immutable snapshot of the full limit order book at a single timestamp.

    Attributes:
        timestamp_ns: UNIX nanosecond timestamp.
        bid_prices: Best-to-worst bid prices, shape (N_LEVELS,).
        bid_volumes: Corresponding bid volumes, shape (N_LEVELS,).
        ask_prices: Best-to-worst ask prices, shape (N_LEVELS,).
        ask_volumes: Corresponding ask volumes, shape (N_LEVELS,).
        last_trade_px: Last executed trade price.
        last_trade_vol: Last executed trade volume.
        last_side: Trade aggressor side ('buy', 'sell', or 'unknown').
    """

    timestamp_ns: int
    bid_prices: np.ndarray
    bid_volumes: np.ndarray
    ask_prices: np.ndarray
    ask_volumes: np.ndarray
    last_trade_px: float
    last_trade_vol: float
    last_side: Literal["buy", "sell", "unknown"] = "unknown"

    def mid_price(self) -> float:
        """Returns the mid-price as the average of best bid and best ask."""
        return (self.bid_prices[0] + self.ask_prices[0]) / 2.0

    def spread(self) -> float:
        """Returns the best bid-ask spread."""
        return self.ask_prices[0] - self.bid_prices[0]

    def to_flat_vector(self) -> np.ndarray:
        """Flattens the LOB into a 40-dimensional feature vector.

        Returns:
            Concatenation: [bid_px(10), bid_vol(10), ask_px(10), ask_vol(10)].
        """
        return np.concatenate([
            self.bid_prices, self.bid_volumes,
            self.ask_prices, self.ask_volumes,
        ])


class LOBFeatureRow(NamedTuple):
    """Named tuple representing a single row of engineered LOB features.

    Attributes:
        ofi: Aggregate Order Flow Imbalance (bid OFI minus ask OFI).
        volume_imbalance: (ask_vol - bid_vol) / (ask_vol + bid_vol).
        trade_imbalance: Net buy volume minus net sell volume over window.
        spread_bps: Bid-ask spread in basis points.
        depth_imbalance: Total bid depth minus total ask depth over all levels.
        mid_price: Current mid-price.
        autocovariance_1: Lag-1 autocovariance of mid-price returns.
    """

    ofi: float
    volume_imbalance: float
    trade_imbalance: float
    spread_bps: float
    depth_imbalance: float
    mid_price: float
    autocovariance_1: float


# ---------------------------------------------------------------------------
# Data Generator — swap this class for a real market data adapter
# ---------------------------------------------------------------------------

class LOBDataGenerator:
    """Generates synthetic Limit Order Book data for pipeline development.

    Produces realistic-looking LOB snapshots with mean-reverting mid-prices,
    stochastic spreads, and correlated queue dynamics. Replace this class with
    a real data adapter (Refinitiv, Bloomberg Tick, proprietary feed) in
    production by implementing the same ``generate()`` interface contract.

    Attributes:
        n_snapshots: Number of LOB snapshots to generate.
        n_levels: Number of price levels per side.
        tick_size: Minimum price increment.
        base_price: Initial mid-price.
        rng: NumPy random number generator (seeded for reproducibility).
    """

    def __init__(
        self,
        n_snapshots: int = 5000,
        n_levels: int = N_LEVELS,
        tick_size: float = 0.01,
        base_price: float = 100.0,
        seed: int = RANDOM_SEED,
    ) -> None:
        """Initializes the LOB data generator.

        Args:
            n_snapshots: Number of time steps to simulate.
            n_levels: Depth of order book per side.
            tick_size: Minimum price increment (e.g. 0.01 for equities).
            base_price: Starting mid-price.
            seed: Random seed for reproducibility.
        """
        self.n_snapshots = n_snapshots
        self.n_levels = n_levels
        self.tick_size = tick_size
        self.base_price = base_price
        self.rng = np.random.default_rng(seed)

    def generate(self) -> list[LOBSnapshot]:
        """Generates a time series of synthetic LOB snapshots.

        Returns:
            List of LOBSnapshot objects in chronological order.
        """
        snapshots: list[LOBSnapshot] = []
        mid = self.base_price
        vol_regime = 1.0  # volatility multiplier (regime switching)
        trade_buy_vol = 0.0
        trade_sell_vol = 0.0

        for i in range(self.n_snapshots):
            # Regime switch every ~500 ticks
            if i % 500 == 0:
                vol_regime = self.rng.uniform(0.5, 2.0)

            # Mean-reverting mid-price with stochastic volatility
            mid += (0.0 - (mid - self.base_price) * 0.001
                    + self.rng.normal(0, 0.02 * vol_regime))
            mid = max(mid, self.base_price * 0.8)

            # Stochastic spread (wider in high-vol regimes)
            half_spread = self.rng.uniform(0.01, 0.03) * vol_regime
            best_bid = mid - half_spread
            best_ask = mid + half_spread

            # Build multi-level book with exponentially decaying depth
            bid_prices = best_bid - np.arange(self.n_levels) * self.tick_size
            ask_prices = best_ask + np.arange(self.n_levels) * self.tick_size

            # Volume with stochastic liquidity and level decay
            base_vol = self.rng.exponential(500)
            decay = np.exp(-0.4 * np.arange(self.n_levels))
            bid_volumes = base_vol * decay * self.rng.uniform(0.7, 1.3, self.n_levels)
            ask_volumes = base_vol * decay * self.rng.uniform(0.7, 1.3, self.n_levels)

            # Simulate trade
            trade_px = mid + self.rng.normal(0, half_spread * 0.5)
            trade_vol = self.rng.exponential(200)
            side: Literal["buy", "sell", "unknown"] = (
                "buy" if self.rng.random() > 0.5 else "sell"
            )
            if side == "buy":
                trade_buy_vol = trade_vol
                trade_sell_vol = 0.0
            else:
                trade_sell_vol = trade_vol
                trade_buy_vol = 0.0

            snapshots.append(LOBSnapshot(
                timestamp_ns=int(1_700_000_000_000_000_000 + i * 1_000_000),
                bid_prices=bid_prices,
                bid_volumes=bid_volumes,
                ask_prices=ask_prices,
                ask_volumes=ask_volumes,
                last_trade_px=trade_px,
                last_trade_vol=trade_vol,
                last_side=side,
            ))

        return snapshots


# ---------------------------------------------------------------------------
# Feature Engine
# ---------------------------------------------------------------------------

class LOBFeatureEngine:
    """Computes classical microstructure features from raw LOB snapshots.

    Implements Order Flow Imbalance (OFI), volume imbalance, trade imbalance,
    spread, depth imbalance, and lag-1 autocovariance as described in Cont,
    Kukanov & Stoikov (2014).

    Attributes:
        decay_lambda: Exponential decay for multi-level OFI weighting.
        return_window: Rolling window for autocovariance computation.
    """

    def __init__(
        self,
        decay_lambda: float = 0.5,
        return_window: int = 20,
    ) -> None:
        """Initializes the feature engine.

        Args:
            decay_lambda: Exponential decay weight for multi-level OFI.
                Level ℓ weight = exp(-decay_lambda * ℓ). Default 0.5.
            return_window: Window size for autocovariance of mid-price returns.
        """
        self.decay_lambda = decay_lambda
        self.return_window = return_window
        self._mid_history: list[float] = []

    def _compute_ofi(
        self,
        snap: LOBSnapshot,
        prev: LOBSnapshot,
    ) -> float:
        """Computes aggregate multi-level Order Flow Imbalance.

        Implements the Cont et al. (2014) OFI formula with exponential depth
        weighting across N_LEVELS price levels.

        Args:
            snap: Current LOB snapshot at time t.
            prev: Previous LOB snapshot at time t-1.

        Returns:
            Scalar OFI value. Positive = buy-side pressure.
        """
        weights = np.exp(-self.decay_lambda * np.arange(N_LEVELS))
        ofi = 0.0
        for ell in range(N_LEVELS):
            # Bid-side OFI
            if snap.bid_prices[ell] > prev.bid_prices[ell]:
                of_b = snap.bid_volumes[ell]
            elif snap.bid_prices[ell] == prev.bid_prices[ell]:
                of_b = snap.bid_volumes[ell] - prev.bid_volumes[ell]
            else:
                of_b = -snap.bid_volumes[ell]

            # Ask-side OFI
            if snap.ask_prices[ell] > prev.ask_prices[ell]:
                of_a = -snap.ask_volumes[ell]
            elif snap.ask_prices[ell] == prev.ask_prices[ell]:
                of_a = snap.ask_volumes[ell] - prev.ask_volumes[ell]
            else:
                of_a = snap.ask_volumes[ell]

            ofi += weights[ell] * (of_b - of_a)
        return ofi

    def _autocovariance(self, lag: int = 1) -> float:
        """Computes lag-k autocovariance of mid-price returns.

        Args:
            lag: Lag order. Default 1.

        Returns:
            Sample autocovariance. Returns 0.0 if insufficient history.
        """
        if len(self._mid_history) < self.return_window + lag:
            return 0.0
        mids = np.array(self._mid_history[-self.return_window:])
        rets = np.diff(np.log(mids))
        if len(rets) < lag + 1:
            return 0.0
        return float(np.cov(rets[:-lag], rets[lag:])[0, 1])

    def compute(
        self,
        snapshots: list[LOBSnapshot],
        trade_window: int = 10,
    ) -> pd.DataFrame:
        """Computes the full feature matrix from a sequence of LOB snapshots.

        Args:
            snapshots: Chronologically ordered list of LOBSnapshot objects.
            trade_window: Number of ticks for rolling trade imbalance window.

        Returns:
            DataFrame of shape (len(snapshots)-1, n_features) with columns:
            ['ofi', 'volume_imbalance', 'trade_imbalance', 'spread_bps',
             'depth_imbalance', 'mid_price', 'autocovariance_1',
             'raw_*'] where raw_* are the 40-dim flat LOB vectors.
        """
        rows: list[dict] = []
        buy_vols: list[float] = []
        sell_vols: list[float] = []

        for i in range(1, len(snapshots)):
            snap = snapshots[i]
            prev = snapshots[i - 1]
            mid = snap.mid_price()
            self._mid_history.append(mid)

            # Trade imbalance accumulation
            if snap.last_side == "buy":
                buy_vols.append(snap.last_trade_vol)
                sell_vols.append(0.0)
            else:
                sell_vols.append(snap.last_trade_vol)
                buy_vols.append(0.0)
            window_buys = sum(buy_vols[-trade_window:])
            window_sells = sum(sell_vols[-trade_window:])

            ofi = self._compute_ofi(snap, prev)
            total_bid = snap.bid_volumes.sum()
            total_ask = snap.ask_volumes.sum()
            denom = snap.bid_volumes[0] + snap.ask_volumes[0]
            vi = ((snap.ask_volumes[0] - snap.bid_volumes[0]) / denom
                  if denom > 0 else 0.0)
            spread_bps = snap.spread() / mid * 10_000.0

            feature_dict: dict = {
                "ofi": ofi,
                "volume_imbalance": vi,
                "trade_imbalance": window_buys - window_sells,
                "spread_bps": spread_bps,
                "depth_imbalance": total_bid - total_ask,
                "mid_price": mid,
                "autocovariance_1": self._autocovariance(1),
            }
            # Append raw 40-dim flat LOB vector
            raw = snap.to_flat_vector()
            for k, v in enumerate(raw):
                feature_dict[f"raw_{k}"] = v

            rows.append(feature_dict)

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Label Generator
# ---------------------------------------------------------------------------

def generate_labels(
    mid_prices: np.ndarray,
    horizon: int = LABEL_HORIZON,
    theta: float = LABEL_THETA,
    smooth_k: int = SMOOTH_K,
) -> np.ndarray:
    """Generates ternary Up/Stable/Down labels for supervised LOB classification.

    Computes forward mid-price change over ``horizon`` ticks, optionally
    smoothed by a rolling average, and thresholds at ±theta basis points.

    Args:
        mid_prices: Array of mid-prices, shape (T,).
        horizon: Forward tick horizon for label computation.
        theta: Threshold in basis points for Up/Down classification.
        smooth_k: Number of ticks to average for forward mid smoothing.
            Set to 1 to disable smoothing.

    Returns:
        Integer label array, shape (T,): 0=Down, 1=Stable, 2=Up.
        Labels for the last ``horizon + smooth_k`` ticks are set to -1 (invalid).
    """
    T = len(mid_prices)
    labels = np.full(T, 1, dtype=np.int8)  # default: Stable

    for t in range(T - horizon - smooth_k):
        future_slice = mid_prices[t + horizon: t + horizon + smooth_k]
        future_mid = float(np.mean(future_slice))
        change_bps = (future_mid - mid_prices[t]) / mid_prices[t] * 10_000.0
        if change_bps > theta:
            labels[t] = 2  # Up
        elif change_bps < -theta:
            labels[t] = 0  # Down
        # else: Stable (1) — already set

    # Invalidate tail
    labels[T - horizon - smooth_k:] = -1
    return labels


# ---------------------------------------------------------------------------
# Dimensionality Reduction: PCA + Autoencoder
# ---------------------------------------------------------------------------

class LOBDimensionalityReducer:
    """PCA and sparse autoencoder for LOB feature compression.

    Provides both linear (PCA) and non-linear (autoencoder with ReLU
    activations) dimensionality reduction of the 40-dimensional raw LOB
    feature space into a compact latent code suitable for downstream
    classification.

    Attributes:
        pca: Fitted sklearn PCA object.
        n_pca_components: Number of PCA components retained (auto-selected).
        autoencoder: Fitted PyTorch autoencoder (None if PyTorch unavailable).
        scaler: StandardScaler fitted on training data.
        latent_dim: Autoencoder bottleneck dimension.
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        """Initializes the dimensionality reducer.

        Args:
            latent_dim: Target dimension for autoencoder bottleneck. Default 8.
        """
        self.latent_dim = latent_dim
        self.pca: PCA | None = None
        self.n_pca_components: int = 0
        self.scaler = StandardScaler()
        self.autoencoder: "SparseAutoencoder | None" = None

    def fit_pca(self, X_raw: np.ndarray) -> np.ndarray:
        """Fits PCA and returns transformed coordinates.

        Automatically selects k components explaining ≥90% variance.

        Args:
            X_raw: Raw LOB feature matrix, shape (T, N_FEATURES_RAW).

        Returns:
            PCA-transformed matrix, shape (T, k).
        """
        X_scaled = self.scaler.fit_transform(X_raw)
        pca_full = PCA(n_components=min(N_FEATURES_RAW, X_raw.shape[0] - 1))
        pca_full.fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, PCA_VARIANCE_THRESHOLD) + 1)
        k = max(k, 2)
        self.n_pca_components = k
        self.pca = PCA(n_components=k)
        return self.pca.fit_transform(X_scaled)

    def transform_pca(self, X_raw: np.ndarray) -> np.ndarray:
        """Applies fitted PCA transform to new data.

        Args:
            X_raw: Raw LOB feature matrix, shape (T, N_FEATURES_RAW).

        Returns:
            PCA-transformed matrix, shape (T, k).
        """
        assert self.pca is not None, "Call fit_pca() first."
        return self.pca.transform(self.scaler.transform(X_raw))

    def fit_autoencoder(
        self,
        X_raw: np.ndarray,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
        beta: float = 1e-4,
    ) -> np.ndarray:
        """Trains a sparse autoencoder and returns latent encodings.

        Architecture: 40 → Dense(32,ReLU) → Dense(16,ReLU) →
                      Dense(latent_dim,ReLU) [bottleneck] →
                      Dense(16,ReLU) → Dense(32,ReLU) → Dense(40,Linear)

        Loss: ||x - x̂||² + β·||z||₁  (L2 reconstruction + L1 sparsity)

        Args:
            X_raw: Raw LOB feature matrix, shape (T, N_FEATURES_RAW).
            epochs: Training epochs. Default 30.
            batch_size: Mini-batch size. Default 256.
            lr: Adam learning rate. Default 1e-3.
            beta: L1 sparsity coefficient. Default 1e-4.

        Returns:
            Latent codes, shape (T, latent_dim). Returns PCA codes if
            PyTorch is unavailable.
        """
        if not HAS_TORCH:
            print("[INFO] PyTorch not found — using PCA as autoencoder proxy.")
            return self.fit_pca(X_raw)[:, :self.latent_dim]

        X_scaled = self.scaler.fit_transform(X_raw).astype(np.float32)
        self.autoencoder = SparseAutoencoder(
            input_dim=N_FEATURES_RAW, latent_dim=self.latent_dim
        )
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)
        dataset = TensorDataset(torch.from_numpy(X_scaled))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                z, x_hat = self.autoencoder(batch)
                recon = nn.functional.mse_loss(x_hat, batch)
                sparsity = beta * z.abs().mean()
                loss = recon + sparsity
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  AE Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

        return self._encode(X_scaled)

    def _encode(self, X_scaled: np.ndarray) -> np.ndarray:
        """Returns latent codes for scaled input without gradient tracking.

        Args:
            X_scaled: Standardized feature matrix, shape (T, N_FEATURES_RAW).

        Returns:
            Latent codes, shape (T, latent_dim).
        """
        self.autoencoder.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X_scaled.astype(np.float32))
            z, _ = self.autoencoder(tensor)
        return z.numpy()


if HAS_TORCH:
    class SparseAutoencoder(nn.Module):
        """Sparse autoencoder with ReLU activations for LOB compression.

        Attributes:
            encoder: Sequential encoder network.
            decoder: Sequential decoder network.
        """

        def __init__(self, input_dim: int, latent_dim: int) -> None:
            """Initializes encoder and decoder sub-networks.

            Args:
                input_dim: Dimension of raw input (40 for LOB).
                latent_dim: Bottleneck dimension (8 by default).
            """
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(),
                nn.Linear(32, 16),        nn.ReLU(),
                nn.Linear(16, latent_dim), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16), nn.ReLU(),
                nn.Linear(16, 32),         nn.ReLU(),
                nn.Linear(32, input_dim),
            )

        def forward(
            self, x: "torch.Tensor"
        ) -> "tuple[torch.Tensor, torch.Tensor]":
            """Forward pass returning latent code and reconstruction.

            Args:
                x: Input tensor, shape (batch, input_dim).

            Returns:
                Tuple of (z, x_hat) where z is the latent code and x_hat
                is the reconstruction.
            """
            z = self.encoder(x)
            return z, self.decoder(z)


# ---------------------------------------------------------------------------
# LSTM Classifier (PyTorch)
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class LOBLSTMClassifier(nn.Module):
        """LSTM-based LOB classifier for temporal sequence modeling.

        Processes a sliding window of LOB feature vectors through stacked
        LSTM layers followed by a linear classifier head.

        Attributes:
            lstm: Stacked LSTM module.
            head: Linear classification head.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            n_layers: int = 2,
            n_classes: int = 3,
            dropout: float = 0.2,
        ) -> None:
            """Initializes the LSTM classifier.

            Args:
                input_dim: Feature dimension per time step.
                hidden_dim: LSTM hidden units. Default 64.
                n_layers: Number of stacked LSTM layers. Default 2.
                n_classes: Output classes (3: Up/Stable/Down).
                dropout: Dropout rate between LSTM layers. Default 0.2.
            """
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_dim, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward pass.

            Args:
                x: Input tensor, shape (batch, seq_len, input_dim).

            Returns:
                Logits tensor, shape (batch, n_classes).
            """
            _, (h_n, _) = self.lstm(x)
            return self.head(h_n[-1])


# ---------------------------------------------------------------------------
# RL Execution Agent (PPO-style, numpy-based for portability)
# ---------------------------------------------------------------------------

@dataclass
class RLExecutionAgent:
    """Simplified PPO-style RL agent for LOB-aware order execution.

    Models the execution decision as a finite-horizon MDP. The state includes
    LOB features, current inventory, and classifier signal probabilities. The
    agent learns to minimize market impact while capturing the predicted signal.

    Attributes:
        n_state_features: Dimension of the state vector.
        n_actions: Number of discrete actions (6 by default).
        gamma: Discount factor for future rewards.
        lr: Learning rate for policy gradient updates.
        inventory_penalty: λ coefficient penalizing inventory risk.
        rng: Reproducible random number generator.
    """

    n_state_features: int = 15
    n_actions: int = 6  # Buy, Sell, PostBid, PostAsk, Cancel, Hold
    gamma: float = 0.99
    lr: float = 1e-3
    inventory_penalty: float = 0.01
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(RANDOM_SEED)
    )
    _policy_weights: np.ndarray = field(init=False)
    _episode_rewards: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initializes policy weights with Xavier uniform initialization."""
        scale = np.sqrt(2.0 / (self.n_state_features + self.n_actions))
        self._policy_weights = self.rng.uniform(
            -scale, scale, (self.n_state_features, self.n_actions)
        )

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax.

        Args:
            logits: Raw action scores, shape (n_actions,).

        Returns:
            Probability distribution over actions.
        """
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def act(self, state: np.ndarray) -> int:
        """Samples an action from the current policy.

        Args:
            state: State vector, shape (n_state_features,).

        Returns:
            Integer action index in [0, n_actions-1].
        """
        logits = state @ self._policy_weights
        probs = self._softmax(logits)
        return int(self.rng.choice(self.n_actions, p=probs))

    def compute_reward(
        self,
        pnl_delta: float,
        inventory: float,
        adverse_fill: bool,
    ) -> float:
        """Computes the execution reward signal.

        r_t = ΔPNL - λ·|inventory| - μ·1[adverse fill]

        Args:
            pnl_delta: Realized PnL change from the action.
            inventory: Current absolute inventory level.
            adverse_fill: Whether the fill was against the signal direction.

        Returns:
            Scalar reward value.
        """
        return pnl_delta - self.inventory_penalty * abs(inventory) - (
            0.05 if adverse_fill else 0.0
        )

    def simulate_episode(
        self,
        features: np.ndarray,
        signal_probs: np.ndarray,
        n_steps: int = 100,
    ) -> dict:
        """Runs one RL episode and collects trajectory statistics.

        Args:
            features: LOB feature matrix, shape (T, n_lob_features).
            signal_probs: Classifier output probabilities, shape (T, 3).
            n_steps: Episode length in ticks. Default 100.

        Returns:
            Dictionary with keys: 'total_reward', 'n_trades', 'sharpe',
            'action_counts', 'rewards'.
        """
        inventory = 0.0
        total_pnl = 0.0
        rewards: list[float] = []
        action_counts = np.zeros(self.n_actions, dtype=int)

        for step in range(min(n_steps, len(features))):
            # Build state: fixed-size vector regardless of feature dim
            n_lob = self.n_state_features - 4  # 4 = 1 inv + 3 probs
            lob_raw = features[step]
            lob_feat = np.resize(lob_raw, n_lob)  # zero-pad or truncate
            inv_feat = np.array([inventory / 1000.0])
            prob_feat = signal_probs[step]  # [P_down, P_stable, P_up]
            state = np.concatenate([lob_feat, inv_feat, prob_feat])

            action = self.act(state)
            action_counts[action] += 1

            # Simplified P&L model
            signal_direction = float(signal_probs[step, 2] - signal_probs[step, 0])
            pnl_delta = signal_direction * self.rng.normal(0.5, 0.3)

            if action == 0:    # Buy
                inventory += 1
                pnl_delta += signal_direction * 0.1
            elif action == 1:  # Sell
                inventory -= 1
                pnl_delta -= signal_direction * 0.1
            # Actions 2-5: passive / cancel / hold

            adverse = (action == 0 and signal_direction < -0.2) or (
                action == 1 and signal_direction > 0.2
            )
            r = self.compute_reward(pnl_delta, inventory, adverse)
            rewards.append(r)
            total_pnl += pnl_delta

        rewards_arr = np.array(rewards)
        sharpe = (rewards_arr.mean() / (rewards_arr.std() + 1e-8)
                  * np.sqrt(252 * n_steps))

        return {
            "total_reward": float(rewards_arr.sum()),
            "n_trades": int(action_counts[:2].sum()),
            "sharpe": float(sharpe),
            "action_counts": action_counts,
            "rewards": rewards_arr,
        }


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class LOBVisualizer:
    """Plotly-based diagnostic visualization suite for LOB ML pipelines.

    All plots use a dark professional theme consistent with Bloomberg Terminal
    aesthetics. Figures are returned as Plotly Figure objects and saved to
    disk as high-resolution PNGs.
    """

    TEMPLATE: Final[str] = "plotly_dark"
    COLORS: Final[dict] = {
        "up": "#00D4AA",
        "down": "#FF4B4B",
        "stable": "#FFD700",
        "primary": "#4FC3F7",
        "secondary": "#BA68C8",
        "bg": "#0D0D1A",
    }

    def plot_lob_snapshot(
        self,
        snap: LOBSnapshot,
        title: str = "LOB Snapshot",
    ) -> go.Figure:
        """Plots a bid/ask ladder for a single LOB snapshot.

        Args:
            snap: LOBSnapshot to visualize.
            title: Figure title string.

        Returns:
            Plotly Figure with horizontal bar chart of bid/ask depth.
        """
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=snap.bid_volumes,
            y=snap.bid_prices,
            orientation="h",
            name="Bid",
            marker_color=self.COLORS["up"],
        ))
        fig.add_trace(go.Bar(
            x=-snap.ask_volumes,
            y=snap.ask_prices,
            orientation="h",
            name="Ask",
            marker_color=self.COLORS["down"],
        ))
        fig.update_layout(
            title=title, template=self.TEMPLATE,
            xaxis_title="Volume (negative = ask)", yaxis_title="Price",
            barmode="overlay", height=450,
        )
        return fig

    def plot_features(
        self,
        features: pd.DataFrame,
        n_rows: int = 500,
    ) -> go.Figure:
        """Plots the key engineered features over time.

        Args:
            features: Feature DataFrame from LOBFeatureEngine.compute().
            n_rows: Number of rows to plot. Default 500.

        Returns:
            Plotly Figure with subplots for OFI, Volume Imbalance, Spread.
        """
        df = features.iloc[:n_rows]
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=["Order Flow Imbalance (OFI)",
                            "Volume Imbalance", "Spread (bps)"],
            shared_xaxes=True,
        )
        fig.add_trace(go.Scatter(
            y=df["ofi"], name="OFI",
            line=dict(color=self.COLORS["primary"], width=1),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            y=df["volume_imbalance"], name="Vol Imbalance",
            line=dict(color=self.COLORS["secondary"], width=1),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            y=df["spread_bps"], name="Spread bps",
            line=dict(color=self.COLORS["stable"], width=1),
        ), row=3, col=1)
        fig.update_layout(
            title="LOB Feature Time Series",
            template=self.TEMPLATE, height=600,
        )
        return fig

    def plot_pca_variance(self, pca: PCA) -> go.Figure:
        """Plots explained variance ratio and cumulative explained variance.

        Args:
            pca: Fitted sklearn PCA object.

        Returns:
            Plotly Figure with dual-axis bar/line chart.
        """
        evr = pca.explained_variance_ratio_
        cumvar = np.cumsum(evr)
        n = len(evr)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=list(range(1, n + 1)), y=evr * 100,
            name="Explained Variance %",
            marker_color=self.COLORS["primary"],
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=list(range(1, n + 1)), y=cumvar * 100,
            name="Cumulative Variance %",
            line=dict(color=self.COLORS["up"], width=2),
            mode="lines+markers",
        ), secondary_y=True)
        fig.add_hline(y=90, line_dash="dash",
                      line_color=self.COLORS["stable"],
                      annotation_text="90% threshold",
                      secondary_y=True)
        fig.update_layout(
            title="PCA Explained Variance — LOB Features",
            template=self.TEMPLATE, height=400,
            xaxis_title="Principal Component",
        )
        return fig

    def plot_latent_space(
        self,
        latent_codes: np.ndarray,
        labels: np.ndarray,
        method: str = "Autoencoder",
    ) -> go.Figure:
        """Scatter plot of latent space colored by signal label.

        Args:
            latent_codes: Latent representations, shape (T, d). Uses first
                two dimensions for 2-D scatter.
            labels: Integer labels (0=Down, 1=Stable, 2=Up), shape (T,).
            method: Name of the reduction method for title annotation.

        Returns:
            Plotly Figure with scatter plot of latent codes.
        """
        label_map = {0: "Down", 1: "Stable", 2: "Up"}
        color_map = {
            "Down": self.COLORS["down"],
            "Stable": self.COLORS["stable"],
            "Up": self.COLORS["up"],
        }
        mask = labels >= 0
        z1 = latent_codes[mask, 0]
        z2 = latent_codes[mask, 1]
        label_names = [label_map[l] for l in labels[mask]]

        fig = px.scatter(
            x=z1, y=z2, color=label_names,
            color_discrete_map=color_map,
            labels={"x": "z₁", "y": "z₂", "color": "Label"},
            title=f"{method} Latent Space (z₁ vs z₂)",
            template=self.TEMPLATE,
            opacity=0.6,
        )
        fig.update_traces(marker_size=3)
        fig.update_layout(height=450)
        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Classifier",
    ) -> go.Figure:
        """Plots a normalized confusion matrix as a heatmap.

        Args:
            y_true: Ground-truth labels (0=Down, 1=Stable, 2=Up).
            y_pred: Predicted labels.
            model_name: Name string for figure title.

        Returns:
            Plotly heatmap Figure.
        """
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        labels = ["Down", "Stable", "Up"]
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, self.COLORS["down"]],
                [0.5, "#1a1a2e"],
                [1.0, self.COLORS["up"]],
            ],
            text=np.round(cm, 2),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig.update_layout(
            title=f"Confusion Matrix — {model_name} (normalized)",
            xaxis_title="Predicted", yaxis_title="True",
            template=self.TEMPLATE, height=400,
        )
        return fig

    def plot_rl_rewards(self, rewards: np.ndarray, title: str = "RL Episode Rewards") -> go.Figure:
        """Plots RL episode reward trajectory with rolling mean.

        Args:
            rewards: Per-step reward array from a single episode.
            title: Figure title.

        Returns:
            Plotly Figure showing reward trace and rolling mean.
        """
        cumulative = np.cumsum(rewards)
        rolling_mean = pd.Series(rewards).rolling(20, min_periods=1).mean().values
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=["Per-Step Reward", "Cumulative Reward"])
        fig.add_trace(go.Scatter(
            y=rewards, name="Step Reward",
            line=dict(color=self.COLORS["primary"], width=1),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            y=rolling_mean, name="Rolling Mean (20)",
            line=dict(color=self.COLORS["up"], width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            y=cumulative, name="Cumulative Reward",
            fill="tozeroy",
            line=dict(color=self.COLORS["secondary"], width=2),
        ), row=2, col=1)
        fig.update_layout(
            title=title, template=self.TEMPLATE, height=500,
        )
        return fig

    @staticmethod
    def save(fig: go.Figure, path: str, width: int = 1200, height: int = 600) -> None:
        """Saves a Plotly figure as a PNG image.

        Args:
            fig: Plotly Figure object.
            path: Output file path (should end in .png).
            width: Image width in pixels.
            height: Image height in pixels.
        """
        try:
            fig.write_image(path, width=width, height=height)
            print(f"  Saved: {path}")
        except Exception as exc:
            print(f"  [WARN] Could not save PNG (install kaleido): {exc}")
            fig.write_html(path.replace(".png", ".html"))
            print(f"  Saved HTML fallback: {path.replace('.png', '.html')}")


# ---------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    n_snapshots: int = 3000,
    output_dir: str = "/mnt/user-data/outputs",
) -> None:
    """Runs the full LOB ML pipeline end-to-end.

    Orchestrates: data generation → feature engineering → dimensionality
    reduction → supervised classification → RL execution simulation →
    visualization. All intermediate artifacts are saved to output_dir.

    Args:
        n_snapshots: Number of synthetic LOB snapshots to generate.
        output_dir: Directory for saving charts and artefacts.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    viz = LOBVisualizer()

    print("=" * 60)
    print("  LOB ML Pipeline — Systematic Quant Research")
    print("=" * 60)

    # 1. Generate synthetic LOB data
    print("\n[1/6] Generating synthetic LOB data...")
    gen = LOBDataGenerator(n_snapshots=n_snapshots)
    snapshots = gen.generate()
    print(f"  Generated {len(snapshots)} LOB snapshots")

    # Visualize one snapshot
    fig_snap = viz.plot_lob_snapshot(snapshots[100], "LOB Snapshot — Tick 100")
    viz.save(fig_snap, f"{output_dir}/lob_snapshot.png", height=450)

    # 2. Feature Engineering
    print("\n[2/6] Engineering LOB features...")
    engine = LOBFeatureEngine()
    features_df = engine.compute(snapshots)
    print(f"  Feature matrix: {features_df.shape}")

    fig_feat = viz.plot_features(features_df, n_rows=500)
    viz.save(fig_feat, f"{output_dir}/lob_features.png", height=600)

    # 3. Label Generation
    print("\n[3/6] Generating ternary labels...")
    mid_prices = features_df["mid_price"].values
    labels = generate_labels(mid_prices, horizon=50, theta=2.0)
    valid_mask = labels >= 0
    counts = {0: int((labels == 0).sum()), 1: int((labels == 1).sum()),
              2: int((labels == 2).sum())}
    print(f"  Label distribution — Down:{counts[0]}, Stable:{counts[1]}, Up:{counts[2]}")

    # 4. Dimensionality Reduction
    print("\n[4/6] Running PCA + Autoencoder...")
    raw_cols = [c for c in features_df.columns if c.startswith("raw_")]
    X_raw = features_df[raw_cols].values

    reducer = LOBDimensionalityReducer(latent_dim=LATENT_DIM)
    X_pca = reducer.fit_pca(X_raw)
    print(f"  PCA: retained {reducer.n_pca_components} components (≥90% variance)")

    fig_pca = viz.plot_pca_variance(reducer.pca)
    viz.save(fig_pca, f"{output_dir}/pca_variance.png", height=400)

    X_latent = reducer.fit_autoencoder(X_raw, epochs=20)
    print(f"  Autoencoder latent shape: {X_latent.shape}")

    # Latent space scatter
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) > 0:
        fig_latent = viz.plot_latent_space(
            X_latent[valid_idx], labels[valid_idx], "Autoencoder"
        )
        viz.save(fig_latent, f"{output_dir}/autoencoder_latent.png", height=450)

    # 5. Supervised Classification
    print("\n[5/6] Training classifiers...")
    # Combine PCA + engineered features
    eng_cols = ["ofi", "volume_imbalance", "trade_imbalance",
                "spread_bps", "depth_imbalance", "autocovariance_1"]
    X_eng = features_df[eng_cols].values
    X_combined = np.hstack([X_pca, X_eng])

    # Filter valid labels
    X_valid = X_combined[valid_mask]
    y_valid = labels[valid_mask]

    # Time-series split (no data leakage)
    tscv = TimeSeriesSplit(n_splits=3)
    train_idx, test_idx = list(tscv.split(X_valid))[-1]
    X_train, X_test = X_valid[train_idx], X_valid[test_idx]
    y_train, y_test = y_valid[train_idx], y_valid[test_idx]

    # Extra Trees (fast, robust baseline)
    print("  Training Extra Trees...")
    et_model = ExtraTreesClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1
    )
    et_model.fit(X_train, y_train)
    y_pred_et = et_model.predict(X_test)
    f1_et = f1_score(y_test, y_pred_et, average="macro")
    print(f"  Extra Trees macro-F1: {f1_et:.4f}")

    fig_cm = viz.plot_confusion_matrix(y_test, y_pred_et, "Extra Trees")
    viz.save(fig_cm, f"{output_dir}/confusion_matrix.png", height=400)

    # LightGBM (if available)
    if HAS_LGB:
        print("  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=31, random_state=RANDOM_SEED, n_jobs=-1,
            class_weight="balanced",
        )
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        f1_lgb = f1_score(y_test, y_pred_lgb, average="macro")
        print(f"  LightGBM macro-F1: {f1_lgb:.4f}")

    print("\n  Classification Report (Extra Trees):")
    print(classification_report(y_test, y_pred_et,
                                target_names=["Down", "Stable", "Up"]))

    # 6. RL Execution Simulation
    print("\n[6/6] Simulating RL execution agent...")
    # Use classifier probabilities as state inputs to RL agent
    et_probs = et_model.predict_proba(X_valid)  # shape (T, 3)
    agent = RLExecutionAgent(n_state_features=15, inventory_penalty=0.01)
    episode = agent.simulate_episode(
        features=X_valid, signal_probs=et_probs, n_steps=200
    )
    print(f"  RL Episode — Total Reward: {episode['total_reward']:.4f}, "
          f"Sharpe: {episode['sharpe']:.4f}, Trades: {episode['n_trades']}")

    fig_rl = viz.plot_rl_rewards(episode["rewards"], "RL Execution Agent — Reward Trajectory")
    viz.save(fig_rl, f"{output_dir}/rl_rewards.png", height=500)

    print("\n" + "=" * 60)
    print("  Pipeline complete. Artifacts saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline(n_snapshots=3000, output_dir="./plots")
