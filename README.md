# ML for Limit Order Books — Feature Engineering & Signal Generation
### Systematic Macro Quant Research · High-Frequency Microstructure
#### LOB Feature Extraction · Trading Signal Generation · Deep Learning · RL

> **Delivery philosophy:** Every section below follows a *microstructure intuition first, math as evidence second* structure. At Citadel, Jane Street, or Two Sigma, interviewers probe two levels deeper on any technique named. The safest rule: never cite a model you cannot calibrate from first principles on live tick data.

---
---

[↩️ Back to README.md](../README.md)

---
---

## ⏱️ Module Budget

```
MODULE                      SECTION    FOCUS                          QUANT LENS
──────────────────────────  ─────────  ─────────────────────────────  ──────────────────────────────────
LOB FEATURE EXTRACTION      F1 – F4    Classical + ML feature eng.    OFI, Volume/Trade Imbalance, AE
TRADING SIGNAL GENERATION   S1 – S4    Target def., classifiers        DeepLOB, LSTM, XGBoost, RL
DIMENSIONALITY REDUCTION    D1 – D2    PCA + Autoencoder               Latent LOB representations
PRODUCTION SYSTEMS          P1 – P2    Inference, backtesting          Design-by-contract, latency
```

> **Priority rule:** Feature quality determines signal ceiling. A weak feature set with a strong model underperforms a strong feature set with a weak model. Build LOB features first; classifier choice is secondary.

---

## Table of Contents

### 📊 LOB FEATURE EXTRACTION
- [F1 · Order Flow Imbalance (OFI) — Classical Microstructure Signal](#f1--order-flow-imbalance-ofi--classical-microstructure-signal)
- [F2 · Volume & Trade Imbalance — Queue Dynamics](#f2--volume--trade-imbalance--queue-dynamics)
- [F3 · CNN & Autoencoder Feature Extraction — ML Approach](#f3--cnn--autoencoder-feature-extraction--ml-approach)
- [F4 · PCA for LOB Dimensionality Reduction](#f4--pca-for-lob-dimensionality-reduction)

### 🎯 TRADING SIGNAL GENERATION
- [S1 · Target Definition — Up / Stable / Down Labels](#s1--target-definition--up--stable--down-labels)
- [S2 · Deep Learning Classifiers — DeepLOB, CNN, LSTM, MLP](#s2--deep-learning-classifiers--deeplob-cnn-lstm-mlp)
- [S3 · Tree-Based Classifiers — XGBoost, LightGBM, Extra Trees](#s3--tree-based-classifiers--xgboost-lightgbm-extra-trees)
- [S4 · Reinforcement Learning — RL-Based Execution & Signal Refinement](#s4--reinforcement-learning--rl-based-execution--signal-refinement)

### 🧠 DIMENSIONALITY REDUCTION
- [D1 · PCA on LOB Snapshots](#d1--pca-on-lob-snapshots)
- [D2 · Autoencoder Latent Space Representations](#d2--autoencoder-latent-space-representations)

### 🏭 PRODUCTION SYSTEMS
- [P1 · Design-by-Contract & Data-Oriented Architecture](#p1--design-by-contract--data-oriented-architecture)
- [P2 · Backtesting & Performance Attribution](#p2--backtesting--performance-attribution)

- **[Quick-Reference Equation Sheet](#quick-reference-equation-sheet)**

[🔝 Back to Top](#-table-of-contents)

---
---

# 📊 LOB FEATURE EXTRACTION

---

## F1 · Order Flow Imbalance (OFI) — Classical Microstructure Signal

**Open with the intuition (15 seconds):**
> "Order Flow Imbalance quantifies the net buying vs selling pressure at the best bid and ask. A large positive OFI signals aggressive buy-side queue building — a leading indicator of upward price pressure within the next 1–10 ticks. At Jane Street, OFI variants are among the highest-signal features in any LOB prediction stack."

---

### OFI — Formal Definition

The bid-side OFI contribution at time $t$:

$$OF_{b,t} = \begin{cases} v_{b,t} & \text{if } p_{b,t} > p_{b,t-1} \\ v_{b,t} - v_{b,t-1} & \text{if } p_{b,t} = p_{b,t-1} \\ -v_{b,t} & \text{if } p_{b,t} < p_{b,t-1} \end{cases}$$

The ask-side OFI contribution:

$$OF_{a,t} = \begin{cases} -v_{a,t} & \text{if } p_{a,t} > p_{a,t-1} \\ v_{a,t} - v_{a,t-1} & \text{if } p_{a,t} = p_{a,t-1} \\ v_{a,t} & \text{if } p_{a,t} < p_{a,t-1} \end{cases}$$

The aggregate OFI:

$$\boxed{OFI_t = OF_{b,t} - OF_{a,t}}$$

**Say it out loud:** *"OFI accumulates the net change in liquidity at the best quotes. When the bid queue grows and the ask queue shrinks simultaneously, OFI spikes positively — the market is pricing in buy pressure before any trade occurs. This is why OFI is a leading, not lagging, microstructure signal."*

---

### OFI Multi-Level Extension

For a production system, extend OFI across the top $L$ levels of the book:

$$OFI_t^{(L)} = \sum_{\ell=1}^{L} w_\ell \cdot (OF_{b,t}^{(\ell)} - OF_{a,t}^{(\ell)})$$

where $w_\ell = e^{-\lambda \ell}$ decays exponentially with depth. In Citadel's equity stat-arb books, $L=5$ and $\lambda \approx 0.5$ captures 85%+ of predictive OFI variance.

```
OFI SIGNAL INTERPRETATION TABLE

  OFI Value     Market Condition             Expected Mid Move (1–5 ticks)
  ──────────    ─────────────────────────    ─────────────────────────────
  OFI >> 0      Buy-side queue dominates     ↑ Upward pressure; go long
  OFI ≈  0      Balanced order flow          → Stable; mean reversion
  OFI << 0      Sell-side queue dominates    ↓ Downward pressure; go short
  |OFI| spike   Sudden liquidity imbalance   High conviction signal
```

[🔝 Back to Top](#-table-of-contents)

---
---

## F2 · Volume & Trade Imbalance — Queue Dynamics

**Open with the intuition (15 seconds):**
> "Volume imbalance measures the static snapshot of buy vs sell liquidity available at the touch. Trade imbalance measures the executed flow. Together they capture both latent intent and revealed conviction. In a Two Sigma systematic context, these two features combined have an information coefficient of ~0.04–0.06 on 10-tick forward mid returns — modest alone but highly complementary to OFI."

---

### Volume Imbalance

$$\text{Volume Imbalance}_t = \frac{v_{a,t} - v_{b,t}}{v_{a,t} + v_{b,t}} \in [-1, +1]$$

**Interpretation:** A value near $+1$ means the ask queue is vastly deeper — sell-side supply dominates, suppressing upward moves. Near $-1$: bid queue deeper — downward pressure is absorbed.

---

### Trade Imbalance

Aggregates executed buy vs sell volume over a window $[t_{k-1}, t_k]$:

$$\text{Trade Imbalance}_t = \sum_{n=N(t_{k-1})}^{N(t_k)} b_n - \sum_{n=N(t_{k-1})}^{N(t_k)} s_n$$

where $b_n$ is buyer-initiated trade volume and $s_n$ seller-initiated. Positive values indicate aggressive buying; negative, aggressive selling.

```
FEATURE COMBINATION MATRIX

  Volume Imbalance   Trade Imbalance   OFI        Regime               Action
  ─────────────────  ───────────────   ────────   ──────────────────   ──────────────────
  > 0                > 0               > 0        Strong bull signal   Aggressive long
  > 0                < 0               ≈ 0        Absorption regime    Wait — reversal risk
  < 0                < 0               < 0        Strong bear signal   Aggressive short
  ≈ 0                ≈ 0               ≈ 0        Thin / illiquid      Reduce size / wait
```

[🔝 Back to Top](#-table-of-contents)

---
---

## F3 · CNN & Autoencoder Feature Extraction — ML Approach

**Open with the intuition (15 seconds):**
> "A raw LOB snapshot is a 2×L price-volume matrix, updating hundreds of times per second. Hand-crafted features like OFI are linear projections of this matrix. CNNs learn non-linear spatial patterns across price levels and time. Autoencoders compress the LOB into a low-dimensional latent code — equivalent to a learned, non-linear PCA. The compressed representation is what feeds the downstream classifier."

---

### DeepLOB Architecture

The DeepLOB model (Zhang et al., IEEE TSP 2019) processes raw LOB input through:

1. **Convolutional block:** $3 \times 1$ filters extract local price-level features; $1 \times 2$ filters pool bid/ask pairs
2. **Inception module:** Parallel $1 \times 1$, $3 \times 1$, $5 \times 1$ filters capture multi-scale temporal patterns
3. **LSTM layers:** Capture sequential dependencies across LOB snapshots
4. **Softmax output:** $P(\text{Up}), P(\text{Stable}), P(\text{Down})$

```
INPUT (100 LOB snapshots × 40 features)
  │
  ├─ Conv2D(32, 1×2) → BN → LeakyReLU  [bid/ask pairing]
  ├─ Conv2D(32, 4×1) → BN → LeakyReLU  [temporal patterns]
  │
  ├─ Inception: [1×1 | 3×1 | 5×1 | MaxPool]
  │
  ├─ LSTM(64) → LSTM(64)
  │
  └─ Dense(3) → Softmax  [Up / Stable / Down]
```

---

### Autoencoder for LOB Compression

The autoencoder learns a compressed representation $z \in \mathbb{R}^d$ of the raw LOB snapshot $x \in \mathbb{R}^{40}$:

$$\text{Encoder: } z = f_\theta(x) = \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2)$$

$$\text{Decoder: } \hat{x} = g_\phi(z) = W_4 \cdot \text{ReLU}(W_3 z + b_3) + b_4$$

$$\mathcal{L}_{AE} = \|x - \hat{x}\|_2^2 + \beta \|z\|_1 \quad \text{(sparse autoencoder variant)}$$

The latent code $z$ with $d \ll 40$ becomes the feature vector for the signal classifier. At production quant firms, $d \in [8, 16]$ preserves 90%+ of reconstruction quality.

**ReLU Rationale:** ReLU activations ( $\max(0, z)$ ) enforce sparsity in the latent representation — most hidden units are inactive for any given LOB snapshot, creating disentangled features where each latent dimension captures a distinct liquidity regime.

[🔝 Back to Top](#-table-of-contents)

---
---

## F4 · PCA for LOB Dimensionality Reduction

**Open with the intuition (15 seconds):**
> "PCA on LOB snapshots finds the principal axes of liquidity variation. The first PC typically captures overall market depth; the second captures bid/ask asymmetry; the third captures queue shape. These three components explain 70–85% of total LOB variance on liquid equity futures. PCA is the linear baseline; autoencoders are the non-linear extension."

---

### PCA on LOB Snapshots

Given a matrix $X \in \mathbb{R}^{T \times 40}$ of standardized LOB snapshots:

$$X = U \Sigma V^\top \quad \text{(SVD decomposition)}$$

The $k$-th principal component score at time $t$: $z_{t,k} = x_t^\top v_k$

**Explained variance ratio:** $\text{EVR}_k = \sigma_k^2 / \sum_j \sigma_j^2$

```
TYPICAL PC INTERPRETATION FOR EQUITY FUTURES LOB

  PC1 (≈35% variance)   Overall depth level — correlated with VIX inverse
  PC2 (≈20% variance)   Bid/ask asymmetry — leading indicator of OFI
  PC3 (≈12% variance)   Queue shape curvature — signals passive accumulation
  PC4–PC5 (≈10%)        Cross-level interactions
  PC6+ (< 25%)          Noise — drop in production
```

[🔝 Back to Top](#-table-of-contents)

---
---

# 🎯 TRADING SIGNAL GENERATION

---

## S1 · Target Definition — Up / Stable / Down Labels

**Open with the intuition (15 seconds):**
> "Label quality is the ceiling of any supervised ML system. A poorly defined target creates irrecoverable label noise. The standard approach is a smoothed mid-price change over a forward horizon, thresholded at ±θ. Choosing θ is a signal-to-noise tradeoff: too small → Stable dominates (80%+ class), too large → rare signals, poor generalization."

---

### Target Variable

The ternary label at time $t$ with forward horizon $\Delta$ ticks:

$$\text{target}_t = \begin{cases} \text{Down} & \text{if } (\text{mid}_{t+\Delta} - \text{mid}_t) < -\theta \\ \text{Stable} & \text{if } -\theta \leq (\text{mid}_{t+\Delta} - \text{mid}_t) \leq \theta \\ \text{Up} & \text{if } (\text{mid}_{t+\Delta} - \text{mid}_t) > \theta \end{cases}$$

where:

$$
\text{mid}_t = (p_{b,t} + p_{a,t}) / 2
$$

and $\theta$ is the threshold (typically 0.5–2 tick sizes).

**Smoothed variant (production preferred):**

$$\text{mid}_{t+\Delta}^{\text{smooth}} = \frac{1}{k} \sum_{j=1}^{k} \text{mid}_{t+\Delta + j}$$

Rolling-average smoothing reduces label flip noise from transient liquidity shocks.

```
THRESHOLD SELECTION GUIDE

  θ (in tick sizes)   Up/Down %    Stable %    IC Range    Recommended Use
  ─────────────────   ──────────   ─────────   ─────────   ─────────────────────────
  0.5                 12% each     76%         0.03–0.05   Ultra-HF (< 10ms horizon)
  1.0                 20% each     60%         0.05–0.08   HF (10ms–1s horizon)
  2.0                 30% each     40%         0.07–0.12   Mid-freq (1s–60s horizon)
  5.0                 38% each     24%         0.08–0.14   Low-freq (1min–5min)
```

[🔝 Back to Top](#-table-of-contents)

---
---

## S2 · Deep Learning Classifiers — DeepLOB, CNN, LSTM, MLP

**Open with the intuition (15 seconds):**
> "Deep learning classifiers operate directly on raw LOB time-series tensors. The key architectural tradeoff is spatial feature extraction (CNN) vs temporal memory (LSTM). DeepLOB combines both. In practice, DeepLOB achieves 65–72% accuracy on liquid equities — approximately 8–15pp above a majority-class baseline — which translates to Sharpe 1.5–2.5 in backtests before transaction costs."

---

### Classifier Comparison Matrix

```
CLASSIFIER     INPUT FORMAT          STRENGTHS                   WEAKNESSES             LATENCY
─────────────  ─────────────────     ──────────────────────────  ─────────────────────  ────────
DeepLOB        LOB tensor (T×40)     Best accuracy; spatial+seq  Large; slow to train   5–15ms
CNN            LOB image             Fast; spatial patterns      No temporal memory     1–3ms
LSTM           Feature sequence      Temporal dependencies       Vanishing gradient     2–5ms
MLP            Feature vector        Fast inference; simple      No sequence modeling   <1ms
XGBoost        Feature vector        Interpretable; fast train   No raw tensor input    <1ms
LightGBM       Feature vector        Fastest train; low memory   Same as XGBoost        <1ms
Extra Trees    Feature vector        Low variance; robust        Lower accuracy         <1ms
Ensemble       Multiple outputs      Best Sharpe in practice     Complex deployment     5–10ms
```

### Confusion Matrix Interpretation

The confusion matrix for a ternary LOB classifier reveals two failure modes:

```
                    PREDICTED
                  Down  Stable  Up
TRUE    Down   [ TP_D   FN_DS  FN_DU ]
        Stable [ FP_SD  TP_S   FP_SU ]
        Up     [ FN_UD  FN_US  TP_U  ]
```

**Critical failure modes for P&L:**
- Off-diagonal `Down → Up` or `Up → Down` errors: **directional flip** — catastrophic, takes maximum loss
- Off-diagonal `Down → Stable` or `Up → Stable` errors: **missed signal** — no loss but no gain
- For execution: minimize directional flips; tolerate missed signals

**Weighted loss for imbalanced classes:**

$$\mathcal{L}_{CE} = -\sum_{c \in \{D,S,U\}} w_c \cdot y_c \log(\hat{p}_c)$$

Set $w_{\text{Stable}} < w_{\text{Up}} = w_{\text{Down}}$ to penalize directional errors more heavily.

[🔝 Back to Top](#-table-of-contents)

---
---

## S3 · Tree-Based Classifiers — XGBoost, LightGBM, Extra Trees

**Open with the intuition (15 seconds):**
> "Tree ensembles operate on engineered features — OFI, volume imbalance, spreads, autocovariance — not raw tensors. They train in minutes vs hours for deep models, interpret cleanly via SHAP, and generalize well out-of-sample. In a live trading context at a systematic macro fund, LightGBM with 200 features wins on the Sharpe-per-compute tradeoff 70% of the time vs deep learning."

---

### Feature Importance with SHAP

For tree models, SHAP values decompose the prediction:

$$f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j(x_j)$$

where $\phi_j$ is the marginal contribution of feature $j$. In production LOB models:

- **OFI** typically ranks #1–2 in SHAP importance
- **Spread** (ask − bid) ranks #3–5 — a proxy for adverse selection cost
- **Volume imbalance** ranks #4–7
- **Autoencoder latent codes** can rank #2–8 depending on regime

[🔝 Back to Top](#-table-of-contents)

---
---

## S4 · Reinforcement Learning — RL-Based Execution & Signal Refinement

**Open with the intuition (15 seconds):**
> "Supervised classifiers predict direction. RL agents learn to act optimally under uncertainty — sizing, timing, and cancellation decisions that a supervised model cannot make. The key insight: the signal classifier outputs a belief state; the RL agent uses that belief state as part of its observation to decide whether to submit, wait, or cancel. This is how state-of-the-art systems at top HFT firms work."

---

### MDP Formulation for LOB Trading

The RL problem is formulated as a Markov Decision Process:

$$(\mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{T}, \gamma)$$

**State space $\mathcal{S}$:**

$$s_t = [\text{LOB features}_t, \text{position}_t, \text{inventory risk}_t, \hat{p}_t^{\text{Up}}, \hat{p}_t^{\text{Down}}]$$

**Action space $\mathcal{A}$:**

$$\lbrace Buy Market, Sell Market, Post Bid, Post Ask, Cancel, Hold \rbrace$$

**Reward function:**

$$r_t = \underbrace{\Delta \text{PnL}_t}_{\text{realized}} - \underbrace{\lambda \cdot |\text{inventory}_t|}_{\text{inventory penalty}} - \underbrace{\mu \cdot \mathbb{1}[\text{adverse fill}]}_{\text{adverse selection cost}}$$

**Algorithm:** Proximal Policy Optimization (PPO) with:
- Actor: MLP $[128, 64, |\mathcal{A}|]$ with ReLU activations
- Critic: MLP $[128, 64, 1]$ for value estimation
- Entropy regularization to prevent premature policy collapse

```
RL TRAINING CURRICULUM FOR LOB

  Phase 1 (Episodes 0–500)     Learn to avoid adverse selection (high λ)
  Phase 2 (Episodes 500–2000)  Learn optimal entry timing (reduce λ)
  Phase 3 (Episodes 2000+)     Learn inventory management (full reward)
```

[🔝 Back to Top](#-table-of-contents)

---
---

# 🧠 DIMENSIONALITY REDUCTION

---

## D1 · PCA on LOB Snapshots

The PCA pipeline for production LOB systems:

1. **Standardize** each of the 40 features (10 levels × 4 fields) to zero mean, unit variance
2. **Fit PCA** on a rolling 20-day training window; refit weekly
3. **Select $k$** components via 90% explained variance criterion
4. **Monitor** PC loadings for structural breaks — a sudden shift in PC1 loadings signals a microstructure regime change

$$k^{\*} = \min \lbrace k : \frac{\sum_{j=1}^k \sigma_j^2}{\sum_{j=1}^{40} \sigma_j^2} \geq 0.90 \rbrace$$

[🔝 Back to Top](#-table-of-contents)

---

## D2 · Autoencoder Latent Space Representations

The autoencoder bottleneck architecture for LOB:

```
INPUT (40-dim LOB snapshot)
  │
  Dense(32) → ReLU  [encoder layer 1]
  │
  Dense(16) → ReLU  [encoder layer 2]
  │
  Dense(8)  → ReLU  [BOTTLENECK — latent code z]
  │
  Dense(16) → ReLU  [decoder layer 1]
  │
  Dense(32) → ReLU  [decoder layer 2]
  │
  Dense(40) → Linear [reconstruction x̂]
```

**Variational Autoencoder (VAE) extension** for generative LOB modeling:

$$\mathcal{L}_{VAE} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{regularization}}$$

The KL divergence term forces $z \sim \mathcal{N}(0, I)$, enabling interpolation between liquidity regimes in latent space — a technique used for stress-testing LOB models.

[🔝 Back to Top](#-table-of-contents)

---
---

# 🏭 PRODUCTION SYSTEMS

---

## P1 · Design-by-Contract & Data-Oriented Architecture

```python
# Canonical LOB data contract — replace fabricated data with real feed here
LOBSnapshot = TypedDict('LOBSnapshot', {
    'timestamp_ns':    np.int64,       # Nanosecond UNIX timestamp
    'bid_prices':      np.ndarray,     # Shape (10,) — 10 bid levels
    'bid_volumes':     np.ndarray,     # Shape (10,) — corresponding volumes
    'ask_prices':      np.ndarray,     # Shape (10,) — 10 ask levels
    'ask_volumes':     np.ndarray,     # Shape (10,) — corresponding volumes
    'last_trade_px':   np.float64,     # Last traded price
    'last_trade_vol':  np.float64,     # Last traded volume
})
```

**Data-oriented principle:** The LOB pipeline is a pure data transformation. Each stage takes a well-typed array and returns a well-typed array. No hidden state; no in-place mutation. This makes every stage unit-testable in isolation — the standard at Citadel's systematic equities pod.

---

## P2 · Backtesting & Performance Attribution

**Sharpe-consistent backtest protocol:**

$$\text{Sharpe}_{\text{annualized}} = \frac{\mu_{\text{daily}} \cdot \sqrt{252}}{\sigma_{\text{daily}}}$$

**Critical production checks:**
- No lookahead: all features computed on $t-1$ data; label computed from $t+\Delta$ data
- Walk-forward cross-validation with 20-day training, 5-day test windows
- Transaction cost model: $c = \text{half-spread} + \text{market impact}$; use the Almgren-Chriss model for impact
- Turnover cap:

$$
\\|\text{position}_{t} - \text{position}_{t-1}\\| \leq \text{POV} \cdot \text{ADV}_t
$$

[🔝 Back to Top](#-table-of-contents)

---
---

# Quick-Reference Equation Sheet

```
══════════════════════════════════════════════════════════════════════════════
LOB FEATURE EXTRACTION
══════════════════════════════════════════════════════════════════════════════

Order Flow Imbalance:
  OFI_t = OF_{b,t} - OF_{a,t}
  OF_{b,t}: +v if bid price rises, Δv if price same, -v if price falls
  OF_{a,t}: -v if ask price rises, Δv if price same, +v if price falls

Multi-Level OFI:
  OFI_t^(L) = Σ_ℓ exp(-λℓ) · (OF_{b,t}^(ℓ) - OF_{a,t}^(ℓ))   [λ ≈ 0.5]

Volume Imbalance:
  VI_t = (v_a - v_b) / (v_a + v_b)   ∈ [-1, +1]

Trade Imbalance:
  TI_t = Σ b_n - Σ s_n   over window [t_{k-1}, t_k]

Autoencoder (sparse variant):
  z = f_θ(x) = ReLU(W_2·ReLU(W_1·x + b_1) + b_2)
  L_AE = ||x - x̂||² + β||z||₁

══════════════════════════════════════════════════════════════════════════════
TRADING SIGNAL GENERATION
══════════════════════════════════════════════════════════════════════════════

Target Label:
  Up    if (mid_{t+Δ} - mid_t) > +θ
  Down  if (mid_{t+Δ} - mid_t) < -θ
  Stable otherwise

Weighted Cross-Entropy:
  L_CE = -Σ_c w_c · y_c · log(p̂_c)   [w_Stable < w_Up = w_Down]

RL Reward:
  r_t = ΔPNL_t - λ·|inventory_t| - μ·1[adverse fill]

PPO Policy Gradient:
  L_PPO = E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]   [ε = 0.2]

══════════════════════════════════════════════════════════════════════════════
DIMENSIONALITY REDUCTION
══════════════════════════════════════════════════════════════════════════════

PCA — Optimal k:
  k* = min{k : Σ_j^k σ_j² / Σ_j^40 σ_j² ≥ 0.90}

VAE ELBO:
  L_VAE = E[log p_θ(x|z)] - D_KL(q_φ(z|x) || N(0,I))

══════════════════════════════════════════════════════════════════════════════
PRODUCTION EXECUTION RULES
══════════════════════════════════════════════════════════════════════════════

Sharpe (annualized):        μ_daily · √252 / σ_daily
Max Position (POV rule):    |Δpos_t| ≤ POV · ADV_t      [POV = 3–8%]
Transaction Cost Model:     c = half-spread + α·(Q/ADV)^0.6   [Almgren-Chriss]
Signal Decay Check:          IC_t = corr(signal_{t-k}, label_t)  vs k
```

[🔝 Back to Top](#-table-of-contents)

---

*Last updated: June 2026 · ML for LOB Microstructure — Systematic Quant Research*

[↩️ Back to README.md](../README.md)
