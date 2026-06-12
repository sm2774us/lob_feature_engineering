# Systematic Macro Feature Engineering: Production ML for Cross-Asset Alpha
### HLS Trading · Systematic Macro Research · FX · Commodities · Futures · Rates
#### Feature Engineering · Dimensionality Reduction · PCA · Autoencoders · RL · Mid-to-High Frequency

> **HLS Trading context:** HLS operates a live systematic macro strategy across FX, commodities, futures, and rates, rebalanced ~4× daily with MFT capabilities and institutional infrastructure. This document reflects the *statistics-first* research culture of the firm: every feature must demonstrate genuine predictive content, survive regime shifts, and meet production latency budgets. The founders' combined tenure at Exodus, Millennium, and DRW sets the bar: no technique for its own sake, no noise dressed as signal.

---

[↩️ Back to README.md](../README.md)

---

## ⏱️ Module Budget

```
MODULE                         SECTION    FOCUS                                   QUANT LENS
─────────────────────────────  ─────────  ──────────────────────────────────────  ──────────────────────────────────────
RAW DATA CONTRACTS             E0         Design-by-contract, tick schemas        Point-in-time safety, FX/Futures/Rates
CLASSICAL MACRO FEATURES       E1 – E4    Carry, trend, value, growth             Cross-asset signal construction
MICROSTRUCTURE FEATURES        E5 – E6    OFI, VPIN, spread elasticity            Intraday execution signal layer
DIMENSIONALITY REDUCTION       E7 – E8    PCA + Autoencoder                       Latent regime representations
ML & RL FEATURE PIPELINES      E9 – E10   ReLU networks, PPO, SHAP                Non-linear alpha extraction
PRODUCTION VALIDATION          E11        Walk-forward IC, signal decay           Anti-overfitting framework
```

> **HLS priority rule:** At mid-to-high frequency (4× daily rebalance), feature quality is the primary PnL driver. A carry signal with a 3-month IC of 0.04 and zero lookahead beats a neural network with IC 0.08 that leaks future CPI prints. Build stationary, point-in-time features first. ML is the amplifier, not the source.

---

## Table of Contents

### 📐 DATA CONTRACTS & FOUNDATIONS
- [E0 · Data Contracts & Point-in-Time Architecture](#e0--data-contracts--point-in-time-architecture)

### 📊 CLASSICAL MACRO FEATURE ENGINEERING
- [E1 · Carry Features — FX, Rates, Commodities](#e1--carry-features--fx-rates-commodities)
- [E2 · Trend & Momentum Features — Time-Series & Cross-Sectional](#e2--trend--momentum-features--time-series--cross-sectional)
- [E3 · Value & Mean-Reversion Features](#e3--value--mean-reversion-features)
- [E4 · Macro Surprise Features — Growth & Inflation Signals](#e4--macro-surprise-features--growth--inflation-signals)

### 🔬 MICROSTRUCTURE FEATURE ENGINEERING
- [E5 · Order Flow & Microstructure Features](#e5--order-flow--microstructure-features)
- [E6 · VPIN, Adverse Selection & Execution Quality Features](#e6--vpin-adverse-selection--execution-quality-features)

### 🧠 DIMENSIONALITY REDUCTION
- [E7 · PCA for Cross-Asset Regime Compression](#e7--pca-for-cross-asset-regime-compression)
- [E8 · Autoencoder Latent Macro States — ReLU & VAE](#e8--autoencoder-latent-macro-states--relu--vae)

### 🤖 ML & RL FEATURE PIPELINES
- [E9 · Non-Linear Feature Interactions — ReLU Networks & Attention](#e9--non-linear-feature-interactions--relu-networks--attention)
- [E10 · Reinforcement Learning for Dynamic Feature Weighting](#e10--reinforcement-learning-for-dynamic-feature-weighting)

### 🏭 PRODUCTION VALIDATION
- [E11 · Signal Validation, IC Decay & Walk-Forward Protocols](#e11--signal-validation-ic-decay--walk-forward-protocols)

- **[Quick-Reference Equation Sheet](#quick-reference-equation-sheet)**

[🔝 Back to Top](#-table-of-contents)

---
---

# 📐 DATA CONTRACTS & FOUNDATIONS

---

## E0 · Data Contracts & Point-in-Time Architecture

**Open with the intuition (15 seconds):**
> "At HLS, every signal feeds a live MFT book. A single lookahead contamination — one CPI print seen before its release timestamp — does not just invalidate a backtest, it invalidates an entire research cycle. The data contract is the primary institutional defence: a typed, immutable, bitemporal record that separates *when data was valid* from *when it was known*. This is the standard at Millennium and Citadel's systematic macro pods."

---

### Bitemporal Data Model

Every macro data record carries two timestamps:

```math
\text{valid\_time} = \text{period the data describes} \quad \text{(e.g., May CPI = May)}
```

```math
\text{knowledge\_time} = \text{when the firm could first act on it} \quad \text{(e.g., June 12, 08:30 EST release)}
```

Feature computation uses **knowledge\_time** exclusively. Forward-fill, interpolation, and any form of temporal alignment that leaks knowledge\_time are strictly prohibited.

```python
# HLS Trading — Canonical cross-asset data contracts
# Replace fabricated generators with live vendor feeds (e.g., Bloomberg, Refinitiv, Quandl)

MacroTickRecord = TypedDict('MacroTickRecord', {
    'knowledge_time_ns':  np.int64,      # UNIX nanoseconds — when firm can act
    'valid_time_ns':      np.int64,      # UNIX nanoseconds — period described
    'asset_id':           str,           # Bloomberg ticker / internal symbol
    'asset_class':        str,           # 'FX' | 'RATES' | 'COMMODITY' | 'EQUITY_IDX'
    'price':              np.float64,    # Mid price, settlement, or spot rate
    'bid':                np.float64,    # Best bid (NaN for daily data)
    'ask':                np.float64,    # Best ask (NaN for daily data)
    'volume':             np.float64,    # Notional or contract volume
    'open_interest':      np.float64,    # Futures OI (NaN for spot FX)
})

MacroSurpriseRecord = TypedDict('MacroSurpriseRecord', {
    'knowledge_time_ns':  np.int64,      # Release timestamp — strictly enforced
    'indicator':          str,           # 'CPI_YOY' | 'NFP' | 'GDP_QOQ' | 'FOMC_RATE'
    'actual':             np.float64,    # Released value
    'consensus':          np.float64,    # Bloomberg/Reuters survey median
    'prior':              np.float64,    # Prior period value
    'revision':           np.float64,    # Revision to prior (NaN if none)
})
```

**Data-oriented principle:** The feature pipeline is a pure, stateless transformation. Each stage maps a typed record array → typed feature array with no hidden state, in-place mutation, or cross-contamination between assets. Every transformation is unit-testable on synthetic data before live feed integration.

[🔝 Back to Top](#-table-of-contents)

---
---

# 📊 CLASSICAL MACRO FEATURE ENGINEERING

---

## E1 · Carry Features — FX, Rates, Commodities

**Open with the intuition (15 seconds):**
> "Carry is the return earned by holding an asset absent price change — the foundational risk premium in systematic macro. In FX it is the interest rate differential; in rates it is the roll-down along the yield curve; in commodities it is the convenience yield embedded in the futures term structure. At AQR and Graham Capital, carry is consistently among the top-3 ICs in cross-asset momentum-carry combination models. For HLS's MFT rebalance cadence, the intraday carry signal is the *slow* anchor against which microstructure features are calibrated."

---

### FX Carry — Interest Rate Parity Deviation

The raw FX carry signal for currency pair $i$ at time $t$:

$$\text{Carry}_{i,t}^{\text{FX}} = \frac{F_{i,t}^{(k)} - S_{i,t}}{S_{i,t}} \approx r_{i,t}^{\text{foreign}} - r_{i,t}^{\text{domestic}}$$

where $F_{i,t}^{(k)}$ is the $k$-period forward rate and $S_{i,t}$ is the spot rate. The signal is normalized for cross-sectional comparability:

$$z_{i,t}^{\text{carry}} = \frac{\text{Carry}_{i,t}^{\text{FX}} - \mu_t^{\text{carry}}}{\sigma_t^{\text{carry}}}$$

**Volatility-adjusted carry** (the standard at Millennium FX desks):

$$\text{Carry}_{i,t}^{\text{vol-adj}} = \frac{r_{i,t}^{\text{foreign}} - r_{i,t}^{\text{domestic}}}{\sigma_{i,t}^{(22)}}$$

where $\sigma_{i,t}^{(22)}$ is the 22-day realized volatility of returns — normalizing carry by vol converts the raw yield differential into a risk-adjusted unit.

---

### Rates Carry — Yield Curve Roll-Down

For a bond or rate futures position, the carry decomposes into:

$$\text{Carry}^{\text{rates}} = \underbrace{y_t^{(n)}}_{\text{current yield}} - \underbrace{y_{t+\Delta}^{(n-\Delta)}}_{\text{yield after roll}} \cdot \frac{\Delta T}{T}$$

The **slope carry** (long 2Y, short 10Y) captures the term premium:

$$\text{SlopeCarry}_t = y_t^{(10)} - y_t^{(2)} - \mathbb{E}_t[\Delta \text{yield curve slope}]$$

---

### Commodities Carry — Roll Yield

Futures roll yield for commodity $i$ between front-month $F_1$ and second-month $F_2$:

$$\text{RollYield}_{i,t} = \frac{F_{1,t} - F_{2,t}}{F_{2,t}} \cdot \frac{252}{\Delta_{\text{contract}}}$$

Positive roll yield (backwardation) = commodity is scarce; negative (contango) = oversupply. This feature is a primary input to HLS's commodity futures alpha book.

```
CARRY SIGNAL INTERPRETATION TABLE

  Asset Class    Carry Signal     Regime                    HLS Expected Position
  ─────────────  ───────────────  ────────────────────────  ─────────────────────
  G10 FX         High vs Low IR   Risk-on carry trade       Long AUD/JPY, Short EUR/CHF
  Rates          Steep 2s10s      Bull steepener            Long back-end futures
  Commodities    Backwardation    Physical supply squeeze   Long front-month futures
  Cross-asset    Carry composite  Regime-conditioned        Vol-scaled carry basket
```

[🔝 Back to Top](#-table-of-contents)

---
---

## E2 · Trend & Momentum Features — Time-Series & Cross-Sectional

**Open with the intuition (15 seconds):**
> "Trend-following is the oldest and most robust systematic macro strategy — AHL, Winton, and Graham Capital have run it profitably for decades. The signal is the sign of past returns scaled by volatility. What distinguishes HLS's implementation is the multi-scale architecture: combining fast signals (intraday EWMA crossovers) with slow signals (monthly TSMOM) via a Gram-Schmidt orthogonalized combination that eliminates redundant information and maximizes independent breadth."

---

### Time-Series Momentum (TSMOM)

The canonical TSMOM signal over lookback $L$:

$$S_{i,t}^{\text{TSMOM}(L)} = \mathbf{sign}\left(\sum_{\tau=1}^{L} r_{i,t-\tau}\right) \cdot \frac{\mu_{i,t}^{(L)}}{\sigma_{i,t}^{(L)}}$$

**Multi-scale TSMOM** (the production standard at systematic macro firms):

$$S_{i,t}^{\text{multi}} = \sum_{k \in \{5, 21, 63, 252\}} \phi_k \cdot S_{i,t}^{\text{TSMOM}(k)}$$

where weights $\phi_k$ are optimized via elastic-net on expanding-window cross-validation. The 5-day component is the primary intraday anchor for HLS's 4× daily rebalance.

---

### EWMA Crossover Signal

$$\text{EWMA-cross}_{i,t}^{(s,f)} = \frac{\text{EWMA}_{i,t}^{(f)} - \text{EWMA}_{i,t}^{(s)}}{\sigma_{i,t}^{(s)}}$$

where $f < s$ (fast vs slow spans). The signal is continuous and avoids the discrete crossing rule's look-ahead bias. Standard spans at Citadel systematic macro: $(f, s) \in \{(2,8), (4,16), (8,32)\}$ days.

---

### Cross-Sectional Momentum (CSMOM)

Ranks assets within a universe and goes long top decile, short bottom decile:

$$\text{CSMOM}_{i,t}^{(L)} = \text{rank}_{t}\left(R_{i,t-L:t}\right) - \frac{N+1}{2}$$

**Gram-Schmidt orthogonalization** to de-correlate TSMOM and CSMOM:

$$\widetilde{\text{CSMOM}}_{i,t} = \text{CSMOM}_{i,t} - \frac{\text{Cov}(\text{CSMOM}, \text{TSMOM})}{\text{Var}(\text{TSMOM})} \cdot \text{TSMOM}_{i,t}$$

This isolates the incremental cross-sectional alpha not already captured by the time-series signal — a technique drawn directly from Gram-Schmidt signal orthogonalization (§4 of SYSTEMATIC\_MACRO.md).

[🔝 Back to Top](#-table-of-contents)

---
---

## E3 · Value & Mean-Reversion Features

**Open with the intuition (15 seconds):**
> "Value in macro is not a price-to-book ratio. It is a structural model of fair value — PPP for FX, real yield gap for rates, cost-of-carry for commodities — and the signal is the deviation from that fair value. At HLS's rebalance frequency, value operates as the *slow anchor* that prevents momentum signals from chasing extended regimes. The IC is low (~0.02) but highly complementary to carry and trend."

---

### FX Value — Purchasing Power Parity (PPP) Deviation

$$\text{PPP-dev}_{i,t} = \ln(S_{i,t}) - \ln\left(\frac{P_{i,t}^{\text{domestic}}}{P_{i,t}^{\text{foreign}}}\right)$$

where $P$ is the CPI price level. A large positive deviation (currency overvalued vs PPP) generates a short signal; negative (undervalued) generates a long signal.

**Z-score normalization** across pairs:

$$z_{i,t}^{\text{value}} = \frac{\text{PPP-dev}_{i,t} - \mu_{\text{cross-section},t}}{\sigma_{\text{cross-section},t}}$$

---

### Rates Value — Real Yield Gap

$$\text{RealYieldGap}_{i,t} = y_{i,t}^{\text{nominal}} - \pi_{i,t}^{\text{breakeven}} - y_{t}^{\text{global avg}}$$

Sovereigns with real yield gaps significantly above global average attract capital inflows. This is a primary factor in HLS's rates futures book.

---

### Commodity Value — Mean-Reversion in Spread

For commodity $i$, the value signal is the deviation of the spot-futures basis from its long-run mean:

$$\text{CommodityValue}_{i,t} = \frac{(S_{i,t} - F_{i,t}^{(3m)}) - \overline{(S-F)}_{i}}{\sigma_{S-F,i}}$$

where the mean and standard deviation are computed on a 252-day expanding window (no lookahead).

[🔝 Back to Top](#-table-of-contents)

---
---

## E4 · Macro Surprise Features — Growth & Inflation Signals

**Open with the intuition (15 seconds):**
> "Economic data releases are the primary discrete information events in systematic macro. A positive NFP surprise is not just a number — it reprices rate expectations, USD crosses, and commodity demand simultaneously within milliseconds. The Standardized Surprise Index (SSI) converts raw actuals-vs-consensus into a z-score comparable across indicators and time. At HLS, SSI features are tokenized with strict knowledge-time enforcement: no surprise is ever visible before its Bloomberg release timestamp."

---

### Standardized Surprise Index (SSI)

$$\text{SSI}_{k,t} = \frac{\text{Actual}_{k,t} - \text{Consensus}_{k,t}}{\sigma_{k}^{\text{historical surprise}}}$$

where $\sigma_k$ is the rolling 36-month standard deviation of realized surprises for indicator $k$. The SSI is unit-free and comparable across NFP, CPI, GDP, PMI, etc.

**Surprise momentum** — persistence of surprises (Cieslak & Povala, 2015):

$$\text{SurpriseMom}_{k,t} = \sum_{\tau=0}^{T} \delta^\tau \cdot \text{SSI}_{k,t-\tau} \quad \delta = 0.85$$

---

### Multi-Country Surprise Composite (Citi ESI analog)

$$\text{ESI}_{t} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{1}[\text{SSI}_{k,t} > 0] - \mathbf{1}[\text{SSI}_{k,t} < 0]$$

A rising ESI signals broadening positive macro momentum — long equities / cyclical FX; falling ESI signals deterioration — long bonds / defensive FX.

```
MACRO SURPRISE × ASSET CLASS IMPACT MATRIX

  Surprise Type     FX Impact           Rates Impact        Commodity Impact
  ───────────────   ─────────────────   ─────────────────   ──────────────────
  +NFP (US)         USD long ↑↑         Rate hike → short   Demand positive ↑
  +CPI (US)         USD long ↑          Rate hike → short   Energy/metals ↑↑
  -GDP (China)      AUD short ↓↓        Global safe havens  Industrial metals ↓↓
  +PMI (Germany)    EUR long ↑          Bund pressure ↑     Energy demand ↑
  FOMC hawkish      USD long ↑↑↑        Front-end short ↓↓  Gold short ↓
```

[🔝 Back to Top](#-table-of-contents)

---
---

# 🔬 MICROSTRUCTURE FEATURE ENGINEERING

---

## E5 · Order Flow & Microstructure Features

**Open with the intuition (15 seconds):**
> "HLS rebalances 4× daily with MFT infrastructure — this means microstructure features are not decorative; they determine entry quality. Order Flow Imbalance (OFI) at the session-level is a leading indicator of intraday price direction in futures (ES, NQ, ZN, CL) that predates the print by 1–15 minutes. VPIN (Volume-Synchronized Probability of Informed Trading) is HLS's primary adverse-selection filter: if VPIN exceeds 0.7 before a macro release, the execution engine reduces aggressive crossing and waits for queue re-establishment."

---

### Order Flow Imbalance (OFI) — Multi-Level

The bid-side OFI at level $\ell$:

$$OF_{b,t}^{(\ell)} = \begin{cases} v_{b,t}^{(\ell)} & p_{b,t}^{(\ell)} > p_{b,t-1}^{(\ell)} \\
v_{b,t}^{(\ell)} - v_{b,t-1}^{(\ell)} & p_{b,t}^{(\ell)} = p_{b,t-1}^{(\ell)} \\
-v_{b,t}^{(\ell)} & p_{b,t}^{(\ell)} < p_{b,t-1}^{(\ell)} \end{cases}$$

Multi-level OFI with exponential depth weighting:

$$\boxed{OFI_t^{(L)} = \sum_{\ell=1}^{L} e^{-\lambda \ell} \cdot \left(OF_{b,t}^{(\ell)} - OF_{a,t}^{(\ell)}\right)} \quad \lambda \approx 0.5$$

**Cumulative OFI window** for the HLS session-level signal:

$$\text{COFI}_t^{(W)} = \sum_{\tau=t-W}^{t} OFI_\tau^{(L)}$$

where $W$ is a 15-minute rolling window — matches HLS's intraday rebalance horizon.

---

### Spread Elasticity

$$\text{SpreadElasticity}_t = \frac{\Delta(P_a^{(1)} - P_b^{(1)})}{\Delta \text{OFI}_t}$$

High spread elasticity = the book reacts to order flow with widening spreads → adverse selection risk is elevated. Used by HLS as a position-scaling dampener: when SpreadElasticity > 95th percentile, reduce position delta by 30%.

[🔝 Back to Top](#-table-of-contents)

---
---

## E6 · VPIN, Adverse Selection & Execution Quality Features

**Open with the intuition (15 seconds):**
> "VPIN (Easley, López de Prado, O'Hara 2012) measures the probability that a counterparty is informed — i.e., trading against you with a structural information advantage. In practice at HLS, VPIN spikes before and during macro data releases (NFP, FOMC, CPI) as informed participants front-run the print. Conditioning execution on VPIN dramatically reduces adverse selection costs and is one of the primary alpha-preservation techniques at systematic macro MFT firms."

---

### VPIN Construction

Partition volume into buckets of size $V$ (e.g., 1000 contracts). For each bucket $\tau$:

$$\text{Buy volume}_{b,\tau} = \sum_n v_n \cdot \Phi\left(\frac{p_n - p_{n-1}}{\sigma_{\Delta p}}\right)$$

$$\text{Sell volume}_{s,\tau} = V - \text{Buy volume}_{b,\tau}$$

VPIN over a rolling window of $n$ buckets:

$$\boxed{\text{VPIN}_t = \frac{\sum_{\tau=t-n}^{t} |B_\tau - S_\tau|}{n \cdot V}}$$

**VPIN-conditioned execution logic (HLS framework):**

```
VPIN EXECUTION CONDITIONING TABLE

  VPIN Range     Interpretation              HLS Execution Response
  ────────────   ──────────────────────────  ───────────────────────────────────
  VPIN < 0.3     Low adverse selection       Full aggressive crossing permitted
  0.3 – 0.5      Normal market conditions    Standard passive/aggressive mix
  0.5 – 0.7      Elevated informed flow      Reduce aggressive crossing 50%
  VPIN > 0.7     High adverse selection      Passive-only; await queue reset
  VPIN > 0.85    Pre-release/news event      Halt execution; widen risk limits
```

[🔝 Back to Top](#-table-of-contents)

---
---

# 🧠 DIMENSIONALITY REDUCTION

---

## E7 · PCA for Cross-Asset Regime Compression

**Open with the intuition (15 seconds):**
> "A systematic macro universe of 60 instruments — G10 FX pairs, DM/EM rates, commodity futures, equity index futures — yields feature matrices with 100–500 columns after engineering. PCA compresses these into 5–8 orthogonal latent macro factors that explain 85–90% of cross-asset variance. The first PC typically captures Global Risk-On/Off (the 'beta' factor); PC2 captures Rates/Duration; PC3 captures Dollar Direction; PC4 captures Commodity Supply Shocks. These latent factors are the true inputs to HLS's allocation model."

---

### PCA Pipeline for Cross-Asset Macro

**Step 1 — Standardize.** Each feature $j$ is standardized to zero mean, unit variance on the training window:

$$\tilde{x}_{i,j,t} = \frac{x_{i,j,t} - \hat{\mu}_{j,\text{train}}}{\hat{\sigma}_{j,\text{train}}}$$

**Step 2 — Eigendecomposition.** Solve:

$$\hat{\Sigma} w_k = \lambda_k w_k \quad k = 1, \dots, K$$

**Step 3 — Optimal component selection:**

$$k^{\*} = \min\left\{k : \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^K \lambda_j} \geq 0.90\right\}$$

**Step 4 — Rolling refitting.** Refit PCA on a 60-day rolling window; check loading stability via cosine similarity between consecutive PC vectors:

$$\text{Stability}_k = \cos\theta(w_k^{(t)}, w_k^{(t-1)}) = \frac{w_k^{(t)} \cdot w_k^{(t-1)}}{\|w_k^{(t)}\| \|w_k^{(t-1)}\|}$$

A stability below 0.90 on PC1 signals a **macro regime break** — this triggers a risk reduction protocol at HLS.

---

### Random Matrix Theory (RMT) Denoising

For a feature matrix $X \in \mathbb{R}^{T \times N}$ with $Q = T/N$, noise eigenvalues follow the Marchenko-Pastur distribution:

$$\lambda_{\pm} = \sigma^2\left(1 \pm \frac{1}{\sqrt{Q}}\right)^2$$

**Signal eigenvalues** satisfy $\lambda_k > \lambda_+$. All others are noise. This is cleaner than an arbitrary 90% variance threshold and is the standard at AQR's risk models.

```
PCA LOADINGS INTERPRETATION TABLE (Typical HLS Universe)

  PC    Explained Variance   Primary Driver                 Assets Loading Highest
  ────  ───────────────────  ─────────────────────────────  ──────────────────────
  PC1   35–50%               Global Risk-On/Off (Beta)      SPX, AUD/JPY, Crude, HY
  PC2   15–25%               Rates / Duration                TY, ZN, Bund, EURUSD
  PC3   8–12%                Dollar Direction (DXY)          All G10 FX vs USD
  PC4   5–8%                 Commodity Supply / EM           Copper, AUD, BRL, CL
  PC5   3–6%                 Volatility Regime               VIX level, FX vol surface
```

[🔝 Back to Top](#-table-of-contents)

---
---

## E8 · Autoencoder Latent Macro States — ReLU & VAE

**Open with the intuition (15 seconds):**
> "PCA is a linear projection. Financial regimes are not linear — the covariance structure during a Fed pivot is fundamentally different from a risk-off liquidity event, yet both might project onto the same PC1 loading. A deep autoencoder with ReLU activations learns *non-linear* manifolds in the feature space, capturing regime interactions that PCA misses. At HLS, the autoencoder bottleneck (8–12 nodes) feeds the downstream allocation model alongside the PCA components — the two are complementary, not substitutes."

---

### Autoencoder Architecture for Macro Features

Input: standardized feature vector $x \in \mathbb{R}^{D}$ (e.g., $D = 120$ engineered macro features)

**Encoder:**

$$h_1 = \text{ReLU}(W_1 x + b_1) \in \mathbb{R}^{64}$$

$$h_2 = \text{ReLU}(W_2 h_1 + b_2) \in \mathbb{R}^{32}$$

$$z = \text{ReLU}(W_3 h_2 + b_3) \in \mathbb{R}^{d} \quad d \in \{8, 12, 16\}$$

**Decoder:**

$$\hat{h}_2 = \text{ReLU}(W_4 z + b_4) \in \mathbb{R}^{32}$$

$$\hat{h}_1 = \text{ReLU}(W_5 \hat{h}_2 + b_5) \in \mathbb{R}^{64}$$

$$\hat{x} = W_6 \hat{h}_1 + b_6 \in \mathbb{R}^{D}$$

**Sparse autoencoder loss** (enforces disentanglement):

$$\boxed{\mathcal{L}_{AE} = \underbrace{\|x - \hat{x}\|_2^2}_{\text{reconstruction}} + \underbrace{\beta \|z\|_1}_{\text{sparsity penalty}}} \quad \beta = 10^{-3}$$

**ReLU rationale:** $\text{ReLU}(z) = \max(0, z)$ creates *sparse activations* — most latent nodes are zero for any given macro snapshot. This means each active node corresponds to a distinct, identifiable regime feature (e.g., "EM stress mode", "commodity supercycle mode"), rather than a diffuse linear blend.

---

### Variational Autoencoder (VAE) Extension

The VAE learns a *generative* model of macro states, enabling regime interpolation and stress-scenario generation:

$$\mathcal{L}_{VAE} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{D_{KL}(q_\phi(z|x) \| \mathcal{N}(0,I))}_{\text{KL regularization}}$$

**HLS application:** The VAE latent space is used to generate synthetic stress scenarios (e.g., 2008-style risk-off, 2022 inflation shock) by sampling $z$ near the encoded historical regime centroids. These synthetic returns feed the tail risk budgeting model.

```
AUTOENCODER ARCHITECTURE DIAGRAM

  INPUT (D=120 macro features)
    │
    Dense(64) → ReLU  → Dropout(0.1)   [encoder layer 1]
    │
    Dense(32) → ReLU  → Dropout(0.1)   [encoder layer 2]
    │
    Dense(12) → ReLU                   [BOTTLENECK — latent z]
    │                                   ↑ sparse, disentangled
    Dense(32) → ReLU                   [decoder layer 1]
    │
    Dense(64) → ReLU                   [decoder layer 2]
    │
    Dense(D)  → Linear                 [reconstruction x̂]
    │
    LOSS: ||x - x̂||² + β||z||₁
```

[🔝 Back to Top](#-table-of-contents)

---
---

# 🤖 ML & RL FEATURE PIPELINES

---

## E9 · Non-Linear Feature Interactions — ReLU Networks & Attention

**Open with the intuition (15 seconds):**
> "Raw features are additive. Markets are multiplicative. A carry signal and an OFI signal are individually weak; in combination during an FX carry unwind regime, they are strongly predictive. A feed-forward network with ReLU activations learns these multiplicative interactions automatically. Attention mechanisms learn *which* features matter *when* — conditioning on the macro regime state encoded by the autoencoder. This is the architecture used in Citadel's cross-asset alpha models."

---

### Feature Interaction Network

The interaction network maps the feature vector $x$ and latent state $z$ to a scalar alpha prediction $\hat{\alpha}$:

**Layer 1 — Concatenate:**

$$\tilde{x} = [x; z] \in \mathbb{R}^{D + d}$$

**Layer 2 — Regime-conditioned projection:**

$$h = \text{ReLU}(W_h \tilde{x} + b_h)$$

**Layer 3 — Attention over feature groups** (carry, trend, value, micro):

$$\alpha_{g} = \text{softmax}\left(\frac{Q_g K_g^T}{\sqrt{d_k}}\right) V_g$$

**Output:**

$$\hat{\alpha}_t = W_{\text{out}} \cdot \sum_g \alpha_g + b_{\text{out}}$$

**SHAP attribution** for risk committee reporting (Cluster-SHAP for correlated macro features):

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[v(S \cup \{j\}) - v(S)\right]$$

[🔝 Back to Top](#-table-of-contents)

---
---

## E10 · Reinforcement Learning for Dynamic Feature Weighting

**Open with the intuition (15 seconds):**
> "Static feature weights — even well-calibrated ones — cannot adapt to real-time regime transitions. An RL agent operating at HLS's rebalance frequency learns to *weight* features dynamically: amplifying carry weight in trending regimes, suppressing it during risk-off events, and pivoting to microstructure features during macro release windows. The PPO algorithm with a differentiable portfolio constraint layer is the production-grade approach used at Millennium and AQR's dynamic allocation desks."

---

### MDP Formulation for Feature-Weighted Allocation

**State space:**

$$s_t = \left[\underbrace{z_t^{\text{AE}}}_{\text{latent macro state}},\; \underbrace{\text{PCA}_t^{(k^{\*})}}_{\text{factor scores}},\; \underbrace{\text{SSI}_t}_{\text{macro surprise}},\; \underbrace{\text{VPIN}_t}_{\text{adverse selection}},\; \text{position}_t,\; \text{drawdown}_t\right]$$

**Action space:**

$$a_t = \left[w_{\text{carry}},\; w_{\text{trend}},\; w_{\text{value}},\; w_{\text{micro}},\; \text{leverage}_t\right] \in [0,1]^4 \times [0, L_{\max}]$$

**Reward function (risk-adjusted PnL):**

$$r_t = \underbrace{\hat{\alpha}_t \cdot \Delta P_t}_{\text{realized PnL}} - \underbrace{\lambda_{\text{tc}} \cdot |\Delta w_t|}_{\text{turnover cost}} - \underbrace{\lambda_{\text{dd}} \cdot \max(0, -\text{drawdown}_t - D_{\max})}_{\text{drawdown penalty}}$$

**PPO policy update:**

$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right] \quad \epsilon = 0.2$$

**Differentiable constraint layer** (embeds hard risk limits inside the RL action):

$$w_t^{\*} = \arg\min_w \|w - a_t\|_2^2 \quad \text{s.t.} \quad \|w\|_1 \leq L_{\max},\; w_i \geq 0$$

Solved via a differentiable QP layer (cvxpylayers or OSQP-torch), ensuring every output is portfolio-feasible before hitting the execution engine.

```
RL TRAINING CURRICULUM (HLS SYSTEMATIC MACRO)

  Phase 1 (0–200 episodes)      Learn feature regime mapping (high λ_tc)
  Phase 2 (200–1000 episodes)   Learn carry/trend switching (reduce λ_tc)
  Phase 3 (1000–3000 episodes)  Learn drawdown management (full reward)
  Phase 4 (3000+ episodes)      Live paper-trading evaluation (out-of-sample)
```

[🔝 Back to Top](#-table-of-contents)

---
---

# 🏭 PRODUCTION VALIDATION

---

## E11 · Signal Validation, IC Decay & Walk-Forward Protocols

**Open with the intuition (15 seconds):**
> "At HLS, no feature enters production without passing three gates: (1) a statistically significant IC on a walk-forward out-of-sample window, (2) a monotonic IC decay profile (IC should not increase with longer prediction horizons — that is lookahead), and (3) a transaction-cost-adjusted Sharpe above 1.0 net of realistic execution costs. The Bonferroni-corrected t-statistic guards against data mining; the Deflated Sharpe Ratio (DSR) of Bailey & López de Prado guards against backtest overfitting."

---

### Information Coefficient (IC) Validation

$$\text{IC}_t = \text{Spearman}\left(\text{signal}_{t}, r_{t+h}\right)$$

**Walk-forward protocol:**
- Training window: 252 days (expanding)
- Test window: 63 days (rolling)
- Minimum tests: 8 non-overlapping test periods

**IC t-statistic (corrected for autocorrelation):**

$$t_{\text{IC}} = \frac{\bar{\text{IC}}}{\sigma_{\text{IC}} / \sqrt{N_{\text{eff}}}} \quad N_{\text{eff}} = \frac{N}{1 + 2\sum_{k=1}^{K} \rho_k}$$

**Bonferroni correction for $M$ signals tested:**

$$\alpha_{\text{corrected}} = \frac{0.05}{M}$$

---

### Deflated Sharpe Ratio (Bailey & López de Prado)

$$\text{DSR} = \Phi\left[\frac{(\widehat{SR} - SR^{\*}) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{SR}^2}}\right]$$

where $SR^{\*}$ is the expected maximum Sharpe under the null (derived from the number of trials $M$), $\hat{\gamma}_3$ is skewness, and $\hat{\gamma}_4$ is excess kurtosis of the strategy returns.

**HLS minimum bar:** DSR ≥ 0.95 before any strategy enters the live book.

---

### IC Decay Profile

A valid predictive signal should show monotonically decaying IC as the prediction horizon $h$ extends:

$$\text{IC}(h) \sim IC_0 \cdot e^{-\kappa h}$$

where $\kappa$ is the signal decay rate. **Red flag:** if IC(3-day) > IC(1-day), the feature has lookahead or is measuring a mean-reversion artifact rather than a true predictive signal.

```
SIGNAL VALIDATION CHECKLIST (HLS Production Gate)

  ✓  Walk-forward IC t-stat > 2.6 (Bonferroni corrected)
  ✓  IC Sharpe (IC / σ_IC) > 0.5 annualized
  ✓  Monotonically decaying IC(h) profile — no uptick at h > 5 days
  ✓  DSR ≥ 0.95 (Bailey-López de Prado)
  ✓  Transaction-cost-adjusted Sharpe > 1.0 net of realistic half-spread + impact
  ✓  No structural breaks in rolling IC over 2+ year window
  ✓  Feature computed on knowledge_time only — zero lookahead
  ✓  VPIN-conditioned backtest (execution halted when VPIN > 0.7)
```

[🔝 Back to Top](#-table-of-contents)

---
---

# Quick-Reference Equation Sheet

```
══════════════════════════════════════════════════════════════════════════════
CLASSICAL MACRO FEATURES
══════════════════════════════════════════════════════════════════════════════

FX Carry (vol-adjusted):
  Carry_vol = (r_foreign - r_domestic) / σ_{22d}

Commodity Roll Yield:
  RollYield = (F1 - F2) / F2 · (252 / Δ_contract)   [annualized]

TSMOM Signal:
  S_i = sign(Σ r_{t-L:t}) · μ_L / σ_L

Gram-Schmidt Orthogonalization (CSMOM ⊥ TSMOM):
  CSMOM_orth = CSMOM - Cov(CSMOM, TSMOM)/Var(TSMOM) · TSMOM

Standardized Surprise Index:
  SSI_k = (Actual_k - Consensus_k) / σ_k^{historical}

══════════════════════════════════════════════════════════════════════════════
MICROSTRUCTURE FEATURES
══════════════════════════════════════════════════════════════════════════════

Multi-Level OFI:
  OFI_t^(L) = Σ_ℓ exp(-λℓ) · (OF_{b,t}^(ℓ) - OF_{a,t}^(ℓ))   [λ≈0.5]

VPIN:
  VPIN_t = Σ_τ |B_τ - S_τ| / (n · V)

══════════════════════════════════════════════════════════════════════════════
DIMENSIONALITY REDUCTION
══════════════════════════════════════════════════════════════════════════════

PCA Optimal k:
  k* = min{k : Σ_j^k λ_j / Σ_j^K λ_j ≥ 0.90}

RMT Noise Threshold:
  λ+ = σ²(1 + 1/√Q)²   [Q = T/N]

Autoencoder (sparse):
  z = ReLU(W3·ReLU(W2·ReLU(W1·x)))
  L_AE = ||x - x̂||² + β||z||₁   [β = 1e-3]

VAE ELBO:
  L_VAE = E[log p_θ(x|z)] - D_KL(q_φ(z|x) || N(0,I))

══════════════════════════════════════════════════════════════════════════════
RL FEATURE WEIGHTING
══════════════════════════════════════════════════════════════════════════════

PPO Clip Objective:
  L^CLIP = E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]   [ε=0.2]

Reward:
  r_t = α_t·ΔP_t - λ_tc·|Δw_t| - λ_dd·max(0, -dd_t - D_max)

══════════════════════════════════════════════════════════════════════════════
PRODUCTION VALIDATION
══════════════════════════════════════════════════════════════════════════════

IC t-stat (autocorr-corrected):
  t_IC = IC_bar / (σ_IC / √N_eff)   N_eff = N / (1 + 2Σρ_k)

Deflated Sharpe Ratio:
  DSR = Φ[(SR_hat - SR*) · √(T-1) / √(1 - γ3·SR + (γ4-1)/4·SR²)]

IC Decay Model:
  IC(h) ≈ IC_0 · exp(-κ·h)   [κ = signal decay rate]
```

[🔝 Back to Top](#-table-of-contents)

---

*Last updated: June 2026 · HLS Trading · Systematic Macro Feature Engineering — FX · Commodities · Futures · Rates*

[↩️ Back to README.md](../README.md)
