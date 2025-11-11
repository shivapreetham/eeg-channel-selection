# Gating Methods: Theoretical Differences Explained

## Core Concept

All gating methods aim to **learn which EEG channels are important**, but they differ in **HOW and WHEN** they decide channel importance.

---

## 1. Baseline (No Gating)

```
Input (64 channels) → EEG-ARNN → Output
```

**What it does:**
- Uses ALL channels equally
- No channel weighting or selection
- Pure CNN + Graph Convolution

**Key characteristic:**
- No mechanism to suppress unimportant channels

---

## 2. Static Gating

```
Input (64 channels) → [Multiply by FIXED gates] → EEG-ARNN → Output
                              ↑
                    Learnable parameters (C,)
                    Same for ALL inputs
```

**What it does:**
- Learns **one fixed gate value per channel** (64 gate values total)
- Gate values are **learned during training** but **same for every trial**
- Gates are parameters: `gate = sigmoid(gate_logit)` where `gate_logit` is learnable
- Multiplies input: `x = x * gate_values`

**Key characteristic:**
- **Input-independent**: Gate values don't change based on the input
- Each channel gets a fixed importance weight
- Gates learned via backpropagation + L1 regularization

**Loss function:**
```
Loss = CrossEntropy + λ * mean(|gate_values|)
                       ↑
              Encourages sparsity (some gates → 0)
```

**Example:**
- Channel Fz might always get gate = 0.95 (important)
- Channel T5 might get gate = 0.12 (not important)
- These values are SAME for every single trial

---

## 3. Adaptive Gating

```
Input (64 channels) → [Compute statistics] → Gate Network → Gates (64 values)
        ↓                                                      ↓
        └─────────────────[Multiply]───────────────────────────┘
                              ↓
                         EEG-ARNN → Output
```

**What it does:**
- Learns a **neural network** that computes gate values from the input
- Gate values are **different for each trial** based on input features
- Steps:
  1. Compute channel statistics: `mean` and `std` for each channel
  2. Concatenate: `stats = [ch_mean, ch_std]` → shape (C*2,)
  3. Pass through gate network:
     ```
     gates = Sigmoid(Linear(ReLU(Linear(stats))))
     ```
  4. Multiply input: `x = x * gates`

**Key characteristic:**
- **Input-dependent**: Different trials get different gate values
- Adapts to the current input signal quality
- Gate network is learnable

**Example:**
- Trial 1 (clean signal): Fz gets gate = 0.95, T5 gets gate = 0.85
- Trial 2 (noisy signal): Fz gets gate = 0.60, T5 gets gate = 0.10
- Gates **change dynamically** based on each trial's characteristics

---

## 4. Early Halting

```
Input (64 channels) → [Compute halt probs] → Halt Network → Halt probabilities (64)
        ↓                                                           ↓
        ↓                                                   [Threshold decision]
        ↓                                                           ↓
        └─────────────────[Multiply by active_mask]─────────────────┘
                              ↓
                         EEG-ARNN → Output
```

**What it does:**
- Learns a **neural network** that predicts **halting probability** for each channel
- Channels with high halt probability are "turned off"
- Steps:
  1. Compute channel statistics: `mean` and `std` for each channel
  2. Pass through halt network:
     ```
     halt_probs = Sigmoid(Linear(ReLU(Linear(stats))))
     ```
  3. **During training**: `active_mask = 1.0 - halt_probs` (soft masking)
  4. **During inference**: `active_mask = (halt_probs < threshold).float()` (hard masking)
  5. Multiply input: `x = x * active_mask`

**Key characteristic:**
- **Binary decision** during inference (channel is ON or OFF)
- **Progressive dropout**: Channels are gradually "halted" based on their usefulness
- Trained with penalty: encourages model to use fewer channels

**Loss function:**
```
Loss = CrossEntropy + penalty * mean(active_channels)
                       ↑
              Encourages using fewer channels
```

**Example:**
- During training:
  - Trial 1: Fz active_mask = 0.95, T5 active_mask = 0.15
  - Trial 2: Fz active_mask = 0.80, T5 active_mask = 0.05

- During inference (threshold = 0.5):
  - Trial 1: Fz ON (halt_prob=0.05 < 0.5), T5 OFF (halt_prob=0.85 > 0.5)
  - Trial 2: Fz ON (halt_prob=0.20 < 0.5), T5 OFF (halt_prob=0.95 > 0.5)

---

## Key Differences Summary

| Aspect | Static Gating | Adaptive Gating | Early Halting |
|--------|--------------|-----------------|---------------|
| **Gate computation** | Fixed learnable parameters | Neural network from input stats | Neural network from input stats |
| **Input-dependent?** | ❌ No | ✅ Yes | ✅ Yes |
| **Gate values** | Continuous [0, 1] | Continuous [0, 1] | Continuous in training, Binary in inference |
| **Number of parameters** | C (64) | 2C² + C (network weights) | 2C² + C (network weights) |
| **Regularization** | L1 on gates | L1 on gates | Penalty on active channels |
| **Objective** | Learn fixed channel importance | Learn input-adaptive weighting | Learn to progressively drop channels |
| **Interpretation** | "Which channels are globally important?" | "Which channels are important for THIS trial?" | "When should we stop using a channel?" |

---

## Mathematical Formulation

### Static Gating
```
g_c ∈ ℝ^C                    (learnable parameters)
gates = σ(g_c)               (sigmoid activation)
x' = x ⊙ gates               (element-wise multiply)
y = EEG-ARNN(x')
Loss = CE(y, target) + λ₁ * ||gates||₁
```

### Adaptive Gating
```
stats = [mean(x), std(x)]    (shape: 2C)
gates = GateNet(stats)       (neural network)
      = σ(W₂ · ReLU(W₁ · stats + b₁) + b₂)
x' = x ⊙ gates               (element-wise multiply, per-sample)
y = EEG-ARNN(x')
Loss = CE(y, target) + λ₁ * mean(|gates|)
```

### Early Halting
```
stats = [mean(x), std(x)]           (shape: 2C)
halt_probs = HaltNet(stats)         (neural network)
           = σ(W₂ · ReLU(W₁ · stats + b₁) + b₂)

Training:   active = 1 - halt_probs (soft)
Inference:  active = (halt_probs < τ) (hard, binary)

x' = x ⊙ active
y = EEG-ARNN(x')
Loss = CE(y, target) + λ₂ * mean(active)
```

Where:
- C = number of channels (64)
- ⊙ = element-wise multiplication
- σ = sigmoid function
- τ = halting threshold (0.5)
- λ₁ = L1 regularization weight (1e-3)
- λ₂ = halting penalty weight (1e-2)

---

## Conceptual Differences

### Static Gating
**Philosophy**: "Some channels are always more important than others"

**Analogy**: Like having favorite tools in a toolbox. You always prefer the hammer over the tiny screwdriver, regardless of the job.

**Use case**: When channel importance is **consistent across all trials**

### Adaptive Gating
**Philosophy**: "Channel importance depends on the current input"

**Analogy**: Like choosing tools based on the job. Use hammer for nails, screwdriver for screws. The tool choice adapts to the task.

**Use case**: When channel quality or relevance **varies between trials**

### Early Halting
**Philosophy**: "Process channels sequentially and stop when not useful"

**Analogy**: Like reading a book chapter by chapter and stopping when you've learned enough. You don't need to read all chapters if the first few give you the answer.

**Use case**: When you want **computational efficiency** by dropping channels during inference

---

## Why Early Halting is Different from Simple Gating

### The Key Distinction: **Intent and Mechanism**

1. **Static/Adaptive Gating**: Soft weighting
   - Multiply input by continuous values [0, 1]
   - All channels still contribute (just with different weights)
   - Goal: Learn importance weights

2. **Early Halting**: Hard dropout
   - Binary decision during inference (ON/OFF)
   - Channels either fully contribute or don't contribute at all
   - Goal: Progressively eliminate channels to save computation

### Training Behavior

**Static/Adaptive Gating**:
```python
# Gates are continuous
gates = [0.95, 0.87, 0.12, 0.56, ...]
x_weighted = x * gates
# All channels still flow through network
```

**Early Halting**:
```python
# Training: soft (like gating)
active_mask_train = 1 - [0.05, 0.13, 0.88, 0.44, ...]
                      = [0.95, 0.87, 0.12, 0.56, ...]

# Inference: HARD decision
halt_probs = [0.05, 0.13, 0.88, 0.44, ...]
threshold = 0.5
active_mask_test = [1, 1, 0, 1, ...]  # Binary!
# Channels 3 is COMPLETELY DROPPED
```

### Different Penalty

**Gating (Static/Adaptive)**:
```python
# Penalty on gate magnitude (sparsity)
loss += lambda * mean(abs(gates))
# Encourages some gates → 0
```

**Early Halting**:
```python
# Penalty on number of active channels
loss += lambda * mean(active_mask)
# Encourages using FEWER channels overall
```

---

## Practical Impact

### Static Gating
- **Best for**: Global channel selection (same channels always important)
- **Channel selection**: Gates directly indicate importance
- **Computational cost**: No extra cost (just multiplication)

### Adaptive Gating
- **Best for**: Trial-specific weighting (different channels important for different trials)
- **Channel selection**: Need to average gates across trials
- **Computational cost**: Extra forward pass through gate network

### Early Halting
- **Best for**: Efficient inference (drop channels early)
- **Channel selection**: Determines which channels can be safely dropped
- **Computational cost**: Can SAVE computation by actually removing channels

---

## Why Your Results Show What They Show

### Static Gating + Gate Selection = -1.05% drop (IMPROVEMENT!)

**Why it works so well:**
1. Static gates learn TRUE global channel importance
2. During channel selection, you pick channels with highest gate values
3. This is **perfectly aligned**: selection criterion matches what the model learned
4. Removing low-gate channels actually helps by removing noise

### Adaptive Gating + Gate Selection = +0.60% drop (small degradation)

**Why it's not as good:**
1. Gates change per trial, so averaging loses information
2. A channel might be important for SOME trials but not others
3. Removing it hurts the trials where it was important

### Early Halting + Gate Selection = +2.83% drop

**Why it's moderate:**
1. Halting network learns to drop channels progressively
2. But it's optimized for binary decisions, not importance ranking
3. Selecting based on "halt probability" is less aligned than static gates

---

## Summary

**Static Gating** = "Learn fixed importance weights"
- One weight per channel
- Same for all inputs
- Best for channel selection

**Adaptive Gating** = "Learn to weight channels based on current input"
- Different weights per input
- Adapts to input quality
- Best for handling variable signal quality

**Early Halting** = "Learn when to stop using channels"
- Probabilistic during training
- Binary during inference
- Best for computational efficiency

The fundamental difference is:
- **Static**: Fixed importance
- **Adaptive**: Input-dependent importance
- **Halting**: Progressive channel dropout with binary decisions
