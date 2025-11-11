# Neural Network Architecture for Adaptive Gating and Early Halting

## Overview

Both **Adaptive Gating** and **Early Halting** use the **EXACT SAME neural network architecture**. The only difference is:
- **Adaptive Gating**: Uses output as continuous weights
- **Early Halting**: Uses output as halt probabilities (binary during inference)

---

## The Neural Network (Gate Network / Halt Network)

### Architecture

```python
nn.Sequential(
    nn.Linear(C * 2, C),    # Layer 1: Input layer
    nn.ReLU(),              # Activation
    nn.Linear(C, C),        # Layer 2: Output layer
    nn.Sigmoid()            # Final activation
)
```

Where `C = 64` (number of EEG channels)

### Detailed Breakdown

#### **Layer 1: Input Layer**
```python
nn.Linear(C * 2, C)
# Input size: 128 (64 channels × 2 statistics)
# Output size: 64 (one per channel)
# Parameters: (128 × 64) + 64 bias = 8,256 parameters
```

**What it does:**
- Takes concatenated channel statistics (mean + std) for all 64 channels
- Input shape: (batch_size, 128)
  - First 64 values: channel means
  - Last 64 values: channel stds
- Computes: `output = W₁ × input + b₁`
- Produces 64 intermediate features

#### **ReLU Activation**
```python
nn.ReLU()
# Applies: f(x) = max(0, x)
```

**What it does:**
- Non-linear activation
- Zeros out negative values
- Keeps positive values unchanged

#### **Layer 2: Output Layer**
```python
nn.Linear(C, C)
# Input size: 64
# Output size: 64 (one per channel)
# Parameters: (64 × 64) + 64 bias = 4,160 parameters
```

**What it does:**
- Takes 64 intermediate features from Layer 1
- Produces 64 output values (one per channel)
- Computes: `output = W₂ × hidden + b₂`

#### **Sigmoid Activation**
```python
nn.Sigmoid()
# Applies: f(x) = 1 / (1 + e^(-x))
# Output range: [0, 1]
```

**What it does:**
- Squashes output to [0, 1] range
- Interprets as:
  - **Adaptive Gating**: Gate strength (0 = fully blocked, 1 = fully open)
  - **Early Halting**: Halt probability (0 = keep using, 1 = stop using)

---

## Total Parameters

**Layer 1**: 128 × 64 + 64 = **8,256 parameters**
**Layer 2**: 64 × 64 + 64 = **4,160 parameters**

**Total**: **12,416 parameters**

---

## Input Statistics Computation

### Step 1: Extract EEG Data
```python
x.shape = (batch_size, 1, 64, 769)
# batch_size = number of trials in batch
# 1 = input channel dimension (for Conv2d)
# 64 = EEG channels
# 769 = timepoints
```

### Step 2: Compute Statistics per Channel
```python
x_squeeze = x.squeeze(1)  # Remove dim=1 → (batch_size, 64, 769)

# Compute mean for each of 64 channels
ch_mean = x_squeeze.mean(dim=2)  # → (batch_size, 64)

# Compute std for each of 64 channels
ch_std = x_squeeze.std(dim=2)    # → (batch_size, 64)
```

**What this does:**
- For each trial and each channel, compute:
  - **Mean**: Average amplitude across all 769 timepoints
  - **Std**: Standard deviation across all 769 timepoints
- This gives you 2 numbers per channel = 128 total features

### Step 3: Concatenate Statistics
```python
stats = torch.cat([ch_mean, ch_std], dim=1)
# → (batch_size, 128)
# First 64 values: means of channels 1-64
# Last 64 values: stds of channels 1-64
```

### Step 4: Pass Through Neural Network
```python
gates = self.gate_net(stats)  # → (batch_size, 64)
# or
halt_probs = self.halt_net(stats)  # → (batch_size, 64)
```

---

## Example with Real Numbers

Let's say we have:
- **Batch size**: 32 trials
- **Channels**: 64 EEG channels
- **Timepoints**: 769

### Forward Pass

**Input EEG data:**
```
x.shape = (32, 1, 64, 769)
```

**Step 1: Compute statistics**
```python
x_squeeze = x.squeeze(1)         # (32, 64, 769)
ch_mean = x_squeeze.mean(dim=2)  # (32, 64)
ch_std = x_squeeze.std(dim=2)    # (32, 64)
stats = torch.cat([ch_mean, ch_std], dim=1)  # (32, 128)
```

Example values for one trial:
```
Trial 0:
  Channel Fz mean: 0.023
  Channel Fz std:  1.234
  Channel Cz mean: -0.145
  Channel Cz std:  0.892
  ... (64 channels total)

stats[0] = [0.023, -0.145, ..., (64 means), 1.234, 0.892, ..., (64 stds)]
          = 128 values total
```

**Step 2: Layer 1 (Linear)**
```python
hidden = W1 @ stats + b1
# W1 shape: (64, 128)
# stats shape: (32, 128)
# hidden shape: (32, 64)
```

**Step 3: ReLU**
```python
hidden = max(0, hidden)  # (32, 64)
```

**Step 4: Layer 2 (Linear)**
```python
output = W2 @ hidden + b2
# W2 shape: (64, 64)
# output shape: (32, 64)
```

**Step 5: Sigmoid**
```python
gates = sigmoid(output)  # (32, 64)
# Each value in [0, 1]
```

**Result:**
```
gates[0] = [0.95, 0.87, 0.23, 0.56, ...]  # 64 gate values for trial 0
gates[1] = [0.82, 0.91, 0.15, 0.67, ...]  # 64 gate values for trial 1
... (32 trials total)
```

---

## Why This Architecture?

### Design Choices

1. **Input = Mean + Std (2 statistics per channel)**
   - **Mean**: Captures DC offset / baseline shift
   - **Std**: Captures signal variability / quality
   - These 2 statistics summarize channel behavior efficiently

2. **2-Layer MLP (Multi-Layer Perceptron)**
   - Simple and effective
   - First layer: Learns combinations of channel statistics
   - Second layer: Maps to per-channel decisions

3. **Hidden dimension = C (64)**
   - Same as number of channels
   - Not too many parameters (avoids overfitting)
   - Enough capacity to learn channel relationships

4. **Sigmoid output**
   - Forces output to [0, 1]
   - Naturally interpretable as probabilities/weights

---

## Adaptive Gating vs Early Halting: Same Network, Different Use

### Adaptive Gating
```python
def forward(self, x):
    gates = self.compute_gates(x)  # (batch_size, 64)
    x = x * gates.view(-1, 1, gates.size(1), 1)
    return super().forward(x)
```

**Uses gates as continuous weights:**
- Gate = 0.95 → channel contributes 95%
- Gate = 0.10 → channel contributes 10%
- All channels still flow through network

### Early Halting
```python
def forward(self, x):
    halt_probs = self.compute_halting_probs(x)  # (batch_size, 64)

    if self.training:
        active_mask = 1.0 - halt_probs  # Soft (continuous)
    else:
        active_mask = (halt_probs < 0.5).float()  # Hard (binary)

    x = x * active_mask.view(-1, 1, active_mask.size(1), 1)
    return super().forward(x)
```

**Training (soft):**
- halt_prob = 0.95 → active = 0.05 (channel almost off)
- halt_prob = 0.10 → active = 0.90 (channel mostly on)

**Inference (hard with threshold=0.5):**
- halt_prob = 0.95 → active = 0 (OFF)
- halt_prob = 0.10 → active = 1 (ON)
- Binary decision!

---

## Initialization

### Adaptive Gating
```python
with torch.no_grad():
    self.gate_net[-2].bias.fill_(2.0)
```

**Why?**
- Layer 2 bias initialized to 2.0
- After sigmoid: `sigmoid(2.0) ≈ 0.88`
- Starts with most channels "open" (high gate values)
- Network learns to close unimportant ones during training

### Early Halting
```python
# No special initialization
# Uses default PyTorch initialization
```

**Why?**
- Starts with random weights
- Network learns which channels to halt during training

---

## Summary Table

| Aspect | Value |
|--------|-------|
| **Architecture** | 2-layer MLP |
| **Input size** | 128 (64 channels × 2 stats) |
| **Hidden size** | 64 |
| **Output size** | 64 (one per channel) |
| **Layer 1** | Linear(128 → 64) + ReLU |
| **Layer 2** | Linear(64 → 64) + Sigmoid |
| **Total parameters** | 12,416 |
| **Input statistics** | Mean + Std per channel |
| **Output range** | [0, 1] |
| **Used by** | Adaptive Gating & Early Halting |

---

## Code Reference

### Adaptive Gating (lines 55-66 in notebook)
```python
self.gate_net = nn.Sequential(
    nn.Linear(C * 2, C),
    nn.ReLU(),
    nn.Linear(C, C),
    nn.Sigmoid()
)

with torch.no_grad():
    self.gate_net[-2].bias.fill_(2.0)
```

### Early Halting (lines 67-74 in notebook)
```python
self.halt_net = nn.Sequential(
    nn.Linear(C * 2, C),
    nn.ReLU(),
    nn.Linear(C, C),
    nn.Sigmoid()
)
# No special initialization
```

**Identical architecture!**

---

## Visualization

```
Input EEG: (batch, 1, 64, 769)
           ↓
    [Compute Stats]
           ↓
   (batch, 128) ← [mean_1, ..., mean_64, std_1, ..., std_64]
           ↓
    [Linear 128→64]
           ↓
        [ReLU]
           ↓
     [Linear 64→64]
           ↓
      [Sigmoid]
           ↓
   (batch, 64) ← gates/halt_probs (one per channel)
           ↓
      [Multiply]
           ↓
  Gated EEG → Main Network
```

---

## Key Takeaway

The neural network is a **simple 2-layer fully-connected network** that:
1. Takes channel-wise statistics (mean + std) as input
2. Learns to predict importance/halt for each channel
3. Has only **12,416 parameters** (very lightweight!)
4. Is **shared** between Adaptive Gating and Early Halting (same architecture)