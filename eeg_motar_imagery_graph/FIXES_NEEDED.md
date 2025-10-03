# EEG Motor Imagery Graph CNN - Implementation Fixes

## Summary

I've analyzed the notebook implementation and compared it with the research paper. The implementation is **correctly designed** according to the paper, but has **TensorFlow 2.x compatibility issues** that need to be fixed.

## Status

### ✅ What's Working
- All dependencies are installed (TensorFlow 2.20.0, MNE 1.10.1, Plotly, NetworkX, etc.)
- Architecture design matches the paper specifications
- TFEM (Temporal Feature Extraction Module) works correctly
- Overall model structure is correct

### ❌ What Needs Fixing

#### 1. **DynamicAdjacencyMatrix Layer - Critical Issue**

**Problem:** In Cell 2, line ~117, the `call()` method uses `K.eye()` which creates a new tensor during execution, causing TensorFlow errors when building the model graph.

**Original Code:**
```python
def call(self, inputs):
    degree = K.sum(K.abs(self.adjacency_matrix), axis=1, keepdims=True)
    degree = K.maximum(degree, 1e-8)
    degree_inv_sqrt = K.pow(degree, -0.5)
    normalized_adj = self.adjacency_matrix * degree_inv_sqrt
    normalized_adj = normalized_adj * K.transpose(degree_inv_sqrt)

    # PROBLEM: K.eye() creates tensor during graph execution
    identity = K.eye(self.num_channels)  # ❌ This fails in TensorFlow 2.x
    normalized_adj = normalized_adj + identity
    return normalized_adj
```

**Fixed Code:**
```python
def call(self, inputs):
    degree = tf.reduce_sum(tf.abs(self.adjacency_matrix), axis=1, keepdims=True)
    degree = tf.maximum(degree, 1e-8)
    degree_inv_sqrt = tf.pow(degree, -0.5)
    normalized_adj = self.adjacency_matrix * degree_inv_sqrt
    normalized_adj = normalized_adj * tf.transpose(degree_inv_sqrt)

    # FIX: Use tf.eye() instead of K.eye()
    identity = tf.eye(self.num_channels, dtype=tf.float32)  # ✅ This works
    normalized_adj = normalized_adj + identity
    return normalized_adj
```

#### 2. **GraphConvolution Layer - Minor Issue**

**Problem:** Uses `K.dot()` and `K.batch_dot()` which have compatibility issues.

**Original Code:**
```python
def call(self, inputs):
    features, adjacency = inputs
    support = K.dot(features, self.kernel)  # ❌
    output = K.batch_dot(adjacency, support)  # ❌
    if self.use_bias:
        output = K.bias_add(output, self.bias)
    if self.activation is not None:
        output = self.activation(output)
    return output
```

**Fixed Code:**
```python
def call(self, inputs):
    if isinstance(inputs, list):
        features, adjacency = inputs
    else:
        raise ValueError("GraphConvolution expects a list of [features, adjacency]")

    support = tf.matmul(features, self.kernel)  # ✅
    output = tf.matmul(adjacency, support)  # ✅
    if self.use_bias:
        output = tf.nn.bias_add(output, self.bias)
    if self.activation is not None:
        output = self.activation(output)
    return output
```

#### 3. **CARM Layer - Needs build() method**

**Problem:** CARM initializes sublayers in `__init__` but doesn't have a `build()` method, causing TensorFlow warnings.

**Fixed Code:**
```python
class CARM(layers.Layer):
    def __init__(self, num_channels, output_dim, dropout_rate=0.25, **kwargs):
        super(CARM, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Build components here instead of __init__
        self.adjacency_layer = DynamicAdjacencyMatrix(self.num_channels)
        self.graph_conv = GraphConvolution(self.output_dim, activation='elu')
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(self.dropout_rate)
        super(CARM, self).build(input_shape)

    def call(self, inputs, training=None):
        adjacency = self.adjacency_layer(inputs)
        graph_output = self.graph_conv([inputs, adjacency])
        output = self.batch_norm(graph_output, training=training)
        output = self.dropout(output, training=training)
        return output
```

#### 4. **Cell 8 - Syntax Error**

**Problem:** Duplicate print statement

**Line 27:**
```python
print(f"   • Test on larger datasets (BCI Competition IV, etc.)")
   print(f"   • Cross-subject and cross-session validation")  # ❌ Extra indentation
```

**Fixed:**
```python
print(f"   • Test on larger datasets (BCI Competition IV, etc.)")
print(f"   • Cross-subject and cross-session validation")  # ✅
```

## How to Apply Fixes

### Option 1: Use the Fixed Components File

I've created a corrected implementation in:
```
fixed_graph_cnn_components.py
```

This file contains all the fixed layers with proper TensorFlow 2.x compatibility.

### Option 2: Manual Fix in Notebook

Replace Cell 2 in the notebook with the contents from `fixed_graph_cnn_components.py`.

## Validation

I've tested the fixed implementation and confirmed:

✅ GraphConvolution layer works correctly
✅ DynamicAdjacencyMatrix layer works correctly
✅ CARM layer works correctly
✅ TFEM layer works correctly
✅ Complete EEG-ARNN model can be built and compiled
✅ Forward pass executes successfully

## Architecture Verification

The implementation **correctly matches the paper**:

### From the Paper (Sun et al., 2023):
- ✅ 3 TFEM-CARM blocks
- ✅ Progressive filter sizes: 16 → 32 → 64
- ✅ Kernel sizes: 16, 8, 4
- ✅ Dropout rates: 0.25, 0.3, 0.4
- ✅ Dynamic adjacency matrix (Equation 15, 17)
- ✅ Graph convolution (Equation 14, 18)
- ✅ Edge Selection (ES) method (Equation 19)
- ✅ Aggregation Selection (AS) method (Equation 20)
- ✅ End-to-end trainable architecture

## Next Steps

1. **Apply the fixes** to Cell 2 in the notebook
2. **Fix the syntax error** in Cell 8
3. **Run all cells** sequentially
4. The notebook will:
   - Load PhysioNet EEG data
   - Preprocess it according to the paper
   - Train the EEG-ARNN model
   - Perform channel selection (ES and AS)
   - Generate comprehensive visualizations

## Paper Reference

**Title:** "Graph Convolution Neural Network Based End-to-End Channel Selection and Classification for Motor Imagery Brain–Computer Interfaces"

**Authors:** Biao Sun, Zhengkun Liu, Zexu Wu, Chaoxu Mu, and Ting Li

**Published:** IEEE Transactions on Industrial Informatics, Vol. 19, No. 9, September 2023

**DOI:** 10.1109/TII.2022.3227736

## Additional Notes

- The notebook uses **PhysioNet dataset** (subjects 1-3, runs 6, 10, 14)
- Training will take time depending on your hardware (GPU recommended)
- The model implements state-of-the-art Graph Neural Network techniques for EEG analysis
- All mathematical formulations match the research paper exactly

---

**Created:** October 4, 2025
**Last Updated:** October 4, 2025
**Status:** Ready for implementation
