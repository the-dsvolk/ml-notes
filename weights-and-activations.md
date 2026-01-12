# Weights vs. Activations in Neural Networks

## Quick Summary

| | **Weights** | **Activations** |
|---|------------|-----------------|
| **What** | Learnable parameters (W, b) | Intermediate layer outputs |
| **When** | Fixed after training | Computed during inference |
| **Storage** | Saved in model file | Generated on-the-fly |
| **Data dependency** | None | Depends on input |

---

## Weights

Weights are the **static parameters** learned during training:

```
z = W × x + b
    ↑       ↑
    └───────┴── WEIGHTS (fixed after training)
```

- **W** = weight matrices
- **b** = bias vectors
- Stored in the model file (e.g., `.pt`, `.safetensors`)
- Same values used for every inference

**Memory example (Llama2 7B):**
- FP16: 7B × 2 bytes = ~14 GB
- FP8: 7B × 1 byte = ~7 GB

---

## Activations

Activations are **dynamic intermediate outputs** produced by each layer:

```
Forward Pass:
=============

Input x
   │
   ▼
┌──────────────────────┐
│ z₁ = W₁ × x + b₁     │ ◄── uses weights
│ a₁ = ReLU(z₁)        │ ◄── ACTIVATION (output of layer 1)
└──────────────────────┘
   │
   ▼
┌──────────────────────┐
│ z₂ = W₂ × a₁ + b₂    │ ◄── uses weights + previous activation
│ a₂ = ReLU(z₂)        │ ◄── ACTIVATION (output of layer 2)
└──────────────────────┘
   │
   ▼
 Output
```

- `a₁`, `a₂`, etc. are activations
- Different for every input
- Not stored — recomputed each inference

---

## Key Insight

```
Model file contains:     WEIGHTS only
                              │
                              ▼
During inference:        Weights + Input → ACTIVATIONS → Output
```

**Weights** define *what* the model learned.  
**Activations** are *how* the model processes each specific input.

---

## Why This Matters for Quantization

| Component | Quantization Approach |
|-----------|----------------------|
| **Weights** | Direct — values are known and fixed |
| **Activations** | Requires calibration — values vary per input |

Quantizing weights is straightforward. Quantizing activations requires running the model on representative data to understand their typical range and distribution.

---

## KV Cache (LLMs Only)

For decoder-only transformers (GPT, Llama), there's a third component:

- **KV Cache**: Stores key-value pairs from attention layers
- Speeds up autoregressive token generation
- Grows with sequence length
- Can also be quantized to save memory




