# Top-K Sampling Analysis and Fix

## The Problem

Your model is currently using **top-k=50 sampling during training**, which is contributing to the poor results in two ways:

### 1. Top-K Sampling is Normal for NLP Inference
- ✅ **Standard practice** for chat/generation (GPT, Llama, etc.)
- ✅ Makes output more creative and less repetitive
- ✅ Balances between coherence and diversity

### 2. Top-K Sampling is BAD for Training
- ❌ **Hides the true accuracy** of the model
- ❌ **Unstable training signal** due to randomness
- ❌ **Masks underlying issues** (exploded or saturated weights)

## Current Behavior

Your training loop does this:
```python
# train_slimpajama.py, line ~436
loss, accuracy, predicted_token_id = self.model.train_step(context_ids, [target_id])
```

Internally, `train_step` uses **top-k=50 sampling** to pick the predicted token. This means:

### Scenario A: Flat Distribution (Exploded Weights)
```
Expected token: "the" (ID: 278)
Model output probabilities:
  Token #278 "the":      0.002% ← Correct answer
  Token #1053 "and":     0.002%
  Token #5823 "Zhejiang": 0.002% ← Gets sampled randomly
  ... (all 32,000 tokens have ~equal probability)
```
**Result**: Accuracy = 0% because sampling picks random tokens from a flat distribution.

### Scenario B: Saturated Distribution (Saturated Weights)
```
Expected token: "the" (ID: 278)
Model output probabilities:
  Token #5823 "Zhejiang": 99.8% ← Wrong, but dominates
  Token #278 "the":        0.1% ← Correct, but never sampled
  Token #1053 "and":       0.05%
```
**Result**: Accuracy = 0% because the model is overconfident in the wrong token.

## The Fix

### Use GREEDY decoding during training

**Greedy decoding** always picks the token with the **highest probability**:
- ✅ Deterministic (same input → same output)
- ✅ Accuracy metric is meaningful
- ✅ Can diagnose if model is learning
- ✅ Stable training signal

**Top-k sampling** should ONLY be used during **inference** (chat, generation).

## Implementation

### Option 1: Modify the C++ decoder (Recommended)

Add a parameter to `train_step` to override the sampling strategy:

```cpp
// In python_binding.cpp or neurogen_bindings.cpp
// When calling train_step, force GREEDY sampling:
decoder_config.strategy = GPUDecoder::SamplingStrategy::GREEDY;
```

### Option 2: Create a separate training decoder

Create two decoder instances:
- `training_decoder` with GREEDY sampling
- `inference_decoder` with TOP_K sampling

Use the training decoder in `train_step`.

### Option 3: Simplify (Quick Fix)

In `train_slimpajama.py`, instead of using `train_step` to get the predicted token:

```python
# OLD (uses top-k sampling):
loss, accuracy, predicted_token_id = self.model.train_step(context_ids, [target_id])

# NEW (use greedy manually):
loss, accuracy, _ = self.model.train_step(context_ids, [target_id])

# Get greedy prediction separately:
output_probs = self.model.forward(context_ids)
predicted_token_id = output_probs.argmax()  # Greedy = highest probability
```

## Expected Results After Fix

With greedy decoding + reduced learning rates (0.0001-0.005), you should see:

### First 50 Chunks:
- **Accuracy**: 0% → 5-10%
- **Loss**: 10.0 → 7.0
- **Predictions**: Random → Common words ("the", ".", "and", "a")

### After 500 Chunks:
- **Accuracy**: 10-20%
- **Loss**: 7.0 → 4.0
- **Predictions**: Common words → Contextually relevant words

### After 5000 Chunks:
- **Accuracy**: 30-50%
- **Loss**: 4.0 → 2.5
- **Predictions**: Contextually relevant → Often correct

## Verification

Run the diagnostic script:
```bash
python diagnose_sampling.py
```

This will:
1. Show you how random the current predictions are
2. Explain why top-k sampling is problematic
3. Recommend next steps

## Summary

| Aspect | Training (Should Use) | Inference (Can Use) |
|--------|----------------------|---------------------|
| **Strategy** | GREEDY | TOP_K or TOP_P |
| **Why** | Deterministic, measurable accuracy | Creative, diverse output |
| **Top-K Value** | N/A (always pick #1) | 40-50 |
| **Temperature** | 1.0 (no scaling) | 0.7-1.0 |

**The fix**: Switch training to greedy decoding, keep top-k for chat/inference.
