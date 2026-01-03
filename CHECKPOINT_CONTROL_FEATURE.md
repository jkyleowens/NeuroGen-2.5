# Checkpoint Control Feature

## Overview

Added a command-line option to enable/disable checkpoint saving during training. This is useful for faster testing and experimentation when you don't need to save model weights.

## Changes Made

### 1. New Configuration Option
Added `save_checkpoint` boolean to `TrainingConfig`:

```python
@dataclass
class TrainingConfig:
    # ... other config options ...
    
    # Checkpoint parameters
    checkpoint_dir: str = "checkpoints"
    save_checkpoint: bool = True  # Enable/disable checkpoint saving
    # Note: Checkpoints are now only saved at completion or on interruption
```

### 2. New Command-Line Argument

```bash
python train_slimpajama.py --no-checkpoint
```

**Arguments:**
- `--no-checkpoint`: Disable checkpoint saving (default: checkpoints are saved)

## Usage

### Normal Training (With Checkpoints)
```bash
# Checkpoints saved at completion and on Ctrl+C
python train_slimpajama.py --test
```

### Fast Testing (Without Checkpoints)
```bash
# No checkpoint saving - faster for quick experiments
python train_slimpajama.py --test --no-checkpoint
```

### Long Training (With Checkpoints)
```bash
# Checkpoints saved at completion and on Ctrl+C
python train_slimpajama.py --max-chunks 1000
```

### Long Training (Without Checkpoints)
```bash
# Useful if you're only interested in metrics, not the model weights
python train_slimpajama.py --max-chunks 1000 --no-checkpoint
```

## Behavior

### When Checkpoints are Enabled (Default)
- ✅ Checkpoint saved when training completes normally
- ✅ Checkpoint saved when you press Ctrl+C (graceful interruption)
- ✅ Checkpoint saved when an error occurs (attempted recovery)
- 📁 Saved to: `checkpoints/full_model.ckpt`

### When Checkpoints are Disabled (`--no-checkpoint`)
- ⏭️ No checkpoint saved at completion
- ⏭️ No checkpoint saved on Ctrl+C
- ⏭️ No checkpoint saved on error
- 🏃 Faster training (no I/O overhead)
- 📊 Visualizations and logs still saved

## Benefits

1. **Faster Testing**: Skip expensive checkpoint I/O during quick experiments
2. **Disk Space**: Don't create large checkpoint files when you don't need them
3. **Flexibility**: Easy to enable/disable per training run
4. **Safety**: Checkpoints enabled by default for production training

## Display

The configuration now shows checkpoint status:

```
================================================================================
🧠 NeuroGen 2.0 - SlimPajama Training
================================================================================
Configuration:
  GPU Device: 0
  Tokens per chunk: 1024
  Max sequence length: 256
  Max chunks: 5
  Vocab size: 32000
  Save checkpoints: ✓ Enabled
================================================================================
```

Or with `--no-checkpoint`:

```
  Save checkpoints: ✗ Disabled
```

## Examples

### Scenario 1: Quick Debugging
```bash
# Test if training runs without errors (no checkpoint needed)
python train_slimpajama.py --test --no-checkpoint
```

### Scenario 2: Hyperparameter Search
```bash
# Try different learning rates without saving models
for lr in 0.001 0.0001 0.00001; do
    echo "Testing learning rate: $lr"
    python train_slimpajama.py --max-chunks 50 --no-checkpoint
done
```

### Scenario 3: Production Training
```bash
# Full training with checkpoint saving (default)
python train_slimpajama.py --max-chunks 10000
```

### Scenario 4: Resume from Checkpoint
```bash
# Load existing checkpoint and continue training
# (checkpoint loading is automatic if checkpoints/full_model.ckpt exists)
python train_slimpajama.py --max-chunks 20000
```

## Technical Details

### Checkpoint Timing
Checkpoints are saved **only** at:
1. **Training completion** (all chunks processed)
2. **Graceful interruption** (Ctrl+C)
3. **Error recovery** (exception occurred)

**NOT saved during training** to avoid I/O overhead and disk thrashing.

### File Size
- Typical checkpoint: **~500MB - 2GB** (depending on model size)
- With `--no-checkpoint`: **0 bytes** (no checkpoint created)
- Visualizations: **~50KB per chart** (always saved)

### Performance Impact

| Aspect | With Checkpoints | Without Checkpoints |
|--------|-----------------|---------------------|
| Training Speed | Normal | Normal |
| Completion Time | +5-30 seconds (saving) | Immediate |
| Disk Writes | 1 checkpoint file | None |
| Memory Usage | Same | Same |

## Summary

- ✅ **Added `--no-checkpoint` flag** for faster testing
- ✅ **Checkpoints enabled by default** for safety
- ✅ **Clear visual feedback** in configuration display
- ✅ **Respects setting** at completion, interruption, and error
- ✅ **Backward compatible** (existing commands work unchanged)

Use `--no-checkpoint` for quick experiments, omit it for production training!
