#!/usr/bin/env python3
"""
Diagnose Sampling Strategy Issues

This script analyzes whether top-k sampling is masking or amplifying
the model's learning problems by examining the actual probability distributions.
"""

import sys
import numpy as np
try:
    import neurogen
    import sentencepiece as spm
except ImportError as e:
    print(f"Error: {e}")
    print("Run: pip install sentencepiece")
    sys.exit(1)


def analyze_probability_distribution(model, tokenizer, test_inputs):
    """Analyze the probability distributions the model outputs"""
    
    print("="*80)
    print("PROBABILITY DISTRIBUTION ANALYSIS")
    print("="*80)
    
    for test_text in test_inputs:
        print(f"\n📝 Input: '{test_text}'")
        
        # Tokenize
        token_ids = tokenizer.EncodeAsIds(test_text)
        print(f"   Tokens: {token_ids}")
        
        if not token_ids:
            print("   ⚠️  Empty tokenization, skipping")
            continue
        
        # Get raw logits/probabilities (we need to modify the model to expose these)
        # For now, let's run the model multiple times and see if we get the same prediction
        predictions = []
        for trial in range(20):
            try:
                # This uses top-k=50 sampling
                predicted_token = model.generate_next_token(token_ids)
                predicted_text = tokenizer.DecodeIds([predicted_token])
                predictions.append((predicted_token, predicted_text))
            except Exception as e:
                print(f"   ❌ Error during generation: {e}")
                break
        
        if not predictions:
            continue
        
        # Analyze prediction consistency
        unique_predictions = {}
        for token_id, text in predictions:
            key = (token_id, text)
            unique_predictions[key] = unique_predictions.get(key, 0) + 1
        
        print(f"\n   Prediction Distribution (20 trials with top-k=50):")
        sorted_preds = sorted(unique_predictions.items(), key=lambda x: x[1], reverse=True)
        
        for (token_id, text), count in sorted_preds[:10]:  # Show top 10
            percentage = (count / 20) * 100
            bar = "█" * int(percentage / 5)
            print(f"   [{token_id:5d}] '{text:15s}' │ {bar:20s} {percentage:5.1f}%")
        
        # Diagnose the issue
        print(f"\n   🔍 Diagnosis:")
        if len(unique_predictions) == 1:
            print("   ⚠️  PROBLEM: Model always predicts the same token!")
            print("   → Weights are likely saturated (overconfident)")
            print("   → Top-k sampling has no effect because one token dominates")
        elif len(unique_predictions) >= 15:
            print("   ⚠️  PROBLEM: Predictions are highly random!")
            print("   → Probability distribution is likely flat")
            print("   → Top-k sampling is just picking random tokens")
        elif sorted_preds[0][1] >= 15:
            print("   ✓ Model has a strong preference (good)")
            print("   → But we need to check if it's the RIGHT token")
        else:
            print("   ⚠️  Model is uncertain (moderate randomness)")
            print("   → Could be normal early in training, or could indicate exploded weights")


def compare_greedy_vs_topk():
    """Compare greedy decoding vs top-k sampling"""
    
    print("\n" + "="*80)
    print("RECOMMENDATION: USE GREEDY DECODING DURING TRAINING")
    print("="*80)
    
    print("""
📊 Why Top-K Sampling is Problematic During Training:

1. **Masking Real Accuracy**: With top-k=50, even if the correct token
   is ranked #2 with 49% probability, you might sample token #15 with 0.8%
   probability. Your accuracy appears worse than it actually is.

2. **Unstable Training Signal**: The model gets different feedback each time
   because the sampling is random. This makes it harder to learn.

3. **Can't Diagnose Issues**: You can't tell if the model is:
   - Putting 99% probability on the wrong token (saturated weights)
   - Putting equal probability on all tokens (exploded weights)
   - Actually learning but getting unlucky with sampling

✅ SOLUTION: Use GREEDY decoding during training
   - Always pick the token with highest probability
   - Accuracy metric becomes meaningful
   - Training signal is deterministic
   - You can see if the model is actually learning

   Use top-k/top-p sampling ONLY during inference for creative text generation.
""")


def main():
    print("🔍 Diagnosing Top-K Sampling Issues\n")
    
    # Load model
    print("Loading model...")
    try:
        config = neurogen.Config()
        config.vocab_size = 32000
        config.embedding_dim = 768
        config.gpu_device_id = 0
        model = neurogen.NeuroGen(config)
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   This is okay - we can still show you the analysis")
        model = None
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load("tokenizer/nlp_agent_tokenizer.model")
        print("✓ Tokenizer loaded\n")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
    
    if model:
        # Test inputs
        test_inputs = [
            "The cat sat on the",
            "Hello, my name is",
            "In the year 2024,",
            "The quick brown fox",
        ]
        
        analyze_probability_distribution(model, tokenizer, test_inputs)
    
    # Show recommendations
    compare_greedy_vs_topk()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Modify the training loop to use GREEDY decoding:
   - In train_slimpajama.py, when creating the model
   - Set sampling_strategy = GREEDY instead of TOP_K

2. Restart training from scratch with the reduced learning rates

3. Monitor the first 50 chunks - you should see:
   - Accuracy gradually increase from 0% to 5-10%
   - Loss gradually decrease
   - Predictions become less random

4. Only use top-k/top-p sampling in chat_interface.py for inference
""")


if __name__ == "__main__":
    main()
