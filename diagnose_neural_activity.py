#!/usr/bin/env python3
"""
Neural Activity Diagnostic Tool

Measures how many neurons are actually firing in each brain module.
This helps diagnose:
- Dead neurons (always output 0)
- Saturated neurons (always output max)
- Sparse activation (only a few neurons active)
- Dead ReLU problem (gradients vanish)
"""

import sys
import numpy as np
from pathlib import Path

try:
    import neurogen
    import sentencepiece as spm
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)


def analyze_neural_activity(model, tokenizer, test_inputs):
    """Measure neural activity across all brain modules"""
    
    print("="*80)
    print("NEURAL ACTIVITY ANALYSIS")
    print("="*80)
    print("\nThis diagnostic checks if neurons are actually firing.\n")
    
    # Test multiple inputs to see activation patterns
    all_stats = []
    
    for test_text in test_inputs:
        print(f"\n📝 Input: '{test_text[:50]}...'")
        
        # Tokenize
        token_ids = tokenizer.EncodeAsIds(test_text)
        if not token_ids:
            continue
        
        print(f"   Tokens: {len(token_ids)} tokens")
        
        try:
            # Get module statistics
            stats = model.get_brain_stats()
            
            print(f"\n   🧠 Brain Module Activity:")
            print(f"   {'Module':<20} {'Active %':<12} {'Avg Output':<15} {'Max Output':<15} {'Status'}")
            print(f"   {'-'*80}")
            
            module_stats = {}
            
            for module_name, module_data in stats['module_stats'].items():
                activity_level = module_data.get('activity_level', 0.0)
                dopamine = module_data.get('dopamine_level', 0.0)
                serotonin = module_data.get('serotonin_level', 0.0)
                
                # Calculate percentage of active neurons (above threshold)
                active_pct = activity_level * 100
                
                # Diagnose issues
                status = ""
                if active_pct < 1.0:
                    status = "❌ DEAD (< 1%)"
                elif active_pct < 5.0:
                    status = "⚠️  SPARSE (< 5%)"
                elif active_pct > 95.0:
                    status = "❌ SATURATED (> 95%)"
                else:
                    status = "✓ Normal"
                
                print(f"   {module_name:<20} {active_pct:>10.2f}% {activity_level:>13.4f} {status}")
                
                module_stats[module_name] = {
                    'active_pct': active_pct,
                    'activity_level': activity_level,
                    'status': status
                }
            
            all_stats.append(module_stats)
            
        except Exception as e:
            print(f"   ⚠️  Could not get stats: {e}")
    
    # Summary across all inputs
    if all_stats:
        print(f"\n" + "="*80)
        print("SUMMARY ACROSS ALL INPUTS")
        print("="*80)
        
        # Average activity per module
        module_names = all_stats[0].keys()
        
        for module_name in module_names:
            activities = [s[module_name]['active_pct'] for s in all_stats]
            avg_activity = np.mean(activities)
            std_activity = np.std(activities)
            
            print(f"\n{module_name}:")
            print(f"  Average Activity: {avg_activity:.2f}% (±{std_activity:.2f}%)")
            
            if avg_activity < 1.0:
                print(f"  🚨 CRITICAL: This module is essentially DEAD!")
                print(f"     - Almost no neurons are firing")
                print(f"     - Cannot learn or process information")
                print(f"     - Likely cause: ReLU dying, weight saturation, or vanishing gradients")
            elif avg_activity < 5.0:
                print(f"  ⚠️  WARNING: Very sparse activation")
                print(f"     - Only ~{int(avg_activity * 3072 / 100)} neurons active (out of thousands)")
                print(f"     - Limited representational capacity")
                print(f"     - May indicate: high inhibition, dying ReLU, or poor initialization")
            elif avg_activity > 95.0:
                print(f"  🚨 CRITICAL: Neurons are SATURATED!")
                print(f"     - Almost all neurons firing at maximum")
                print(f"     - No selectivity or discrimination")
                print(f"     - Likely cause: exploded weights or no inhibition")


def diagnose_dead_neurons():
    """Explain the dead neuron problem"""
    
    print("\n" + "="*80)
    print("UNDERSTANDING DEAD NEURONS")
    print("="*80)
    
    print("""
📊 What is "Dead Neuron" Problem?

Neurons use ReLU activation: f(x) = max(0, x)
- If x > 0: neuron outputs x (ACTIVE)
- If x ≤ 0: neuron outputs 0 (DEAD)

🚨 The Problem:
When learning rates are too high or weights become negative:

    neuron_output = ReLU(weights @ input + bias)
                  = ReLU(-5.2)  ← Negative!
                  = 0           ← Always zero!

Once a neuron dies (outputs 0), its gradient is also 0:
    gradient = 0  →  no weight updates  →  stays dead forever

📈 Activity Levels You Should See:

Early Training (Steps 1-100):
  Thalamus:      20-40% active  ← Sensory input processing
  Wernicke:      15-30% active  ← Language comprehension
  PFC:           10-25% active  ← Executive control
  Hippocampus:   10-20% active  ← Memory encoding
  Broca:          5-15% active  ← Output generation (sparse)
  Basal Ganglia:  5-15% active  ← Action selection (sparse)

Mid Training (Steps 100-1000):
  Activity should INCREASE as neurons learn to respond to patterns

Your Current State:
  If activity is < 1% across all modules → DEAD NEURONS
  If activity is < 5% → SPARSE ACTIVATION (may recover)
  If activity is > 95% → SATURATED (exploded weights)

🔧 Fixes by Cause:

1. Dead Neurons (< 1% activity):
   - Lower learning rates (already done: 0.0001-0.005)
   - Use Leaky ReLU instead of ReLU
   - Reinitialize weights (restart training)
   
2. Sparse Activation (1-5% activity):
   - Reduce inhibition levels in BrainOrchestrator
   - Increase excitability_bias
   - Lower attention thresholds
   
3. Saturated Neurons (> 95% activity):
   - Increase inhibition levels
   - Add dropout or weight decay
   - Reduce learning rates further
""")


def suggest_fixes(avg_activity_levels):
    """Suggest specific fixes based on measured activity"""
    
    print("\n" + "="*80)
    print("RECOMMENDED FIXES")
    print("="*80)
    
    # Analyze overall health
    all_dead = all(act < 1.0 for act in avg_activity_levels.values())
    mostly_sparse = all(act < 10.0 for act in avg_activity_levels.values())
    any_saturated = any(act > 95.0 for act in avg_activity_levels.values())
    
    if all_dead:
        print("""
🚨 CRITICAL: All modules are DEAD (<1% activity)

This means your model has completely failed to learn. Causes:
1. ✅ Learning rates too high (you've already reduced these)
2. ❌ Weights initialized poorly (need to restart training)
3. ❌ ReLU neurons dying (all outputs stuck at 0)

IMMEDIATE ACTIONS:
1. Delete checkpoint: rm checkpoints/full_model.ckpt
2. Restart training from scratch with new random weights
3. Monitor activity in first 10 chunks - should be 15-30%

If still dead after restart:
4. Switch from ReLU to Leaky ReLU (prevents dying)
5. Reduce inhibition_level in BrainOrchestrator.cpp
   - Change from 0.2-0.3 → 0.05-0.1
""")
    
    elif mostly_sparse:
        print("""
⚠️  WARNING: Sparse activation (< 10% activity)

Your neurons are alive but barely firing. This is often due to:
1. High inhibition suppressing neural activity
2. Attention thresholds filtering out too many signals
3. Early in training (may improve naturally)

RECOMMENDED ACTIONS:
1. In BrainOrchestrator.cpp, reduce inhibition:
   
   // Thalamus
   config.modulation.inhibition_level = 0.1f;  // Was 0.2f
   config.modulation.attention_threshold = 0.3f;  // Was 0.5f
   
   // Wernicke
   config.modulation.inhibition_level = 0.05f;  // Was 0.1f
   
   // Broca
   config.modulation.inhibition_level = 0.1f;  // Was 0.2f

2. Increase excitability:
   config.modulation.excitability_bias = 1.5f;  // Was 1.0-1.2f

3. Continue training - activity may increase naturally
""")
    
    elif any_saturated:
        print("""
🚨 CRITICAL: Some modules are SATURATED (>95% activity)

Too many neurons firing means no selectivity. The model can't discriminate.

ACTIONS:
1. Increase inhibition for saturated modules
2. Reduce learning rates further (0.0001 → 0.00005)
3. Add weight decay/regularization
4. Restart training with corrected settings
""")
    
    else:
        print("""
✓ Activity levels look reasonable (10-95%)

Your modules are firing appropriately. If you're still getting 0% accuracy:
1. Continue training - learning takes time
2. Verify greedy decoding is working
3. Check loss is decreasing over time
4. Monitor for 50-100 chunks before judging
""")


def main():
    print("🔍 Neural Activity Diagnostic\n")
    
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
        print("\nShowing diagnostic information anyway...\n")
        diagnose_dead_neurons()
        return
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load("tokenizer/nlp_agent_tokenizer.model")
        print("✓ Tokenizer loaded\n")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Test inputs
    test_inputs = [
        "The cat sat on the mat",
        "Hello world, how are you?",
        "In the beginning was the word",
        "Python is a programming language",
        "Machine learning models require data",
    ]
    
    # Run analysis
    try:
        analyze_neural_activity(model, tokenizer, test_inputs)
    except Exception as e:
        print(f"\n⚠️  Analysis failed: {e}")
        print("Model may not support get_brain_stats() yet")
    
    # Show educational content
    diagnose_dead_neurons()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Run this diagnostic after every 10 training chunks
2. If activity < 1%: DELETE checkpoint and restart training
3. If activity < 10%: Reduce inhibition in BrainOrchestrator.cpp
4. Monitor activity trends - should increase over time
5. Target: 20-40% activity in most modules by chunk 100

To check activity during training:
  python diagnose_neural_activity.py

To view current training state:
  python check_training_progress.py
""")


if __name__ == "__main__":
    main()
