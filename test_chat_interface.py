#!/usr/bin/env python3
"""
Test script for chat_interface.py

Verifies that the interface can:
1. Initialize the model
2. Load checkpoints
3. Generate text
4. Handle different parameters
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from chat_interface import NeuroGenChat
    print("✅ Successfully imported NeuroGenChat\n")
except ImportError as e:
    print(f"❌ Failed to import: {e}\n")
    sys.exit(1)


def test_initialization():
    """Test 1: Model initialization"""
    print("=" * 80)
    print("Test 1: Model Initialization")
    print("=" * 80)
    
    try:
        chat = NeuroGenChat()
        print("✅ Model initialized successfully\n")
        return chat
    except Exception as e:
        print(f"❌ Initialization failed: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def test_single_generation(chat):
    """Test 2: Single token generation"""
    print("=" * 80)
    print("Test 2: Single Token Generation")
    print("=" * 80)
    
    try:
        prompt = "Hello"
        print(f"\n📝 Prompt: \"{prompt}\"")
        response = chat.generate(prompt, max_tokens=10, stream=True)
        print(f"\n✅ Generation successful")
        print(f"📄 Response length: {len(response)} characters\n")
        return True
    except Exception as e:
        print(f"\n❌ Generation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_long_generation(chat):
    """Test 3: Longer generation"""
    print("=" * 80)
    print("Test 3: Longer Generation (50 tokens)")
    print("=" * 80)
    
    try:
        prompt = "The future of artificial intelligence"
        print(f"\n📝 Prompt: \"{prompt}\"")
        response = chat.generate(prompt, max_tokens=50, stream=True)
        print(f"\n✅ Long generation successful")
        print(f"📄 Response length: {len(response)} characters\n")
        return True
    except Exception as e:
        print(f"\n❌ Long generation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_temperature_variation(chat):
    """Test 4: Different temperatures"""
    print("=" * 80)
    print("Test 4: Temperature Variation")
    print("=" * 80)
    
    prompt = "Once upon a time"
    temperatures = [0.3, 0.7, 1.0]
    
    for temp in temperatures:
        try:
            print(f"\n🌡️  Temperature: {temp}")
            print(f"📝 Prompt: \"{prompt}\"")
            response = chat.generate(prompt, max_tokens=20, temperature=temp, stream=True)
            print(f"✅ Generated with temperature {temp}\n")
        except Exception as e:
            print(f"❌ Failed with temperature {temp}: {e}\n")
            return False
    
    return True


def test_stopping_conditions(chat):
    """Test 5: Stopping conditions"""
    print("=" * 80)
    print("Test 5: Stopping Conditions")
    print("=" * 80)
    
    # Test with sentence completion
    prompts = [
        ("Short test", 5),
        ("This is a longer test to see if the model stops appropriately", 30),
        ("Multiple sentences. Should continue. Until done", 50),
    ]
    
    for prompt, max_tok in prompts:
        try:
            print(f"\n📝 Prompt: \"{prompt}\" (max={max_tok})")
            response = chat.generate(prompt, max_tokens=max_tok, stream=True)
            print(f"✅ Stopped appropriately\n")
        except Exception as e:
            print(f"❌ Stopping test failed: {e}\n")
            return False
    
    return True


def test_checkpoint_loading():
    """Test 6: Checkpoint loading"""
    print("=" * 80)
    print("Test 6: Checkpoint Loading")
    print("=" * 80)
    
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        print("\n⚠️  No checkpoints directory found. Skipping test.\n")
        return True
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.bin"))
    
    if not checkpoints:
        print("\n⚠️  No checkpoints found. Skipping test.\n")
        return True
    
    # Test loading latest checkpoint
    latest = checkpoints[-1]
    print(f"\n📂 Loading checkpoint: {latest}")
    
    try:
        chat = NeuroGenChat(checkpoint_path=str(latest))
        print(f"✅ Checkpoint loaded successfully\n")
        
        # Try generating with loaded checkpoint
        print("🧪 Testing generation with loaded checkpoint...")
        response = chat.generate("Test prompt", max_tokens=10, stream=False)
        print(f"✅ Generation with checkpoint successful\n")
        return True
    except Exception as e:
        print(f"❌ Checkpoint loading failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 NeuroGen Chat Interface Test Suite")
    print("=" * 80 + "\n")
    
    results = {}
    
    # Test 1: Initialization
    chat = test_initialization()
    results["initialization"] = chat is not None
    
    if not chat:
        print("❌ Cannot continue tests without successful initialization\n")
        return
    
    # Test 2: Single generation
    results["single_generation"] = test_single_generation(chat)
    
    # Test 3: Long generation
    results["long_generation"] = test_long_generation(chat)
    
    # Test 4: Temperature variation
    results["temperature"] = test_temperature_variation(chat)
    
    # Test 5: Stopping conditions
    results["stopping"] = test_stopping_conditions(chat)
    
    # Test 6: Checkpoint loading (separate initialization)
    results["checkpoint"] = test_checkpoint_loading()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Results Summary")
    print("=" * 80 + "\n")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n{'=' * 80}")
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80 + "\n")
    
    if passed == total:
        print("🎉 All tests passed! The chat interface is ready to use.\n")
        print("Try it with: python3 chat_interface.py")
        print("Or with checkpoint: python3 chat_interface.py --auto-load-latest\n")
    else:
        print("⚠️  Some tests failed. Please check the errors above.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user\n")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
