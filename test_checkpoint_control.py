#!/usr/bin/env python3
"""
Verify Checkpoint Control Feature

This script tests the --no-checkpoint argument.
"""

import subprocess
import sys

def test_help():
    """Test that --help shows the new argument"""
    print("Testing --help output...")
    result = subprocess.run(
        ["python", "train_slimpajama.py", "--help"],
        capture_output=True,
        text=True
    )
    
    if "--no-checkpoint" in result.stdout:
        print("✅ --no-checkpoint argument is documented")
        return True
    else:
        print("❌ --no-checkpoint argument not found in help")
        return False

def test_config_display():
    """Test that configuration shows checkpoint status"""
    print("\nTesting configuration display...")
    
    # Test with checkpoints enabled (default)
    print("\n1. Testing WITH checkpoints (default):")
    result = subprocess.run(
        ["python", "train_slimpajama.py", "--test", "--max-chunks", "0"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if "Save checkpoints: ✓ Enabled" in result.stdout:
        print("✅ Shows checkpoints enabled by default")
        enabled_ok = True
    else:
        print("❌ Does not show checkpoint status correctly")
        print(result.stdout)
        enabled_ok = False
    
    # Test with checkpoints disabled
    print("\n2. Testing WITHOUT checkpoints (--no-checkpoint):")
    result = subprocess.run(
        ["python", "train_slimpajama.py", "--test", "--max-chunks", "0", "--no-checkpoint"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if "Save checkpoints: ✗ Disabled" in result.stdout:
        print("✅ Shows checkpoints disabled with --no-checkpoint")
        disabled_ok = True
    else:
        print("❌ Does not show checkpoint status correctly")
        print(result.stdout)
        disabled_ok = False
    
    return enabled_ok and disabled_ok

def main():
    print("="*80)
    print("Checkpoint Control Feature Verification")
    print("="*80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Help text
    if test_help():
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 2: Configuration display
    if test_config_display():
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
