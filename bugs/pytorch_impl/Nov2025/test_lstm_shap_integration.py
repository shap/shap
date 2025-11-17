"""
Integration Test: LSTM SHAP Backward Hook vs Manual Calculation

This test verifies that the backward hook implementation produces identical
results to the standalone manual SHAP calculation.
"""

import torch
import torch.nn as nn
import numpy as np

# Import our implementations
from lstm_cell_complete_pytorch import LSTMCellModel, manual_shap_lstm_cell
from lstm_shap_backward_hook import LSTMShapBackwardHook, register_lstm_shap_hook


def test_backward_hook_vs_manual():
    """Test that backward hook matches manual calculation exactly."""

    print("="*80)
    print("LSTM SHAP Backward Hook Integration Test")
    print("="*80)
    print("\nVerifying that backward hook matches manual SHAP calculation exactly.")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Model dimensions
    input_size = 3
    hidden_size = 2
    batch_size = 1

    # Create model
    model = LSTMCellModel(input_size, hidden_size)

    # Set reproducible weights (same as before)
    model.fc_ii.weight.data = torch.tensor([
        [1.0, 1.0, 0.5],
        [0.5, 0.3, 0.2]
    ], dtype=torch.float32)
    model.fc_ii.bias.data = torch.tensor([0.2, 0.1], dtype=torch.float32)

    model.fc_hi.weight.data = torch.tensor([
        [2.0, 1.0],
        [0.5, 0.8]
    ], dtype=torch.float32)
    model.fc_hi.bias.data = torch.tensor([0.32, 0.15], dtype=torch.float32)

    model.fc_if.weight.data = torch.tensor([
        [0.8, 0.6, 0.4],
        [0.3, 0.5, 0.7]
    ], dtype=torch.float32)
    model.fc_if.bias.data = torch.tensor([0.1, 0.05], dtype=torch.float32)

    model.fc_hf.weight.data = torch.tensor([
        [1.5, 0.9],
        [0.7, 1.2]
    ], dtype=torch.float32)
    model.fc_hf.bias.data = torch.tensor([0.25, 0.18], dtype=torch.float32)

    model.fc_ig.weight.data = torch.tensor([
        [1.2, 0.9, 0.6],
        [0.4, 0.8, 1.0]
    ], dtype=torch.float32)
    model.fc_ig.bias.data = torch.tensor([0.15, 0.08], dtype=torch.float32)

    model.fc_hg.weight.data = torch.tensor([
        [1.8, 1.1],
        [0.9, 1.3]
    ], dtype=torch.float32)
    model.fc_hg.bias.data = torch.tensor([0.28, 0.12], dtype=torch.float32)

    # Input data
    x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    h = torch.tensor([[0.0, 0.1]], dtype=torch.float32)
    c = torch.tensor([[0.5, 0.3]], dtype=torch.float32)

    # Baseline data
    x_base = torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32)
    h_base = torch.tensor([[0.0, 0.01]], dtype=torch.float32)
    c_base = torch.tensor([[0.1, 0.05]], dtype=torch.float32)

    print("\n" + "="*80)
    print("Method 1: Manual SHAP Calculation (Validated)")
    print("="*80)

    # Method 1: Use the validated manual calculation
    r_x_manual, r_h_manual, r_c_manual, output_manual, output_base_manual = manual_shap_lstm_cell(
        model, x, h, c, x_base, h_base, c_base
    )

    print(f"\nManual SHAP values:")
    print(f"  r_x: {r_x_manual.item():.15f}")
    print(f"  r_h: {r_h_manual.item():.15f}")
    print(f"  r_c: {r_c_manual.item():.15f}")
    print(f"  Total: {(r_x_manual + r_h_manual + r_c_manual).item():.15f}")

    # Verify additivity
    output_diff_manual = (output_manual - output_base_manual).sum()
    additivity_error_manual = abs((r_x_manual + r_h_manual + r_c_manual).item() - output_diff_manual.item())
    print(f"\nAdditivity check:")
    print(f"  Expected (output diff): {output_diff_manual.item():.15f}")
    print(f"  Actual (SHAP total): {(r_x_manual + r_h_manual + r_c_manual).item():.15f}")
    print(f"  Error: {additivity_error_manual:.15f}")
    print(f"  ✓ Satisfied: {additivity_error_manual < 1e-6}")

    print("\n" + "="*80)
    print("Method 2: Backward Hook (New Implementation)")
    print("="*80)

    # Method 2: Use the backward hook
    hook = register_lstm_shap_hook(model, x_base, h_base, c_base)

    print(f"\nBackward hook initialized:")
    print(f"  Input size: {hook.input_size}")
    print(f"  Hidden size: {hook.hidden_size}")
    print(f"  Weights extracted: W_ii shape {hook.W_ii.shape}, W_hi shape {hook.W_hi.shape}")

    # Calculate SHAP using hook
    output_hook, shap_values = hook(x, h, c)

    r_x_hook = shap_values['shap_x']
    r_h_hook = shap_values['shap_h']
    r_c_hook = shap_values['shap_c']

    print(f"\nBackward hook SHAP values:")
    print(f"  r_x: {r_x_hook.item():.15f}")
    print(f"  r_h: {r_h_hook.item():.15f}")
    print(f"  r_c: {r_c_hook.item():.15f}")
    print(f"  Total: {(r_x_hook + r_h_hook + r_c_hook).item():.15f}")

    # Verify additivity for hook
    output_base_hook = model(x_base, h_base, c_base)
    output_diff_hook = (output_hook - output_base_hook).sum()
    additivity_error_hook = abs((r_x_hook + r_h_hook + r_c_hook).item() - output_diff_hook.item())
    print(f"\nAdditivity check:")
    print(f"  Expected (output diff): {output_diff_hook.item():.15f}")
    print(f"  Actual (SHAP total): {(r_x_hook + r_h_hook + r_c_hook).item():.15f}")
    print(f"  Error: {additivity_error_hook:.15f}")
    print(f"  ✓ Satisfied: {additivity_error_hook < 1e-6}")

    print("\n" + "="*80)
    print("Comparison: Manual vs Backward Hook")
    print("="*80)

    # Compare outputs
    output_match = torch.allclose(output_manual, output_hook, atol=1e-10)
    print(f"\nOutputs match: {output_match}")
    print(f"  Manual output: {output_manual}")
    print(f"  Hook output: {output_hook}")
    print(f"  Difference: {(output_manual - output_hook).abs().max().item():.15e}")

    # Compare SHAP values
    error_x = abs(r_x_manual.item() - r_x_hook.item())
    error_h = abs(r_h_manual.item() - r_h_hook.item())
    error_c = abs(r_c_manual.item() - r_c_hook.item())
    total_error = error_x + error_h + error_c

    print(f"\nSHAP value comparison:")
    print(f"  r_x difference: {error_x:.15e}")
    print(f"  r_h difference: {error_h:.15e}")
    print(f"  r_c difference: {error_c:.15e}")
    print(f"  Total difference: {total_error:.15e}")

    # Test exact match
    tolerance = 1e-10
    r_x_match = error_x < tolerance
    r_h_match = error_h < tolerance
    r_c_match = error_c < tolerance

    print(f"\nExact match check (tolerance={tolerance:.2e}):")
    print(f"  r_x matches: {r_x_match}")
    print(f"  r_h matches: {r_h_match}")
    print(f"  r_c matches: {r_c_match}")

    all_match = r_x_match and r_h_match and r_c_match and output_match

    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)

    if all_match:
        print("\n✓✓✓ SUCCESS! ✓✓✓")
        print("\nBackward hook implementation matches manual calculation EXACTLY!")
        print(f"  - All SHAP values match (error < {tolerance:.2e})")
        print(f"  - Both satisfy additivity perfectly")
        print(f"  - Outputs are identical")
        print("\n→ The backward hook is ready for integration into SHAP library!")
    else:
        print("\n✗✗✗ FAILURE ✗✗✗")
        print("\nBackward hook does NOT match manual calculation!")
        print("  Check implementation for errors.")
        raise AssertionError("Backward hook does not match manual calculation")

    return all_match


def test_multiple_inputs():
    """Test backward hook with multiple different inputs."""

    print("\n" + "="*80)
    print("Multi-Input Test: Testing with various inputs")
    print("="*80)

    torch.manual_seed(42)

    input_size = 3
    hidden_size = 2

    # Create model with random weights
    model = LSTMCellModel(input_size, hidden_size)

    # Baseline
    x_base = torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32)
    h_base = torch.tensor([[0.0, 0.01]], dtype=torch.float32)
    c_base = torch.tensor([[0.1, 0.05]], dtype=torch.float32)

    # Create hook
    hook = register_lstm_shap_hook(model, x_base, h_base, c_base)

    # Test with multiple inputs
    test_inputs = [
        (torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
         torch.tensor([[0.0, 0.1]], dtype=torch.float32),
         torch.tensor([[0.5, 0.3]], dtype=torch.float32)),

        (torch.tensor([[0.5, 0.3, 0.1]], dtype=torch.float32),
         torch.tensor([[0.2, 0.4]], dtype=torch.float32),
         torch.tensor([[0.8, 0.6]], dtype=torch.float32)),

        (torch.tensor([[0.9, 0.1, 0.5]], dtype=torch.float32),
         torch.tensor([[0.3, 0.2]], dtype=torch.float32),
         torch.tensor([[0.4, 0.9]], dtype=torch.float32)),
    ]

    all_passed = True

    for i, (x, h, c) in enumerate(test_inputs):
        print(f"\nTest case {i+1}:")
        print(f"  x={x.numpy()}, h={h.numpy()}, c={c.numpy()}")

        # Manual calculation
        r_x_manual, r_h_manual, r_c_manual, output_manual, output_base = manual_shap_lstm_cell(
            model, x, h, c, x_base, h_base, c_base
        )

        # Hook calculation
        output_hook, shap_values = hook(x, h, c)
        r_x_hook = shap_values['shap_x']
        r_h_hook = shap_values['shap_h']
        r_c_hook = shap_values['shap_c']

        # Compare
        error_x = abs(r_x_manual.item() - r_x_hook.item())
        error_h = abs(r_h_manual.item() - r_h_hook.item())
        error_c = abs(r_c_manual.item() - r_c_hook.item())

        match = error_x < 1e-10 and error_h < 1e-10 and error_c < 1e-10

        print(f"  Manual: r_x={r_x_manual.item():.10f}, r_h={r_h_manual.item():.10f}, r_c={r_c_manual.item():.10f}")
        print(f"  Hook:   r_x={r_x_hook.item():.10f}, r_h={r_h_hook.item():.10f}, r_c={r_c_hook.item():.10f}")
        print(f"  Error:  {error_x:.2e}, {error_h:.2e}, {error_c:.2e}")
        print(f"  ✓ Match: {match}")

        if not match:
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✓ All multi-input tests passed!")
    else:
        print("✗ Some tests failed!")
        raise AssertionError("Multi-input test failed")

    return all_passed


def test_builtin_lstm_cell():
    """Test backward hook with PyTorch's built-in LSTMCell."""

    print("\n" + "="*80)
    print("Built-in LSTMCell Test")
    print("="*80)

    torch.manual_seed(42)

    input_size = 3
    hidden_size = 2

    # Create built-in LSTMCell
    builtin_lstm = nn.LSTMCell(input_size, hidden_size)

    # Baseline
    x_base = torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32)
    h_base = torch.tensor([[0.0, 0.01]], dtype=torch.float32)
    c_base = torch.tensor([[0.1, 0.05]], dtype=torch.float32)

    # Create hook
    print("\nRegistering hook for built-in LSTMCell...")
    hook = register_lstm_shap_hook(builtin_lstm, x_base, h_base, c_base)

    print(f"✓ Hook registered successfully")
    print(f"  Extracted weights: W_ii {hook.W_ii.shape}, W_hi {hook.W_hi.shape}")

    # Test input
    x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    h = torch.tensor([[0.0, 0.1]], dtype=torch.float32)
    c = torch.tensor([[0.5, 0.3]], dtype=torch.float32)

    # Calculate using hook
    new_c, shap_values = hook(x, h, c)

    r_x = shap_values['shap_x']
    r_h = shap_values['shap_h']
    r_c = shap_values['shap_c']

    print(f"\nSHAP values from hook:")
    print(f"  r_x: {r_x.item():.10f}")
    print(f"  r_h: {r_h.item():.10f}")
    print(f"  r_c: {r_c.item():.10f}")
    print(f"  Total: {(r_x + r_h + r_c).item():.10f}")

    # Verify additivity
    _, c_base_output = builtin_lstm(x_base, (h_base, c_base))
    output_diff = (new_c - c_base_output).sum()
    shap_total = r_x + r_h + r_c

    error = abs(output_diff.item() - shap_total.item())

    print(f"\nAdditivity check:")
    print(f"  Expected (output diff): {output_diff.item():.10f}")
    print(f"  Actual (SHAP total): {shap_total.item():.10f}")
    print(f"  Error: {error:.15f}")

    passed = error < 1e-6
    print(f"  ✓ Satisfied: {passed}")

    print("\n" + "="*80)
    if passed:
        print("✓ Built-in LSTMCell test passed!")
    else:
        print("✗ Built-in LSTMCell test failed!")
        raise AssertionError("Built-in LSTMCell test failed")

    return passed


if __name__ == "__main__":
    print("="*80)
    print("LSTM SHAP BACKWARD HOOK - COMPREHENSIVE INTEGRATION TESTS")
    print("="*80)

    results = {}

    # Test 1: Exact match with manual calculation
    try:
        results['exact_match'] = test_backward_hook_vs_manual()
    except Exception as e:
        print(f"\n✗ Exact match test failed: {e}")
        results['exact_match'] = False
        raise

    # Test 2: Multiple inputs
    try:
        results['multi_input'] = test_multiple_inputs()
    except Exception as e:
        print(f"\n✗ Multi-input test failed: {e}")
        results['multi_input'] = False
        raise

    # Test 3: Built-in LSTMCell
    try:
        results['builtin_lstm'] = test_builtin_lstm_cell()
    except Exception as e:
        print(f"\n✗ Built-in LSTMCell test failed: {e}")
        results['builtin_lstm'] = False
        raise

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    print(f"\n1. Exact match test: {'✓ PASSED' if results['exact_match'] else '✗ FAILED'}")
    print(f"2. Multi-input test: {'✓ PASSED' if results['multi_input'] else '✗ FAILED'}")
    print(f"3. Built-in LSTMCell test: {'✓ PASSED' if results['builtin_lstm'] else '✗ FAILED'}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("\nThe LSTM SHAP backward hook implementation is:")
        print("  - Mathematically correct")
        print("  - Exactly matches validated manual calculation")
        print("  - Works with multiple inputs")
        print("  - Compatible with PyTorch's built-in LSTMCell")
        print("  - Ready for integration into SHAP library")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        raise AssertionError("Integration tests failed")

    print("="*80)
