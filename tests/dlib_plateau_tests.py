import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.join(os.path.abspath(__file__))), '..' ))

from dlib_loss_plateu import *
import numpy as np
import dlib
from typing import List, Tuple


def generate_large_test_cases() -> List[Tuple[str, List[float], bool]]:
    """Generate various test cases with large samples."""
    np.random.seed(42)
    
    test_cases = []
    
    # Helper function to add noise
    def add_noise(x, noise_level=0.01):
        return x + np.random.normal(0, noise_level, size=len(x))
    
    # 1. Long steady decrease (2000 elements)
    n = 2000
    values = np.linspace(1.0, 0.1, n)
    test_cases.append((
        "Long steady decrease",
        add_noise(values, 0.001).tolist(),
        False
    ))
    
    # 2. Long plateau (2000 elements)
    values = np.full(n, 0.5)
    test_cases.append((
        "Long plateau",
        add_noise(values, 0.001).tolist(),
        True
    ))
    
    # 3. Decrease then plateau (2000 elements)
    values = np.concatenate([
        np.linspace(1.0, 0.5, 1000),
        np.full(1000, 0.5)
    ])
    test_cases.append((
        "Decrease then plateau",
        add_noise(values, 0.001).tolist(),
        True
    ))
    
    # 4. Plateau with periodic spikes (1500 elements)
    base = np.full(1500, 0.5)
    spike_indices = np.arange(100, 1500, 100)
    base[spike_indices] = 1.0
    test_cases.append((
        "Long plateau with periodic spikes",
        base.tolist(),
        True
    ))
    
    # 5. Exponential decay (1000 elements)
    x = np.linspace(0, 10, 1000)
    values = np.exp(-x/2)
    test_cases.append((
        "Exponential decay",
        add_noise(values, 0.001).tolist(),
        False
    ))
    
    # 6. Sigmoid-like transition to plateau (1500 elements)
    x = np.linspace(-10, 10, 1500)
    values = 1 / (1 + np.exp(x))
    test_cases.append((
        "Sigmoid to plateau",
        add_noise(values, 0.001).tolist(),
        True
    ))
    
    # 7. Stepwise decrease (2000 elements)
    steps = np.repeat(np.linspace(1.0, 0.1, 20), 100)
    test_cases.append((
        "Stepwise decrease",
        add_noise(steps, 0.001).tolist(),
        False
    ))
    
    # 8. Oscillating convergence (1500 elements)
    x = np.linspace(0, 15*np.pi, 1500)
    values = 0.5 + np.exp(-x/10) * np.sin(x)
    test_cases.append((
        "Oscillating convergence",
        values.tolist(),
        True
    ))
    
    # 9. Long plateau with outliers (2000 elements)
    values = np.full(2000, 0.5)
    outlier_indices = np.random.choice(2000, 100, replace=False)
    values[outlier_indices] = np.random.uniform(1.0, 2.0, 100)
    test_cases.append((
        "Long plateau with random outliers",
        values.tolist(),
        True
    ))
    
    # 10. Slow convergence (1500 elements)
    x = np.arange(1, 1501)
    values = 1/np.log(x + 1)
    test_cases.append((
        "Slow convergence",
        add_noise(values, 0.0001).tolist(),
        False
    ))
    
    # 11. Noisy sine wave with decreasing amplitude (2000 elements)
    x = np.linspace(0, 20*np.pi, 2000)
    envelope = np.exp(-x/20)
    values = 0.5 + envelope * np.sin(x)
    test_cases.append((
        "Damped sine wave",
        add_noise(values, 0.001).tolist(),
        True
    ))
    
    # 12. Random walk with trend (1500 elements)
    random_walk = np.random.normal(0, 0.01, 1500).cumsum()
    trend = np.linspace(1.0, 0.1, 1500)
    values = trend + random_walk
    test_cases.append((
        "Random walk with decreasing trend",
        values.tolist(),
        False
    ))
    
    return test_cases

def compare_implementations(values: List[float], threshold: int = 5) -> dict:
    """Compare dlib and custom implementations."""
    # Convert to dlib compatible format
    dlib_vec = dlib.vector(values)
    
    dlib_simple = dlib.count_steps_without_decrease(dlib_vec)
    dlib_robust = dlib.count_steps_without_decrease_robust(dlib_vec)
    custom_simple = count_steps_without_decrease(values)
    custom_robust = count_steps_without_decrease_robust(values)
    
    return {
        'dlib_simple': dlib_simple,
        'dlib_robust': dlib_robust,
        'custom_simple': custom_simple,
        'custom_robust': custom_robust,
        'agreement_simple': dlib_simple == custom_simple,
        'agreement_robust': dlib_robust == custom_robust,
        'is_plateau_dlib': dlib_simple > threshold and dlib_robust > threshold,
        'is_plateau_custom': custom_simple > threshold and custom_robust > threshold
    }

def run_comprehensive_tests():
    """Run comprehensive comparison tests."""
    test_cases = generate_large_test_cases()
    failed_cases = []
    passed_cases = []
    
    print("Running Large-Scale Comparison Tests")
    print("=" * 80)
    print(f"Testing {len(test_cases)} cases...\n")
    
    for case_name, values, expected_plateau in test_cases:
        result = compare_implementations(values)
        
        # Check if both implementations agree
        if result['is_plateau_dlib'] == result['is_plateau_custom']:
            passed_cases.append(case_name)
        else:
            failed_cases.append({
                'name': case_name,
                'size': len(values),
                'dlib_result': result['is_plateau_dlib'],
                'custom_result': result['is_plateau_custom'],
                'dlib_steps': (result['dlib_simple'], result['dlib_robust']),
                'custom_steps': (result['custom_simple'], result['custom_robust'])
            })
            # Only print details for failed tests
            print(f"\nTest Case Failed: {case_name}")
            print(f"Sample size: {len(values)}")
            print(f"Dlib detected plateau: {result['is_plateau_dlib']}")
            print(f"Custom detected plateau: {result['is_plateau_custom']}")
            print(f"Steps without decrease (dlib simple/robust): {result['dlib_simple']}/{result['dlib_robust']}")
            print(f"Steps without decrease (custom simple/robust): {result['custom_simple']}/{result['custom_robust']}")
            print("-" * 80)
    
    # Print summary
    print("\nTest Summary")
    print("=" * 80)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Passed: {len(passed_cases)}")
    print(f"Failed: {len(failed_cases)}")
    
    if failed_cases:
        print("\nFailed Cases Summary:")
        print("-" * 80)
        print(f"{'Case Name':<35} {'Dlib':<10} {'Custom':<10}")
        print("-" * 80)
        for case in failed_cases:
            print(f"{case['name']:<35} {str(case['dlib_result']):<10} {str(case['custom_result']):<10}")
    else:
        print("\nAll test cases passed! 🎉")

if __name__ == "__main__":
    run_comprehensive_tests()