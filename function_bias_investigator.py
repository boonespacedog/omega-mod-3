#!/usr/bin/env python3
"""
Investigation of biases in various arithmetic functions modulo small integers.

Functions investigated:
- ω(n): number of distinct prime factors
- σ(n): sum of divisors  
- λ(n): Liouville function
- τ(n): number of divisors
- φ(n): Euler's totient function
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from numba import jit
import time


@jit(nopython=True)
def omega_distinct(n):
    """ω(n) - Count distinct prime factors."""
    if n <= 1:
        return 0
    
    count = 0
    
    # Check if 2 divides n
    if n % 2 == 0:
        count += 1
        while n % 2 == 0:
            n //= 2
    
    # Check odd factors
    i = 3
    while i * i <= n:
        if n % i == 0:
            count += 1
            while n % i == 0:
                n //= i
        i += 2
    
    # If n is still > 1, it's prime
    if n > 1:
        count += 1
    
    return count


@jit(nopython=True)
def sigma(n):
    """σ(n) - Sum of all divisors of n."""
    if n <= 0:
        return 0
    
    sum_divisors = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            sum_divisors += i
            if i * i != n:
                sum_divisors += n // i
        i += 1
    
    return sum_divisors


@jit(nopython=True)
def tau(n):
    """τ(n) - Count of divisors of n."""
    if n <= 0:
        return 0
    
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i * i != n:
                count += 1
        i += 1
    
    return count


@jit(nopython=True)
def phi(n):
    """φ(n) - Euler's totient function."""
    if n <= 0:
        return 0
    
    result = n
    p = 2
    
    if n % 2 == 0:
        while n % 2 == 0:
            n //= 2
        result -= result // 2
    
    p = 3
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 2
    
    if n > 1:
        result -= result // n
    
    return result


@jit(nopython=True)
def liouville(n):
    """λ(n) - Liouville function: (-1)^Ω(n)."""
    if n <= 1:
        return 1
    
    omega_count = 0
    
    # Count all prime factors with multiplicity
    while n % 2 == 0:
        omega_count += 1
        n //= 2
    
    i = 3
    while i * i <= n:
        while n % i == 0:
            omega_count += 1
            n //= i
        i += 2
    
    if n > 1:
        omega_count += 1
    
    return -1 if omega_count % 2 == 1 else 1


def analyze_function_distribution(func, func_name, max_n=10**6, moduli=[2, 3, 4, 5, 6]):
    """Analyze the distribution of an arithmetic function modulo various integers."""
    print(f"\nAnalyzing {func_name} up to n = {max_n:,}")
    print("=" * 60)
    
    results = {}
    
    for mod in moduli:
        counts = defaultdict(int)
        
        # Special handling for Liouville function
        if func_name == "λ(n)" and mod > 2:
            print(f"Note: λ(n) only takes values ±1, showing distribution for λ(n) itself")
            mod = 2
        
        # Compute function values
        for n in range(1, max_n + 1):
            value = func(n)
            if func_name == "λ(n)":
                # Map -1 to 1, and 1 to 0 for modulo arithmetic
                counts[0 if value == 1 else 1] += 1
            else:
                counts[value % mod] += 1
        
        # Calculate statistics
        total = sum(counts.values())
        expected = total / (2 if func_name == "λ(n)" else mod)
        chi_squared = sum((count - expected)**2 / expected for count in counts.values())
        
        # Store results
        fractions = {k: v/total for k, v in counts.items()}
        max_deviation = max(abs(f - 1/len(counts)) for f in fractions.values())
        
        results[mod] = {
            'counts': dict(counts),
            'fractions': fractions,
            'chi_squared': chi_squared,
            'max_deviation': max_deviation
        }
        
        # Print results
        print(f"\nModulus {mod}:")
        for i in range(len(counts)):
            fraction = fractions[i]
            expected_frac = 1/len(counts)
            deviation = (fraction - expected_frac) * 100
            
            if func_name == "λ(n)" and mod == 2:
                label = f"λ(n) = {1 if i == 0 else -1}"
            else:
                label = f"{func_name} ≡ {i} (mod {mod})"
            
            print(f"  {label}: {fraction:.5f} ({deviation:+.2f}% from expected)")
        
        print(f"  Chi-squared: {chi_squared:.2f}")
        print(f"  Max deviation: {max_deviation:.5f} ({max_deviation*100:.2f}%)")
        
        if func_name == "λ(n)":
            break  # Only do mod 2 for Liouville
    
    return results


def compare_all_functions(max_n=10**5):
    """Compare biases across all arithmetic functions."""
    functions = [
        (omega_distinct, "ω(n)", "Distinct prime factors"),
        (sigma, "σ(n)", "Sum of divisors"),
        (tau, "τ(n)", "Number of divisors"),
        (phi, "φ(n)", "Euler's totient"),
        (liouville, "λ(n)", "Liouville function")
    ]
    
    print(f"\nCOMPARING ALL FUNCTIONS AT n = {max_n:,}")
    print("=" * 80)
    
    # Test each function mod 3
    mod = 3
    comparison_results = {}
    
    for func, name, description in functions:
        print(f"\n{name} - {description}")
        print("-" * 40)
        
        # Special case for Liouville
        if name == "λ(n)":
            mod_test = 2
        else:
            mod_test = mod
        
        counts = defaultdict(int)
        
        start_time = time.time()
        for n in range(1, max_n + 1):
            value = func(n)
            if name == "λ(n)":
                counts[0 if value == 1 else 1] += 1
            else:
                counts[value % mod_test] += 1
        
        elapsed = time.time() - start_time
        
        # Calculate statistics
        total = sum(counts.values())
        fractions = {k: v/total for k, v in counts.items()}
        expected = 1/len(counts)
        max_deviation = max(abs(f - expected) for f in fractions.values())
        
        comparison_results[name] = {
            'max_deviation': max_deviation,
            'fractions': fractions,
            'computation_time': elapsed
        }
        
        # Print summary
        for i in sorted(counts.keys()):
            fraction = fractions[i]
            deviation = (fraction - expected) * 100
            print(f"  Class {i}: {fraction:.5f} ({deviation:+.2f}%)")
        
        print(f"  Max deviation: {max_deviation*100:.2f}%")
        print(f"  Computation time: {elapsed:.2f}s")
    
    return comparison_results


def plot_bias_comparison(comparison_results):
    """Create visualization comparing biases across functions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot of maximum deviations
    functions = list(comparison_results.keys())
    max_devs = [comparison_results[f]['max_deviation'] * 100 for f in functions]
    
    bars = ax1.bar(functions, max_devs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9C74F', '#90BE6D'])
    ax1.set_ylabel('Maximum Deviation (%)')
    ax1.set_title('Bias Strength Across Arithmetic Functions')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, max_devs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}%', ha='center', va='bottom')
    
    # Distribution details for each function
    y_pos = 0
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    for i, (func, data) in enumerate(comparison_results.items()):
        fractions = data['fractions']
        x_start = 0
        
        for class_id, fraction in sorted(fractions.items()):
            width = fraction
            ax2.barh(y_pos, width, left=x_start, height=0.8, 
                    label=f'Class {class_id}' if i == 0 else '', 
                    color=colors[class_id % len(colors)])
            
            # Add percentage text
            if width > 0.05:  # Only show text if segment is large enough
                ax2.text(x_start + width/2, y_pos, f'{fraction*100:.1f}%', 
                        ha='center', va='center', fontsize=10)
            
            x_start += width
        
        y_pos += 1
    
    ax2.set_yticks(range(len(functions)))
    ax2.set_yticklabels(functions)
    ax2.set_xlabel('Fraction')
    ax2.set_title('Distribution Breakdown by Function')
    ax2.set_xlim(0, 1)
    ax2.axvline(x=1/3, color='red', linestyle='--', alpha=0.5, label='Expected (1/3)')
    ax2.axvline(x=1/2, color='red', linestyle='--', alpha=0.5, label='Expected (1/2)')
    
    plt.tight_layout()
    plt.savefig('arithmetic_functions_bias_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def investigate_sigma_patterns(max_n=10**5):
    """Deep dive into σ(n) patterns."""
    print("\nDEEP DIVE: σ(n) PATTERNS")
    print("=" * 60)
    
    # Analyze σ(n) for different types of numbers
    patterns = {
        'primes': [],
        'prime_powers': [],
        'semiprimes': [],
        'highly_composite': [],
        'perfect_powers': []
    }
    
    for n in range(2, min(max_n, 10000)):
        sigma_n = sigma(n)
        omega_n = omega_distinct(n)
        
        # Check number type
        if omega_n == 1 and tau(n) == 2:  # Prime
            patterns['primes'].append((n, sigma_n, sigma_n % 3))
        elif omega_n == 1:  # Prime power
            patterns['prime_powers'].append((n, sigma_n, sigma_n % 3))
        elif omega_n == 2 and tau(n) == 4:  # Semiprime (p*q)
            patterns['semiprimes'].append((n, sigma_n, sigma_n % 3))
        elif n == 2**int(np.log2(n)):  # Powers of 2
            patterns['perfect_powers'].append((n, sigma_n, sigma_n % 3))
    
    # Analyze patterns
    for pattern_type, data in patterns.items():
        if not data:
            continue
            
        print(f"\n{pattern_type.replace('_', ' ').title()}:")
        
        # Count mod 3 distribution
        mod3_counts = defaultdict(int)
        for _, _, mod3 in data:
            mod3_counts[mod3] += 1
        
        total = len(data)
        for mod_val in sorted(mod3_counts.keys()):
            count = mod3_counts[mod_val]
            print(f"  σ(n) ≡ {mod_val} (mod 3): {count}/{total} = {count/total:.3f}")
        
        # Show examples
        print(f"  Examples: {data[:5]}")


def main():
    """Run the complete investigation."""
    print("ARITHMETIC FUNCTIONS BIAS INVESTIGATION")
    print("======================================")
    
    # 1. Analyze each function individually
    results = {}
    
    print("\n1. INDIVIDUAL FUNCTION ANALYSIS")
    results['omega_distinct'] = analyze_function_distribution(
        omega_distinct, "ω(n)", max_n=10**5, moduli=[2, 3, 4, 5, 6]
    )
    
    results['sigma'] = analyze_function_distribution(
        sigma, "σ(n)", max_n=10**5, moduli=[2, 3, 4, 5, 6, 7, 8]
    )
    
    results['liouville'] = analyze_function_distribution(
        liouville, "λ(n)", max_n=10**6, moduli=[2]
    )
    
    # 2. Compare all functions
    print("\n2. COMPARATIVE ANALYSIS")
    comparison = compare_all_functions(max_n=10**5)
    
    # 3. Create visualizations
    print("\n3. CREATING VISUALIZATIONS")
    plot_bias_comparison(comparison)
    
    # 4. Deep dive into σ(n)
    investigate_sigma_patterns(max_n=10**4)
    
    # 5. Save results
    print("\n4. SAVING RESULTS")
    output = {
        'individual_results': results,
        'comparison': comparison,
        'analysis_parameters': {
            'max_n': 10**5,
            'functions_tested': ['ω(n)', 'σ(n)', 'τ(n)', 'φ(n)', 'λ(n)']
        }
    }
    
    with open('arithmetic_functions_bias_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print("\nAnalysis complete! Results saved to:")
    print("  - arithmetic_functions_bias_analysis.json")
    print("  - arithmetic_functions_bias_comparison.png")


if __name__ == "__main__":
    main()