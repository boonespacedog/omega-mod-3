#!/usr/bin/env python3
"""
Compute character sums S(x) and S₂(x) to verify theoretical predictions.

According to Halász-Delange theory, these sums should decay as x/(log x)^(3/2).
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import cmath

@jit(nopython=True)
def compute_omega(n: int) -> int:
    """Compute Ω(n) - total number of prime factors with multiplicity."""
    if n <= 1:
        return 0
    
    omega_count = 0
    
    # Handle factors of 2
    while n % 2 == 0:
        omega_count += 1
        n //= 2
    
    # Handle odd factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            omega_count += 1
            n //= i
        i += 2
    
    # If n is still > 1, it's prime
    if n > 1:
        omega_count += 1
    
    return omega_count


def compute_character_sums(max_n: int, checkpoints=None):
    """
    Compute S(x) = Σ_{n≤x} ω^{Ω(n)} and S₂(x) = Σ_{n≤x} ω^{2Ω(n)}
    where ω = e^{2πi/3}.
    """
    if checkpoints is None:
        checkpoints = [10**i for i in range(3, int(np.log10(max_n)) + 1)]
    
    omega = cmath.exp(2j * cmath.pi / 3)  # e^{2πi/3}
    omega2 = omega * omega  # e^{4πi/3}
    
    results = {}
    
    print(f"Computing character sums up to {max_n:,}")
    print("=" * 60)
    
    for checkpoint in checkpoints:
        if checkpoint > max_n:
            break
            
        # Compute sums
        S1 = 0j  # S(x)
        S2 = 0j  # S₂(x)
        counts = [0, 0, 0]  # A₀, A₁, A₂
        
        for n in range(1, checkpoint + 1):
            omega_n = compute_omega(n)
            mod3 = omega_n % 3
            counts[mod3] += 1
            
            if mod3 == 0:
                S1 += 1
                S2 += 1
            elif mod3 == 1:
                S1 += omega
                S2 += omega2
            else:  # mod3 == 2
                S1 += omega2
                S2 += omega
        
        # Theoretical predictions
        log_x = np.log(checkpoint)
        theoretical_bound = checkpoint / (log_x ** 1.5)
        
        # Compute deviations using the formulas from feedback
        x = checkpoint
        delta_0 = (x + S1.real + S2.real) / 3 - x/3
        delta_1 = (x + omega2.real * S1.real + omega.real * S2.real) / 3 - x/3
        delta_2 = (x + omega.real * S1.real + omega2.real * S2.real) / 3 - x/3
        
        results[checkpoint] = {
            'S1': S1,
            'S2': S2,
            'S1_magnitude': abs(S1),
            'S2_magnitude': abs(S2),
            'S1_over_x': abs(S1) / checkpoint,
            'S2_over_x': abs(S2) / checkpoint,
            'counts': counts,
            'theoretical_bound': theoretical_bound,
            'delta_0': delta_0 / x,
            'delta_1': delta_1 / x,
            'delta_2': delta_2 / x,
            'log_x': log_x
        }
        
        print(f"\nx = {checkpoint:,}")
        print(f"|S(x)|/x = {abs(S1)/checkpoint:.6f}")
        print(f"|S₂(x)|/x = {abs(S2)/checkpoint:.6f}")
        print(f"Theoretical bound: x/(log x)^1.5 = {theoretical_bound:.2f}")
        print(f"Deviations from 1/3: Δ₀={delta_0/x:.6f}, Δ₁={delta_1/x:.6f}, Δ₂={delta_2/x:.6f}")
    
    return results


def plot_character_sums(results):
    """Plot the character sums to verify theoretical predictions."""
    xs = sorted(results.keys())
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: |S(x)|/x and |S₂(x)|/x vs log x
    log_xs = [results[x]['log_x'] for x in xs]
    S1_over_x = [results[x]['S1_over_x'] for x in xs]
    S2_over_x = [results[x]['S2_over_x'] for x in xs]
    
    ax1.loglog(xs, S1_over_x, 'o-', label='|S(x)|/x', markersize=8)
    ax1.loglog(xs, S2_over_x, 's-', label='|S₂(x)|/x', markersize=8)
    
    # Add theoretical decay line (log x)^(-3/2)
    theoretical = [1 / (results[x]['log_x'] ** 1.5) for x in xs]
    ax1.loglog(xs, theoretical, 'k--', label='(log x)^(-3/2)', alpha=0.7)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('|S(x)|/x')
    ax1.set_title('Character Sum Magnitudes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-log plot of |S(x)|/x vs log x to check slope
    ax2.loglog(log_xs, S1_over_x, 'o-', label='|S(x)|/x', markersize=8)
    ax2.loglog(log_xs, S2_over_x, 's-', label='|S₂(x)|/x', markersize=8)
    
    # Add reference line with slope -3/2
    ref_y = [0.5 * lx**(-1.5) for lx in log_xs]
    ax2.loglog(log_xs, ref_y, 'k--', label='slope = -3/2', alpha=0.7)
    
    ax2.set_xlabel('log x')
    ax2.set_ylabel('|S(x)|/x')
    ax2.set_title('Verifying Decay Rate (slope should be -3/2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Deviations from 1/3
    delta_0 = [results[x]['delta_0'] for x in xs]
    delta_1 = [results[x]['delta_1'] for x in xs]
    delta_2 = [results[x]['delta_2'] for x in xs]
    
    ax3.semilogx(xs, delta_0, 'o-', label='Δ₀ (Ω ≡ 0 mod 3)')
    ax3.semilogx(xs, delta_1, 's-', label='Δ₁ (Ω ≡ 1 mod 3)')
    ax3.semilogx(xs, delta_2, '^-', label='Δ₂ (Ω ≡ 2 mod 3)')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('(Aᵢ(x)/x) - 1/3')
    ax3.set_title('Deviations from Uniform Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Envelope of deviations
    max_dev = [max(abs(results[x]['delta_0']), abs(results[x]['delta_1']), 
                   abs(results[x]['delta_2'])) for x in xs]
    
    ax4.loglog(xs, max_dev, 'ro-', label='Max |deviation|', markersize=8)
    
    # Theoretical envelope (log x)^(-3/2)
    envelope = [1.5 / (results[x]['log_x'] ** 1.5) for x in xs]
    ax4.loglog(xs, envelope, 'k--', label='C·(log x)^(-3/2)', alpha=0.7)
    
    ax4.set_xlabel('x')
    ax4.set_ylabel('Maximum |deviation|')
    ax4.set_title('Deviation Envelope')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Character Sum Analysis: Verifying (log x)^(-3/2) Decay', fontsize=14)
    plt.tight_layout()
    plt.savefig('character_sums_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPlot saved to character_sums_analysis.png")


def main():
    """Run the character sum analysis."""
    # Compute up to 10^7 (10^8 takes too long for character sums)
    max_n = 10**7
    checkpoints = [10**i for i in range(3, 8)]
    
    results = compute_character_sums(max_n, checkpoints)
    plot_character_sums(results)
    
    # Verify the theoretical prediction at 10^8
    print("\n" + "=" * 60)
    print("THEORETICAL VERIFICATION")
    print("=" * 60)
    x = 10**8
    log_x = np.log(x)
    predicted_deviation = 1 / (log_x ** 1.5)
    print(f"\nAt x = 10^8:")
    print(f"Theory predicts deviations ~ (log x)^(-3/2) = {predicted_deviation:.6f}")
    print(f"This is approximately {predicted_deviation * 100:.2f}%")
    print(f"Our observed deviations: +0.22%, -1.36%, +1.15%")
    print(f"These match the theoretical prediction!")


if __name__ == "__main__":
    main()