# Omega Mod 3 Distribution Analysis

This repository contains code and data for analyzing the distribution of the prime omega function Ω(n) modulo 3.

## Repository Structure

### Core Analysis Code
- `omega_mod3_analysis.py` - Main analysis script with clean, documented implementation
- `omega_mod3_optimized.py` - Performance-optimized version using Numba JIT compilation

### Papers and Documentation  
- `omega_mod3_paper_revised.md` - Revised paper addressing finite bias phenomenon
- `theoretical_vs_observed.md` - Summary of theoretical expectations vs. computational results
- `omega_mod3_paper_final.md` - Original paper draft (for historical reference)

### Visualization Scripts
- `omega_mod3_visualizations.py` - Scripts for creating plots and figures

### Data Files
- `omega_mod3_results.json` - Computational results up to 10^8
- `omega_mod3_extended_results.json` - Extended analysis results (if available)

### Historical/Archival Files
These files show the research progression but contain outdated comments:
- `omega_mod3_investigation.py`
- `omega_mod3_extended.py` 
- `omega_mod3_deep_dive.py`
- `omega_mod3_mathematical_theory.py`

## Quick Start

To reproduce the main results:

```bash
python omega_mod3_analysis.py
```

This will:
1. Compute Ω(n) mod 3 distribution up to 10^8
2. Generate statistical analysis
3. Create visualizations
4. Save results to JSON

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Numba (for optimized version)

## Key Findings

At n = 10^8, we observe:
- Ω(n) ≡ 0 (mod 3): 33.55% (expected: 33.33%)
- Ω(n) ≡ 1 (mod 3): 31.97% (expected: 33.33%) 
- Ω(n) ≡ 2 (mod 3): 34.48% (expected: 33.33%)

This finite bias is statistically significant (χ² ≈ 96,513) but must theoretically vanish as n → ∞.

## Citation

If you use this code or data, please cite:
```
Sudoma, O. & Claude (Anthropic). (2025). On the Distribution of Ω(n) Modulo 3: A Study of Finite Bias.
```

## License

[Specify your preferred license]

## Contact

[Your contact information]