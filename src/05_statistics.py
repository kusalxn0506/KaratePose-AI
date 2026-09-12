"""
05_statistics.py

PURPOSE:
This module performs ALL statistical testing for Paper 1 of the KaratePose-AI research.
It contains functions for normality testing, Wilcoxon signed-rank tests, paired t-tests,
False Discovery Rate correction, Linear Mixed-Effects Models, Kruskal-Wallis, and 
reliability analysis.

DESIGNED FOR:
BSc Honours students who are NEW to these concepts. Extensive comments explain both
the 'how' and the 'why' of each statistical step.

RESEARCH NOTE:
These tests map directly to Section 6.5 of the research proposal.
"""

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, List, Tuple, Union, Optional
from src.config import STATS_CONFIG, PATHS

# =============================================================================
# 1. NORMALITY TESTING
# =============================================================================

def test_normality(data_a: np.ndarray, data_b: np.ndarray) -> Dict[str, Union[float, bool]]:
    """
    CONCEPT: Normality
    Many statistical tests (like the t-test) assume your data follows a "bell curve" 
    (Normal Distribution). If it doesn't, those tests can give false results.
    
    Why test the DIFFERENCES? 
    Because in a paired design (measuring the SAME person in Set A and Set B),
    we don't care if the raw scores are normal. We only care if the *effect* 
    (the difference between Set B and Set A) is normally distributed.
    
    MATH: difference = data_b - data_a
    
    INTERPRETATION:
    If p_value < 0.05, the differences are NOT normally distributed. 
    We must use the non-parametric Wilcoxon test.
    """
    differences = np.array(data_b) - np.array(data_a)
    
    # Handle edge case: if all differences are identical (e.g., all 0), variance is 0.
    if np.all(differences == differences[0]):
        return {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
    
    # Handle edge case: small N. Shapiro-Wilk requires at least N=3.
    if len(differences) < 3:
        return {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
        
    stat, p_value = stats.shapiro(differences)
    
    # STATS_CONFIG['normality_alpha'] is typically 0.05
    is_normal = bool(p_value >= STATS_CONFIG.get('normality_alpha', 0.05))
    
    return {
        'statistic': stat,
        'p_value': p_value,
        'is_normal': is_normal
    }


# =============================================================================
# 2. WILCOXON SIGNED-RANK TEST
# =============================================================================

def wilcoxon_test(data_a: np.ndarray, data_b: np.ndarray, alternative: str = 'two-sided') -> Dict:
    """
    CONCEPT: Wilcoxon Signed-Rank Test
    This is the non-parametric equivalent of the paired t-test. It does not assume
    the differences are normally distributed.
    
    How it works (Step-by-Step):
    1. Calculate the difference for each pair: d = data_b - data_a
    2. Ignore pairs where d = 0 (ties).
    3. Rank the ABSOLUTE differences from smallest to largest (1 is the smallest).
    4. Re-assign the signs to the ranks (positive or negative).
    5. Sum the positive ranks (W+) and the negative ranks (W-).
    
    What it actually measures: 
    It tests if the median of the differences is zero. In our case, it tests 
    if wearing the Gi (Set B) systematically increases or decreases jitter 
    compared to the control (Set A).
    
    CONCEPT: Rank-Biserial Correlation (Effect Size)
    MATH: r_rb = (W+ - W-) / (W+ + W-)
    INTERPRETATION:
    A value of 0.78 means that in 78% of the paired comparisons, the Gi condition 
    had higher jitter. It tells us how BIG the effect is, not just if it's significant.
    """
    diff = np.array(data_b) - np.array(data_a)
    diff = diff[diff != 0] # Remove zero differences
    
    if len(diff) < 2:
        return {'statistic': np.nan, 'p_value': np.nan, 'effect_size_r_rb': np.nan, 
                'effect_interpretation': 'N/A', 'hodges_lehmann_median_diff': np.nan, 
                'ci_lower': np.nan, 'ci_upper': np.nan}
                
    # Run the test
    stat, p_value = stats.wilcoxon(data_a, data_b, alternative=alternative)
    
    # Calculate Rank-Biserial Correlation using pingouin for simplicity and accuracy
    # pingouin automatically calculates W+, W-, and r_rb under the hood
    res = pg.wilcoxon(data_a, data_b, alternative=alternative)
    r_rb = res['RBC'].values[0]
    
    # Interpret Effect Size based on config thresholds
    abs_r = abs(r_rb)
    if abs_r >= STATS_CONFIG.get('effect_large', 0.5):
        interpretation = 'large'
    elif abs_r >= STATS_CONFIG.get('effect_medium', 0.3):
        interpretation = 'medium'
    elif abs_r >= STATS_CONFIG.get('effect_small', 0.1):
        interpretation = 'small'
    else:
        interpretation = 'trivial'
        
    # Calculate Hodges-Lehmann Median Difference
    # It is the median of all pairwise averages of differences
    n = len(diff)
    pairwise_averages = [(diff[i] + diff[j]) / 2.0 for i in range(n) for j in range(i, n)]
    hl_median = np.median(pairwise_averages)
    
    # Approximation for 95% CI of Hodges-Lehmann
    pairwise_averages.sort()
    M = len(pairwise_averages)
    # Expected variance for normal approximation
    z = 1.96 # 95% CI
    k = M / 2.0 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0) / 2.0
    k = int(np.floor(max(0, k)))
    ci_lower = pairwise_averages[k]
    ci_upper = pairwise_averages[M - 1 - k] if M - 1 - k < M else pairwise_averages[-1]
    
    return {
        'statistic': stat,
        'p_value': p_value,
        'effect_size_r_rb': r_rb,
        'effect_interpretation': interpretation,
        'hodges_lehmann_median_diff': hl_median,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


# =============================================================================
# 3. PAIRED T-TEST
# =============================================================================

def paired_ttest(data_a: np.ndarray, data_b: np.ndarray) -> Dict:
    """
    CONCEPT: Paired t-test
    This test is used as a SUPPLEMENTARY test when the normality assumption holds.
    It tests if the mean difference between paired observations is zero.
    
    CONCEPT: Cohen's d (Effect Size)
    MATH: d = (Mean_B - Mean_A) / SD_pooled
    INTERPRETATION:
    Tells us how many standard deviations apart the two means are.
    """
    if len(data_a) < 2:
        return {'statistic': np.nan, 'p_value': np.nan, 'effect_size_d': np.nan, 'effect_interpretation': 'N/A'}
        
    stat, p_value = stats.ttest_rel(data_a, data_b)
    
    # Calculate Cohen's d for paired samples
    # Usually computed as the mean of differences divided by the standard deviation of differences
    diff = np.array(data_b) - np.array(data_a)
    d = np.mean(diff) / np.std(diff, ddof=1)
    
    abs_d = abs(d)
    if abs_d >= 0.8: interpretation = 'large'
    elif abs_d >= 0.5: interpretation = 'medium'
    elif abs_d >= 0.2: interpretation = 'small'
    else: interpretation = 'trivial'
    
    return {
        'statistic': stat,
        'p_value': p_value,
        'effect_size_d': d,
        'effect_interpretation': interpretation
    }


# =============================================================================
# 4. BENJAMINI-HOCHBERG FDR CORRECTION
# =============================================================================

def benjamini_hochberg_correction(p_values: Union[List[float], np.ndarray], q: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    CONCEPT: Multiple Comparisons Problem & False Discovery Rate (FDR)
    Why do we need this?
    If you run 24 tests (e.g., 6 joints * 2 poses * 2 metrics) at alpha=0.05, 
    you expect ~1.2 false positives by pure chance (24 * 0.05 = 1.2). 
    We must adjust our p-values to control for this.
    
    Step-by-step procedure:
    1. Sort all p-values from smallest to largest.
    2. Rank them (1, 2, 3... m).
    3. Compare each p-value to a new threshold: (rank / total_tests) * q
    4. The largest p-value that is smaller than its threshold, and all p-values 
       smaller than it, are considered significant.
    """
    p_vals = np.array(p_values)
    m = len(p_vals)
    
    # If empty
    if m == 0:
        return np.array([]), np.array([])
        
    # Sort indices
    sort_idx = np.argsort(p_vals)
    sorted_p = p_vals[sort_idx]
    
    # Calculate thresholds: (rank / m) * q
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * q
    
    # Find the largest p-value that is less than or equal to its threshold
    below_threshold = sorted_p <= thresholds
    
    if not np.any(below_threshold):
        # No significant results
        is_significant = np.zeros(m, dtype=bool)
    else:
        # Index of the maximum p-value that satisfies the condition
        max_idx = np.max(np.where(below_threshold)[0])
        # Everything up to max_idx is considered significant
        is_significant_sorted = np.arange(m) <= max_idx
        
        # Restore original order
        is_significant = np.zeros(m, dtype=bool)
        is_significant[sort_idx] = is_significant_sorted
        
    # Calculate adjusted p-values
    adj_p = np.zeros(m)
    min_val = 1.0
    for i in range(m-1, -1, -1):
        # Adjusted p = min(p_adjusted_next, p * m / rank)
        val = (sorted_p[i] * m) / ranks[i]
        min_val = min(min_val, val)
        adj_p[sort_idx[i]] = min_val
        
    return adj_p, is_significant


# =============================================================================
# 5. LINEAR MIXED-EFFECTS MODEL
# =============================================================================

def run_lmm(pooled_df: pd.DataFrame, metric_column: str = 'median_jitter'):
    """
    CONCEPT: Linear Mixed-Effects Model (LMM)
    What is a mixed model?
    It accounts for the fact that some practitioners might naturally have higher 
    or lower jitter overall (random intercept), while still testing whether the 
    Gi makes jitter worse overall (fixed effect).
    
    MATH: metric ~ condition + (1|practitioner_id)
    - Fixed Effect: 'condition' (the thing we manipulated: Set A vs Set B)
    - Random Effect: 'practitioner_id' (the natural variance between humans)
    
    RESEARCH NOTE:
    Essential for combining data across the 3 participants without violating the 
    assumption of independence.
    """
    # Ensure categorical variables
    pooled_df['condition'] = pooled_df['condition'].astype(str)
    pooled_df['practitioner_id'] = pooled_df['practitioner_id'].astype(str)
    
    # Define the formula
    formula = f"{metric_column} ~ C(condition)"
    
    # Fit the model
    model = smf.mixedlm(formula, pooled_df, groups=pooled_df["practitioner_id"])
    result = model.fit()
    
    return result


# =============================================================================
# 6. KRUSKAL-WALLIS TEST (SKILL LEVEL)
# =============================================================================

def kruskal_wallis_skill_level(delta_df: pd.DataFrame) -> Dict:
    """
    CONCEPT: Kruskal-Wallis Test
    Tests whether the magnitude of jitter degradation (delta) differs across 
    more than 2 groups. Here, the groups are the 3 practitioners (who have 
    different skill levels).
    """
    # Group the delta values by practitioner
    groups = []
    for pid in delta_df['practitioner_id'].unique():
        group_data = delta_df[delta_df['practitioner_id'] == pid]['delta'].values
        groups.append(group_data)
        
    if len(groups) < 2:
        return {'statistic': np.nan, 'p_value': np.nan}
        
    stat, p_value = stats.kruskal(*groups)
    
    return {
        'statistic': stat,
        'p_value': p_value
    }


# =============================================================================
# 7. RELIABILITY METRICS
# =============================================================================

def compute_reliability(data_per_rep: np.ndarray, metric: str = 'cv') -> Dict:
    """
    CONCEPT: Reliability
    How consistent is the measurement?
    
    - CV% (Coefficient of Variation): (Standard Deviation / Mean) * 100
      Shows relative variability.
    - SEM (Standard Error of Measurement): SD * sqrt(1 - ICC)
      Shows the absolute error band around a single measurement.
    - MDC (Minimal Detectable Change): SEM * 1.96 * sqrt(2)
      The smallest change that is NOT due to measurement error.
    - ICC(3,1) (Intraclass Correlation Coefficient): 
      Used for inter-practitioner or test-retest reliability.
    """
    mean_val = np.mean(data_per_rep)
    sd_val = np.std(data_per_rep, ddof=1)
    
    # CV%
    cv = (sd_val / mean_val) * 100 if mean_val != 0 else np.nan
    
    # Just returning basic descriptive reliability here since ICC requires
    # a specific structured DataFrame format with raters/targets.
    return {
        'mean': mean_val,
        'sd': sd_val,
        'cv_percent': cv
    }


# =============================================================================
# 8. FULL ANALYSIS PIPELINE
# =============================================================================

def run_full_analysis(aggregated_data_a: np.ndarray, aggregated_data_b: np.ndarray) -> Dict:
    """
    Runs the complete statistical analysis pipeline:
    1. Normality test
    2. Wilcoxon (+ paired t-test if normal)
    3. Effect sizes
    4. BH-FDR correction (done outside if running multiple, but here simulated for 1 pair)
    """
    results = {}
    
    # 1. Normality
    normality = test_normality(aggregated_data_a, aggregated_data_b)
    results['normality'] = normality
    
    # 2. Wilcoxon
    wilcoxon = wilcoxon_test(aggregated_data_a, aggregated_data_b)
    results['wilcoxon'] = wilcoxon
    
    # 3. Paired T-Test
    ttest = paired_ttest(aggregated_data_a, aggregated_data_b)
    results['ttest'] = ttest
    
    # Print a comprehensive summary
    summary = f"""
    === STATISTICAL ANALYSIS SUMMARY ===
    Normality (Shapiro-Wilk): W={normality['statistic']:.4f}, p={normality['p_value']:.4f}, Normal={normality['is_normal']}
    
    Wilcoxon Signed-Rank Test: W={wilcoxon['statistic']:.4f}, p={wilcoxon['p_value']:.4f}
    Rank-Biserial Effect Size: {wilcoxon['effect_size_r_rb']:.4f} ({wilcoxon['effect_interpretation']})
    Hodges-Lehmann Median Diff: {wilcoxon['hodges_lehmann_median_diff']:.4f} (95% CI: [{wilcoxon['ci_lower']:.4f}, {wilcoxon['ci_upper']:.4f}])
    
    Paired t-test: t={ttest['statistic']:.4f}, p={ttest['p_value']:.4f}
    Cohen's d: {ttest['effect_size_d']:.4f} ({ttest['effect_interpretation']})
    ====================================
    """
    
    print(summary)
    
    return results


# =============================================================================
# 9. EXECUTION & SYNTHETIC DATA
# =============================================================================

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Generating SYNTHETIC data for pipeline verification...")
    # Create fake Set A (Control) and Set B (Gi/Experimental) data
    # Set A: lower jitter, normal(mean=0.01, sd=0.003)
    set_a_data = np.random.normal(loc=0.01, scale=0.003, size=10)
    # Set B: higher jitter, normal(mean=0.04, sd=0.01)
    set_b_data = np.random.normal(loc=0.04, scale=0.01, size=10)
    
    # Ensure no negative jitter by taking absolute value just in case
    set_a_data = np.abs(set_a_data)
    set_b_data = np.abs(set_b_data)
    
    print("\n--- Running Full Analysis ---")
    run_full_analysis(set_a_data, set_b_data)
    
    # Demonstrate LMM with synthetic DataFrame
    print("\n--- Demonstrating LMM ---")
    np.random.seed(42)
    n_reps = 30
    synth_df = pd.DataFrame({
        'practitioner_id': np.repeat(['P01', 'P02', 'P03'], 20),
        'condition': np.tile(np.repeat(['Set_A', 'Set_B'], 10), 3),
        'median_jitter': np.random.normal(loc=0.05, scale=0.02, size=60)
    })
    # Add an effect for Set_B
    synth_df.loc[synth_df['condition'] == 'Set_B', 'median_jitter'] += 0.03
    
    lmm_results = run_lmm(synth_df)
    print(lmm_results.summary())
    
    # Demonstrate Kruskal-Wallis with synthetic delta DataFrame
    print("\n--- Demonstrating Kruskal-Wallis ---")
    delta_df = pd.DataFrame({
        'practitioner_id': np.repeat(['P01', 'P02', 'P03'], 10),
        'delta': np.concatenate([
            np.random.normal(0.01, 0.005, 10), # P01
            np.random.normal(0.02, 0.005, 10), # P02
            np.random.normal(0.03, 0.005, 10)  # P03
        ])
    })
    kw_res = kruskal_wallis_skill_level(delta_df)
    print(f"Kruskal-Wallis stat: {kw_res['statistic']:.4f}, p-value: {kw_res['p_value']:.4f}")
    
    # Demonstrate BH-FDR correction
    print("\n--- Demonstrating BH-FDR Correction ---")
    p_vals = [0.01, 0.04, 0.03, 0.001, 0.15, 0.5]
    adj_p, sig = benjamini_hochberg_correction(p_vals, q=0.05)
    print("Original p-values:", p_vals)
    print("Adjusted p-values:", adj_p)
    print("Significant?", sig)
