"""
Week 4 Day 4: Final Summary and Documentation
Consolidate all results into final deliverables
"""
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import os

PROJECT_ROOT = "/home/ubuntu/music_uncertainty_project"

print("=" * 70)
print("Week 4 Day 4: Final Summary and Documentation")
print("=" * 70)
print(f"Timestamp: {datetime.now()}")
print()

# ============================================================================
# LOAD FINAL DATA
# ============================================================================
print("Loading final data...")

# Day 2 results
with open(f"{PROJECT_ROOT}/results/week4_day2_calibration_bins.json", 'r') as f:
    day2_results = json.load(f)

# Day 3 corrected results
with open(f"{PROJECT_ROOT}/results/week4_day3_temperature_scaling_corrected.json", 'r') as f:
    day3_results = json.load(f)

# Load test data for verification
test_y = np.load(f"{PROJECT_ROOT}/results/preprocessing/test_y.npy", allow_pickle=True)
test_probs = np.load(f"{PROJECT_ROOT}/results/mc_dropout/test_mean_probs.npy")

print(f"✓ Data loaded: N = {len(test_y)} samples")
print()

# ============================================================================
# FINAL METRICS TABLE
# ============================================================================
print("=" * 70)
print("FINAL METRICS TABLE")
print("=" * 70)

final_metrics = {
    'Metric': ['Accuracy', 'Mean Confidence', 'ECE', 'Confidence-Accuracy Gap'],
    'Before Calibration': [
        f"{day3_results['before_calibration']['accuracy']:.4f}",
        f"{day3_results['before_calibration']['mean_confidence']:.4f}",
        f"{day3_results['before_calibration']['ece']:.4f}",
        f"{day3_results['before_calibration']['confidence_accuracy_gap']:+.4f}"
    ],
    'After Calibration (T=1.60)': [
        f"{day3_results['after_calibration']['accuracy']:.4f}",
        f"{day3_results['after_calibration']['mean_confidence']:.4f}",
        f"{day3_results['after_calibration']['ece']:.4f}",
        f"{day3_results['after_calibration']['confidence_accuracy_gap']:+.4f}"
    ],
    'Change': [
        f"{day3_results['improvements']['accuracy_change']:+.4f}",
        f"{day3_results['before_calibration']['mean_confidence'] - day3_results['after_calibration']['mean_confidence']:+.4f}",
        f"-{day3_results['improvements']['ece_reduction']:.4f} ({-100*day3_results['improvements']['ece_reduction']/day3_results['before_calibration']['ece']:.1f}%)",
        f"{day3_results['improvements']['gap_reduction']:+.4f}"
    ]
}

df_metrics = pd.DataFrame(final_metrics)
print("\n" + df_metrics.to_string(index=False))
print()

# ============================================================================
# SAVE FINAL RELIABILITY DIAGRAMS (with consistent naming)
# ============================================================================
print("=" * 70)
print("SAVING FINAL RELIABILITY DIAGRAMS")
print("=" * 70)

def temperature_scale(probs, temperature):
    """Apply temperature scaling"""
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    logits = np.log(probs_clipped)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)

def plot_reliability_diagram(y_true, y_prob, title, save_path, n_bins=10):
    """Plot reliability diagram"""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = bin_edges[:-1] + 0.5 / n_bins
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        else:
            in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
            bin_counts.append(bin_count)
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)
    
    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)
    
    # Calculate ECE
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / len(y_true)) * np.abs(bin_accs[i] - bin_confs[i])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    valid = bin_counts > 0
    ax.bar(bin_centers[valid], bin_accs[valid], width=0.08, alpha=0.7,
           edgecolor='black', color='steelblue', label=f'Model (ECE={ece:.3f})')
    
    # Gap visualization
    for i in range(n_bins):
        if bin_counts[i] > 0:
            if bin_confs[i] > bin_accs[i]:
                ax.bar(bin_centers[i], bin_confs[i] - bin_accs[i], width=0.08,
                       bottom=bin_accs[i], alpha=0.4, color='red', edgecolor='red')
            elif bin_accs[i] > bin_confs[i]:
                ax.bar(bin_centers[i], bin_accs[i] - bin_confs[i], width=0.08,
                       bottom=bin_confs[i], alpha=0.4, color='green', edgecolor='green')
    
    ax.set_xlabel('Confidence', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add sample counts
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ax.text(bin_centers[i], 0.03, f'n={int(bin_counts[i])}',
                   ha='center', va='bottom', fontsize=9, rotation=90, color='darkblue')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return ece

# Before calibration
acc_before = day3_results['before_calibration']['accuracy']
ece_before = day3_results['before_calibration']['ece']
plot_reliability_diagram(
    test_y, test_probs,
    f'Reliability Diagram (Before Calibration)\nAccuracy = {acc_before:.3f}, ECE = {ece_before:.3f}',
    f'{PROJECT_ROOT}/figures/reliability_diagram_before.png'
)

# After calibration
test_probs_scaled = temperature_scale(test_probs, day3_results['optimal_temperature'])
acc_after = day3_results['after_calibration']['accuracy']
ece_after = day3_results['after_calibration']['ece']
temp = day3_results['optimal_temperature']
plot_reliability_diagram(
    test_y, test_probs_scaled,
    f'Reliability Diagram (After Temperature Scaling, T={temp:.2f})\nAccuracy = {acc_after:.3f}, ECE = {ece_after:.3f}',
    f'{PROJECT_ROOT}/figures/reliability_diagram_after.png'
)

print()

# ============================================================================
# WRITE FINAL CONCLUSION
# ============================================================================
print("=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

final_conclusion = f"""
Week 4: Calibration Analysis - Final Conclusion
===============================================

Dataset: DEAM Music Emotion Classification
Test samples: {len(test_y)}
Optimal temperature: T = {day3_results['optimal_temperature']:.2f}

KEY RESULTS
-----------

Calibration Metrics (Before → After):
- Accuracy: {day3_results['before_calibration']['accuracy']:.4f} → {day3_results['after_calibration']['accuracy']:.4f} (preserved)
- Mean Confidence: {day3_results['before_calibration']['mean_confidence']:.4f} → {day3_results['after_calibration']['mean_confidence']:.4f}
- ECE: {day3_results['before_calibration']['ece']:.4f} → {day3_results['after_calibration']['ece']:.4f} ({-100*day3_results['improvements']['ece_reduction']/day3_results['before_calibration']['ece']:.1f}% reduction)
- Confidence-Accuracy Gap: {day3_results['before_calibration']['confidence_accuracy_gap']:+.4f} → {day3_results['after_calibration']['confidence_accuracy_gap']:+.4f}

MAIN FINDINGS
-------------

1. Systematic Overconfidence Identified
   The model exhibited consistent overconfidence across medium-to-high confidence 
   ranges, with mean confidence (0.733) exceeding actual accuracy (0.642).

2. High-Confidence Errors Significant
   34% of all prediction errors were made with high confidence (>0.7), 
   concentrated in minority classes (calm, tense) and misclassified as happy.

3. Temperature Scaling Effective
   Post-hoc calibration with T=1.60 reduced ECE by 49% while perfectly 
   preserving prediction accuracy.

4. Calibration Improved Without Trade-off
   The calibration successfully brought confidence estimates closer to 
   actual accuracy without sacrificing model performance.

IMPLICATIONS
------------

This demonstrates that:
- The model had a genuine calibration problem distinct from accuracy issues
- Simple post-hoc methods can effectively address overconfidence
- Calibration improvement does not require predictive performance trade-offs

The reliability diagrams clearly show the shift from systematic overconfidence 
(red regions) to more balanced calibration after temperature scaling.
"""

print(final_conclusion)

# Save conclusion to file
with open(f"{PROJECT_ROOT}/week4_final_conclusion.txt", 'w') as f:
    f.write(final_conclusion)
print(f"Saved: {PROJECT_ROOT}/week4_final_conclusion.txt")
print()

# ============================================================================
# UPDATE WEEK4_SUMMARY.MD
# ============================================================================
print("=" * 70)
print("UPDATING WEEK4_SUMMARY.MD")
print("=" * 70)

week4_summary = f"""# Week 4: Calibration Analysis - Final Summary

**Project:** Music Emotion Classification with Uncertainty Quantification  
**Dataset:** DEAM ({len(test_y)} test samples)  
**Completed:** {datetime.now().strftime('%Y-%m-%d')}

---

## Executive Summary

This week addressed the **calibration problem** identified in Weeks 1-3. Through systematic analysis of high-confidence errors and reliability diagrams, we implemented temperature scaling and achieved a **49% reduction in Expected Calibration Error (ECE)** while maintaining prediction accuracy.

---

## Final Results

| Metric | Before Calibration | After Calibration (T={day3_results['optimal_temperature']:.2f}) | Change |
|--------|-------------------|---------------------------------------------------------------|--------|
| **Accuracy** | {day3_results['before_calibration']['accuracy']:.4f} | {day3_results['after_calibration']['accuracy']:.4f} | **Preserved** |
| **Mean Confidence** | {day3_results['before_calibration']['mean_confidence']:.4f} | {day3_results['after_calibration']['mean_confidence']:.4f} | -{day3_results['before_calibration']['mean_confidence'] - day3_results['after_calibration']['mean_confidence']:.4f} |
| **ECE** | {day3_results['before_calibration']['ece']:.4f} | {day3_results['after_calibration']['ece']:.4f} | **-49%** ↓ |
| **Conf-Acc Gap** | {day3_results['before_calibration']['confidence_accuracy_gap']:+.4f} | {day3_results['after_calibration']['confidence_accuracy_gap']:+.4f} | Corrected |

---

## Day-by-Day Progress

### Day 1: High-Confidence Error Analysis
- Identified that **34% of errors** were made with high confidence (>0.7)
- Errors concentrated in minority classes (calm, tense)
- Pattern: Minority samples → High confidence → Misclassified as happy

### Day 2: Reliability Diagram and ECE
- ECE before calibration: **0.098**
- Systematic overconfidence across 0.4-0.9 confidence range
- Mean confidence (0.733) > Accuracy (0.642)

### Day 3: Temperature Scaling
- Optimal temperature: **T = 1.60** (fitted on validation set)
- ECE after calibration: **0.050** (49% reduction)
- Accuracy perfectly preserved at **64.2%**

### Day 4: Final Documentation
- Consolidated all results
- Generated final reliability diagrams
- Documented conclusions

---

## Key Visualizations

| Figure | Description |
|--------|-------------|
| `reliability_diagram_before.png` | Pre-calibration reliability diagram (ECE = 0.098) |
| `reliability_diagram_after.png` | Post-calibration reliability diagram (ECE = 0.050) |

---

## Main Conclusion

> Temperature scaling with T=1.60 significantly improved model calibration, reducing ECE from **0.098 to 0.050** (49% reduction) while perfectly preserving accuracy at **64.2%**. The calibration successfully corrected systematic overconfidence in medium-to-high confidence ranges, bringing the model's confidence estimates much closer to its actual accuracy.

This demonstrates that:
1. The model had a genuine calibration problem (not just poor accuracy)
2. Simple post-hoc calibration can effectively address overconfidence
3. Calibration improvement does not require sacrificing predictive performance

---

## Files Generated

| File | Description |
|------|-------------|
| `week4_final_conclusion.txt` | Final conclusion paragraph |
| `week4_summary.md` | This summary document |
| `reliability_diagram_before.png` | Pre-calibration reliability diagram |
| `reliability_diagram_after.png` | Post-calibration reliability diagram |
| `results/week4_day2_calibration_bins.json` | Day 2 detailed results |
| `results/week4_day3_temperature_scaling_corrected.json` | Day 3 detailed results |

---

## Data Consistency Note

All Week 4 analyses were conducted on the **271-sample test set** from the main project directory (`music_uncertainty_project`) to maintain consistency with Weeks 1-3.

"""

with open(f"{PROJECT_ROOT}/week4_summary.md", 'w') as f:
    f.write(week4_summary)
print(f"Saved: {PROJECT_ROOT}/week4_summary.md")
print()

# ============================================================================
# SAVE FINAL METRICS JSON
# ============================================================================
print("=" * 70)
print("SAVING FINAL METRICS")
print("=" * 70)

final_output = {
    'week': 'Week 4',
    'status': 'COMPLETE',
    'timestamp': str(datetime.now()),
    'dataset': 'DEAM',
    'n_test_samples': len(test_y),
    'optimal_temperature': day3_results['optimal_temperature'],
    'final_metrics': {
        'before_calibration': day3_results['before_calibration'],
        'after_calibration': day3_results['after_calibration'],
        'improvements': day3_results['improvements']
    },
    'key_findings': [
        '34% of errors were high-confidence errors',
        'Systematic overconfidence across 0.4-0.9 confidence range',
        'ECE reduced by 49% with temperature scaling (T=1.60)',
        'Accuracy perfectly preserved at 64.2%'
    ],
    'deliverables': [
        'week4_summary.md',
        'week4_final_conclusion.txt',
        'reliability_diagram_before.png',
        'reliability_diagram_after.png'
    ]
}

with open(f"{PROJECT_ROOT}/results/week4_final.json", 'w') as f:
    json.dump(final_output, f, indent=2)
print(f"Saved: {PROJECT_ROOT}/results/week4_final.json")
print()

print("=" * 70)
print("DAY 4 COMPLETE - WEEK 4 FINISHED")
print("=" * 70)
print("\nFinal deliverables:")
print("  ✓ week4_summary.md")
print("  ✓ week4_final_conclusion.txt")
print("  ✓ reliability_diagram_before.png")
print("  ✓ reliability_diagram_after.png")
print("  ✓ results/week4_final.json")
