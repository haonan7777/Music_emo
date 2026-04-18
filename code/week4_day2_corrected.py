"""
Week 4 Day 2 (CORRECTED): Confidence Bins and Reliability Diagram
Based on music_uncertainty_project with 271 samples
"""
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# SETUP: Explicitly define project root and verify data source
# ============================================================================
PROJECT_ROOT = "/home/ubuntu/music_uncertainty_project"
print("=" * 70)
print("Week 4 Day 2: Confidence Bins & Reliability Diagram")
print("=" * 70)
print(f"Timestamp: {datetime.now()}")
print(f"Project root: {PROJECT_ROOT}")
print()

# Load data files
test_y_path = f"{PROJECT_ROOT}/results/preprocessing/test_y.npy"
test_mean_probs_path = f"{PROJECT_ROOT}/results/mc_dropout/test_mean_probs.npy"
calibration_csv_path = f"{PROJECT_ROOT}/results/calibration/calibration_input_table.csv"

print("Loading data files:")
print(f"  - {test_y_path}")
print(f"  - {test_mean_probs_path}")
print(f"  - {calibration_csv_path}")

y_true = np.load(test_y_path, allow_pickle=True)
y_prob = np.load(test_mean_probs_path)
cal_df = pd.read_csv(calibration_csv_path)

# Verify consistency
assert len(y_true) == 271, f"Expected 271 samples, got {len(y_true)}"
assert y_prob.shape[0] == 271, f"Expected 271 probs, got {y_prob.shape[0]}"
assert len(cal_df) == 271, f"Expected 271 rows, got {len(cal_df)}"

print(f"\n✓ Data loaded successfully: N = {len(y_true)} samples")
print()

# ============================================================================
# BASIC STATISTICS
# ============================================================================
confidences = np.max(y_prob, axis=1)
predictions = np.argmax(y_prob, axis=1)
accuracies = (predictions == y_true).astype(float)

n_samples = len(y_true)
n_correct = int(accuracies.sum())
n_errors = n_samples - n_correct

print("=" * 70)
print("BASIC STATISTICS")
print("=" * 70)
print(f"Total samples: {n_samples}")
print(f"Correct: {n_correct}")
print(f"Errors: {n_errors}")
print(f"Accuracy: {n_correct/n_samples:.4f}")
print(f"Mean confidence: {np.mean(confidences):.4f}")
print(f"Mean entropy: {cal_df['entropy'].mean():.4f}")
print()

# ============================================================================
# PART 1: CONFIDENCE BINS (10 equal-width bins)
# ============================================================================
print("=" * 70)
print("PART 1: CONFIDENCE BIN STATISTICS (10 bins)")
print("=" * 70)

n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_lowers = bin_edges[:-1]
bin_uppers = bin_edges[1:]

bin_stats = []

for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
    # Define bin (upper bound inclusive for last bin)
    if i == n_bins - 1:
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
    else:
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
    
    bin_count = np.sum(in_bin)
    
    if bin_count > 0:
        bin_acc = np.mean(accuracies[in_bin])
        bin_conf = np.mean(confidences[in_bin])
        bin_error_count = int(bin_count * (1 - bin_acc))
        gap = bin_conf - bin_acc  # positive = overconfident
    else:
        bin_acc = 0
        bin_conf = 0
        bin_error_count = 0
        gap = 0
    
    bin_stats.append({
        'bin_id': i + 1,
        'range': f"({bin_lower:.1f}, {bin_upper:.1f}]",
        'count': int(bin_count),
        'accuracy': float(bin_acc),
        'confidence': float(bin_conf),
        'error_count': bin_error_count,
        'gap': float(gap)
    })

# Print bin statistics table
print(f"\n{'Bin':>4} | {'Range':>12} | {'Count':>6} | {'%Total':>7} | {'Accuracy':>9} | {'Confidence':>10} | {'Gap':>7} | {'Errors':>6}")
print("-" * 95)

for stat in bin_stats:
    pct_total = 100 * stat['count'] / n_samples
    gap_str = f"+{stat['gap']:.3f}" if stat['gap'] > 0 else f"{stat['gap']:.3f}"
    print(f"{stat['bin_id']:>4} | {stat['range']:>12} | {stat['count']:>6} | {pct_total:>7.1f}% | {stat['accuracy']:>9.3f} | {stat['confidence']:>10.3f} | {gap_str:>7} | {stat['error_count']:>6}")

print("-" * 95)

# Summary
non_empty_bins = [s for s in bin_stats if s['count'] > 0]
print(f"\nBins with samples: {len(non_empty_bins)}/10")

high_conf_bins = [s for s in bin_stats if s['count'] > 0 and s['bin_id'] >= 7]  # 0.6-1.0
high_conf_samples = sum(s['count'] for s in high_conf_bins)
print(f"Samples in confidence > 0.6: {high_conf_samples} ({100*high_conf_samples/n_samples:.1f}%)")
print()

# ============================================================================
# PART 2: ECE AND MCE
# ============================================================================
print("=" * 70)
print("PART 2: CALIBRATION METRICS")
print("=" * 70)

ece = 0.0
for stat in bin_stats:
    if stat['count'] > 0:
        weight = stat['count'] / n_samples
        gap = abs(stat['confidence'] - stat['accuracy'])
        ece += weight * gap

mce = max([abs(s['confidence'] - s['accuracy']) for s in bin_stats if s['count'] > 0])

print(f"ECE (Expected Calibration Error): {ece:.4f}")
print(f"MCE (Maximum Calibration Error):  {mce:.4f}")
print()

# ============================================================================
# PART 3: RELIABILITY DIAGRAM
# ============================================================================
print("=" * 70)
print("PART 3: GENERATING RELIABILITY DIAGRAM")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 8))

# Perfect calibration line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

# Bin centers for plotting
bin_centers = np.linspace(0, 1, n_bins + 1)[:-1] + 0.5 / n_bins

# Plot bars - only for bins with samples
valid_bins = [s for s in bin_stats if s['count'] > 0]
valid_centers = [bin_centers[s['bin_id']-1] for s in valid_bins]
valid_accs = [s['accuracy'] for s in valid_bins]
valid_confs = [s['confidence'] for s in valid_bins]
valid_counts = [s['count'] for s in valid_bins]

# Accuracy bars
bars = ax.bar(valid_centers, valid_accs, width=0.08, alpha=0.7, 
              edgecolor='black', color='steelblue', 
              label=f'Model Accuracy (ECE={ece:.3f})')

# Gap visualization
for center, acc, conf, count in zip(valid_centers, valid_accs, valid_confs, valid_counts):
    if conf > acc:
        # Overconfident - show red gap
        ax.bar(center, conf - acc, width=0.08, bottom=acc, 
               alpha=0.4, color='red', edgecolor='red')
    elif acc > conf:
        # Underconfident - show green gap
        ax.bar(center, acc - conf, width=0.08, bottom=conf,
               alpha=0.4, color='green', edgecolor='green')

ax.set_xlabel('Confidence', fontsize=13)
ax.set_ylabel('Accuracy', fontsize=13)
ax.set_title('Reliability Diagram (Before Calibration)\nMusic Emotion Classification (N=271)', fontsize=14)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

# Add sample count labels
for center, count in zip(valid_centers, valid_counts):
    ax.text(center, 0.03, f'n={count}', ha='center', va='bottom', 
            fontsize=9, rotation=90, color='darkblue')

plt.tight_layout()
output_path = f"{PROJECT_ROOT}/figures/reliability_diagram_week4_day2.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
print()

# ============================================================================
# PART 4: HIGH CONFIDENCE ANALYSIS
# ============================================================================
print("=" * 70)
print("PART 4: HIGH CONFIDENCE ANALYSIS")
print("=" * 70)

# High confidence (>= 0.8)
high_conf_threshold = 0.8
high_conf_mask = confidences >= high_conf_threshold
high_conf_acc = accuracies[high_conf_mask].mean() if high_conf_mask.sum() > 0 else 0
high_conf_mean_conf = confidences[high_conf_mask].mean() if high_conf_mask.sum() > 0 else 0
high_conf_errors = ((predictions != y_true) & high_conf_mask).sum()

print(f"\nHigh confidence (≥{high_conf_threshold}) analysis:")
print(f"  Samples: {high_conf_mask.sum()} ({100*high_conf_mask.sum()/n_samples:.1f}%)")
print(f"  Mean confidence: {high_conf_mean_conf:.3f}")
print(f"  Actual accuracy: {high_conf_acc:.3f}")
print(f"  Gap: {high_conf_mean_conf - high_conf_acc:+.3f}")
print(f"  Errors in this group: {high_conf_errors}")
print(f"  % of all errors: {100*high_conf_errors/n_errors:.1f}%")

# Confidence distribution
print(f"\nConfidence distribution:")
print(f"  Min: {confidences.min():.3f}")
print(f"  Max: {confidences.max():.3f}")
print(f"  Median: {np.median(confidences):.3f}")
print(f"  Std: {confidences.std():.3f}")
print()

# ============================================================================
# PART 5: SAVE RESULTS
# ============================================================================
print("=" * 70)
print("PART 5: SAVING RESULTS")
print("=" * 70)

day2_results = {
    'day': 'Week 4 Day 2 (CORRECTED)',
    'timestamp': str(datetime.now()),
    'project_root': PROJECT_ROOT,
    'n_samples': n_samples,
    'accuracy': float(n_correct / n_samples),
    'mean_confidence': float(np.mean(confidences)),
    'mean_entropy': float(cal_df['entropy'].mean()),
    'ece': float(ece),
    'mce': float(mce),
    'bin_statistics': bin_stats,
    'high_confidence_analysis': {
        'threshold': high_conf_threshold,
        'n_samples': int(high_conf_mask.sum()),
        'mean_confidence': float(high_conf_mean_conf),
        'accuracy': float(high_conf_acc),
        'gap': float(high_conf_mean_conf - high_conf_acc),
        'errors': int(high_conf_errors),
        'pct_of_all_errors': float(100 * high_conf_errors / n_errors)
    },
    'key_findings': {
        'overconfidence_detected': bool(np.mean(confidences) > n_correct / n_samples),
        'confidence_accuracy_gap': float(np.mean(confidences) - n_correct / n_samples),
        'most_populated_bins': [s['bin_id'] for s in sorted(bin_stats, key=lambda x: -x['count'])[:3]],
        'high_conf_errors_exist': bool(high_conf_mask.sum() > 0 and high_conf_acc < 1.0)
    },
    'note': 'An incorrect calibration run was briefly produced from a different project directory (music_emotion_project, 361 samples). This was discarded, and all Week 4 analyses were re-run using the correct main project directory (music_uncertainty_project, 271 samples) to maintain consistency with Weeks 1-3.'
}

json_path = f"{PROJECT_ROOT}/results/week4_day2_calibration_bins.json"
with open(json_path, 'w') as f:
    json.dump(day2_results, f, indent=2)

print(f"Saved: {json_path}")

print("\n" + "=" * 70)
print("DAY 2 COMPLETE (CORRECTED - 271 SAMPLES)")
print("=" * 70)
