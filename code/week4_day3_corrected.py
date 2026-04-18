"""
Week 4 Day 3 (CORRECTED): Temperature Scaling Calibration
Using SAVED test probabilities from Day 2 (not regenerating)
"""
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# SETUP
# ============================================================================
PROJECT_ROOT = "/home/ubuntu/music_uncertainty_project"
print("=" * 70)
print("Week 4 Day 3: Temperature Scaling (CORRECTED)")
print("=" * 70)
print(f"Timestamp: {datetime.now()}")
print(f"Project root: {PROJECT_ROOT}")
print()

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# LOAD DATA - USING SAVED PROBABILITIES FROM DAY 2
# ============================================================================
print("Loading data...")
print("Using SAVED test probabilities from Day 2 (test_mean_probs.npy)")

# Test set - USE SAVED PROBABILITIES (do not regenerate)
test_y = np.load(f"{PROJECT_ROOT}/results/preprocessing/test_y.npy", allow_pickle=True)
test_mean_probs = np.load(f"{PROJECT_ROOT}/results/mc_dropout/test_mean_probs.npy")

print(f"✓ Test set: {len(test_y)} samples")
print(f"✓ Test probs shape: {test_mean_probs.shape}")

# Verify matches Day 2
test_preds = np.argmax(test_mean_probs, axis=1)
test_acc = np.mean(test_preds == test_y)
test_conf = np.mean(np.max(test_mean_probs, axis=1))

print(f"\nVerification (should match Day 2):")
print(f"  Accuracy: {test_acc:.4f} (Day 2: 0.6421)")
print(f"  Mean Confidence: {test_conf:.4f} (Day 2: 0.7325)")

assert np.isclose(test_acc, 0.6421, rtol=1e-3), "Accuracy doesn't match Day 2!"
assert np.isclose(test_conf, 0.7325, rtol=1e-3), "Mean confidence doesn't match Day 2!"
print("  ✓ Matches Day 2 - using correct data")

# Validation set - need to generate MC dropout predictions
val_X = np.load(f"{PROJECT_ROOT}/results/preprocessing/val_X_scaled.npy")
val_y = np.load(f"{PROJECT_ROOT}/results/preprocessing/val_y.npy", allow_pickle=True)
print(f"✓ Validation set: {len(val_y)} samples")
print()

# ============================================================================
# DEFINE MODEL AND GENERATE VALIDATION PREDICTIONS
# ============================================================================
print("Generating validation MC Dropout predictions...")

class MLP_MC_Dropout(torch.nn.Module):
    def __init__(self, input_dim=520, hidden_dim=128, output_dim=4, dropout_rate=0.3):
        super(MLP_MC_Dropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate
    
    def forward(self, x, use_dropout=True):
        x = self.fc1(x)
        x = F.relu(x)
        if use_dropout:
            x = F.dropout(x, p=self.dropout_rate, training=True)
        x = self.fc2(x)
        return x
    
    def predict_mc_dropout(self, x, n_samples=30):
        self.train()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x, use_dropout=True)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        return np.array(predictions)

# Load model
model = MLP_MC_Dropout(input_dim=520, hidden_dim=128, output_dim=4, dropout_rate=0.3)
model.load_state_dict(torch.load(f"{PROJECT_ROOT}/results/mc_dropout/mlp_mc_dropout_v2.pth", map_location='cpu'))

# Generate validation predictions
val_X_tensor = torch.FloatTensor(val_X)
val_mc_probs = model.predict_mc_dropout(val_X_tensor, n_samples=30)
val_mean_probs = val_mc_probs.mean(axis=0)

print(f"✓ Validation probs shape: {val_mean_probs.shape}")
print()

# ============================================================================
# TEMPERATURE SCALING FUNCTIONS
# ============================================================================
def temperature_scale(probs, temperature):
    """Apply temperature scaling to probabilities"""
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    logits = np.log(probs_clipped)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits)
    scaled_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return scaled_probs

def nll_loss(probs, labels):
    """Negative log-likelihood loss"""
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    n_samples = len(labels)
    return -np.mean(np.log(probs_clipped[np.arange(n_samples), labels]))

def ece_score(probs, labels, n_bins=10):
    """Calculate ECE"""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        else:
            in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / len(labels)) * np.abs(bin_acc - bin_conf)
    
    return ece

# ============================================================================
# FIND OPTIMAL TEMPERATURE ON VALIDATION SET
# ============================================================================
print("=" * 70)
print("TEMPERATURE SCALING")
print("=" * 70)
print("\nSearching for optimal temperature T on validation set...")

temperatures = np.arange(0.5, 2.5, 0.05)
best_temp = 1.0
best_nll = float('inf')

results = []
for temp in temperatures:
    scaled_probs = temperature_scale(val_mean_probs, temp)
    nll = nll_loss(scaled_probs, val_y)
    ece = ece_score(scaled_probs, val_y)
    
    results.append({
        'temperature': float(temp),
        'nll': float(nll),
        'ece': float(ece)
    })
    
    if nll < best_nll:
        best_nll = nll
        best_temp = temp

print(f"\nOptimal temperature: T = {best_temp:.2f}")
print(f"Validation NLL: {best_nll:.4f}")

# ============================================================================
# APPLY TO TEST SET (USING SAVED DAY 2 PROBABILITIES)
# ============================================================================
print("\n" + "=" * 70)
print("APPLYING TO TEST SET")
print("=" * 70)
print("Using SAVED test probabilities from Day 2")

# Before calibration (Day 2 baseline)
test_pred_before = np.argmax(test_mean_probs, axis=1)
test_acc_before = np.mean(test_pred_before == test_y)
test_conf_before = np.mean(np.max(test_mean_probs, axis=1))
test_ece_before = ece_score(test_mean_probs, test_y)

print(f"\nBEFORE Temperature Scaling (Day 2 baseline):")
print(f"  Accuracy: {test_acc_before:.4f}")
print(f"  Mean Confidence: {test_conf_before:.4f}")
print(f"  ECE: {test_ece_before:.4f}")

# After calibration
test_probs_scaled = temperature_scale(test_mean_probs, best_temp)
test_pred_after = np.argmax(test_probs_scaled, axis=1)
test_acc_after = np.mean(test_pred_after == test_y)
test_conf_after = np.mean(np.max(test_probs_scaled, axis=1))
test_ece_after = ece_score(test_probs_scaled, test_y)

print(f"\nAFTER Temperature Scaling (T={best_temp:.2f}):")
print(f"  Accuracy: {test_acc_after:.4f}")
print(f"  Mean Confidence: {test_conf_after:.4f}")
print(f"  ECE: {test_ece_after:.4f}")

print(f"\nIMPROVEMENT:")
print(f"  ECE: {test_ece_before:.4f} → {test_ece_after:.4f} (Δ = {test_ece_before - test_ece_after:+.4f})")
print(f"  Accuracy: {test_acc_before:.4f} → {test_acc_after:.4f} (Δ = {test_acc_after - test_acc_before:+.4f})")
print(f"  Confidence-Accuracy Gap: {test_conf_before - test_acc_before:+.4f} → {test_conf_after - test_acc_after:+.4f}")

# Verify accuracy unchanged (temperature scaling preserves predictions)
assert np.array_equal(test_pred_before, test_pred_after), "Temperature scaling changed predictions!"
print("\n✓ Verified: Temperature scaling preserves all predictions (only changes confidence)")

# ============================================================================
# GENERATE RELIABILITY DIAGRAMS
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING RELIABILITY DIAGRAMS")
print("=" * 70)

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
    
    return fig, ece

# Before calibration (Day 2 baseline)
plot_reliability_diagram(
    test_y, test_mean_probs,
    f'Reliability Diagram (Before Calibration)\nECE = {test_ece_before:.3f}, Accuracy = {test_acc_before:.3f}',
    f'{PROJECT_ROOT}/figures/reliability_before_temp_scaling_corrected.png'
)

# After calibration
plot_reliability_diagram(
    test_y, test_probs_scaled,
    f'Reliability Diagram (After Temperature Scaling, T={best_temp:.2f})\nECE = {test_ece_after:.3f}, Accuracy = {test_acc_after:.3f}',
    f'{PROJECT_ROOT}/figures/reliability_after_temp_scaling_corrected.png'
)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

day3_results = {
    'day': 'Week 4 Day 3 (CORRECTED)',
    'timestamp': str(datetime.now()),
    'project_root': PROJECT_ROOT,
    'note': 'Using SAVED test probabilities from Day 2 (test_mean_probs.npy). Validation probs regenerated with MC dropout.',
    'n_val_samples': int(len(val_y)),
    'n_test_samples': int(len(test_y)),
    'optimal_temperature': float(best_temp),
    'temperature_search_range': [0.5, 2.5],
    'validation_nll': float(best_nll),
    'before_calibration': {
        'accuracy': float(test_acc_before),
        'mean_confidence': float(test_conf_before),
        'ece': float(test_ece_before),
        'confidence_accuracy_gap': float(test_conf_before - test_acc_before)
    },
    'after_calibration': {
        'accuracy': float(test_acc_after),
        'mean_confidence': float(test_conf_after),
        'ece': float(test_ece_after),
        'confidence_accuracy_gap': float(test_conf_after - test_acc_after)
    },
    'improvements': {
        'ece_reduction': float(test_ece_before - test_ece_after),
        'accuracy_change': float(test_acc_after - test_acc_before),
        'gap_reduction': float((test_conf_before - test_acc_before) - (test_conf_after - test_acc_after))
    },
    'temperature_search_results': results
}

json_path = f"{PROJECT_ROOT}/results/week4_day3_temperature_scaling_corrected.json"
with open(json_path, 'w') as f:
    json.dump(day3_results, f, indent=2)

print(f"Saved: {json_path}")

print("\n" + "=" * 70)
print("DAY 3 COMPLETE (CORRECTED)")
print("=" * 70)
print(f"\nFinal Result:")
print(f"  Optimal Temperature: T = {best_temp:.2f}")
print(f"  ECE improved: {test_ece_before:.4f} → {test_ece_after:.4f} ({100*(test_ece_before-test_ece_after)/test_ece_before:.1f}% reduction)")
print(f"  Accuracy preserved: {test_acc_before:.4f} → {test_acc_after:.4f}")
print(f"\n✓ Day 3 now aligned with Day 2 baseline")
