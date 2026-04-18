# Can We Trust a Music Classifier's Confidence?

*A 4-week project on uncertainty and calibration in music emotion classification*

---

## Why I Did This

I got into this after trying a music app that claimed to detect my mood from what I was listening to. It was wrong half the time, but always sounded super confident. That bugged me—so I wanted to see if I could make a classifier that knows when it's guessing.

## What I Built

A 4-class music emotion classifier (happy / calm / tense / sad) using the [DEAM dataset](https://cvml.unige.ch/databases/DEAM/). I trained an MLP, then added MC Dropout to measure uncertainty, and finally used temperature scaling to fix overconfidence.

**The twist**: I didn't just care about accuracy. I wanted to know if the model's "I'm 90% sure" actually meant 90% accuracy.

## What I Found

Honestly? Mixed results.

**Good news**: MC Dropout worked. Wrong predictions had way higher entropy (0.87 vs 0.57, p < 0.001). Borderline songs—those near emotion boundaries—were also more uncertain. The model often "knew" when it was confused.

**Bad news**: 34% of errors were made with high confidence (>70%). These were mostly minority classes (calm, tense) getting misclassified as "happy" with false confidence. The model didn't know what it didn't know.

**Deeper dive**: I also checked whether calibration was the same across all four emotion classes. It wasn't—sad had the highest class-wise calibration error, calm was somewhat underconfident, and tense was actually the best-calibrated. Plus, I tested a continuous "distance to boundary" measure (instead of just binary borderline/non-borderline). The trend was in the right direction, but weak. So the simple borderline split turned out to be more informative than the fancier continuous version.

**The fix**: Temperature scaling with T=1.60 brought confidence in line with reality—dropped ECE from 9.8% to 5.0% without hurting accuracy.

| Before | After (T=1.60) |
|--------|----------------|
| Accuracy: 64.2% | Same: 64.2% |
| Mean confidence: 73.3% | Dropped to: 61.8% |
| ECE: 9.8% | Improved: 5.0% |

## What Surprised Me

I thought uncertainty estimation would catch most bad predictions. It didn't. The model could be very confident and very wrong—especially on minority classes. That was the main lesson: **uncertainty and calibration are different problems**. You need both.

## Project Files

```
├── mini_paper_English.md   # Full write-up (English)
├── mini_paper_Chinese.md   # Chinese version
├── README.md               # This file
├── CODE_README.md          # Code documentation
├── code/                   # Python scripts
│   ├── week4_day2_corrected.py    # ECE and reliability diagrams
│   ├── week4_day3_corrected.py    # Temperature scaling
│   └── week4_day4_final.py        # Summary and figures
├── figures/                # Plots and diagrams
│   ├── entropy_correct_vs_wrong_v2.png
│   ├── entropy_borderline_vs_nonborderline_v2.png
│   ├── boundary_distance_vs_entropy.png
│   ├── uncertainty_vs_accuracy_v3.png
│   ├── high_confidence_errors_class_dist_v2.png
│   ├── confusion_matrix_mlp_test.png
│   ├── valence_arousal_scatter.png
│   ├── class_distribution.png
│   ├── reliability_diagram_before.png
│   └── reliability_diagram_after.png
└── results/                # JSON data
    ├── week4_day2_calibration_bins.json
    ├── week4_day3_temperature_scaling_corrected.json
    ├── week4_final.json
    └── calibration/        # New calibration analyses
        ├── classwise_calibration_summary.csv
        ├── boundary_distance_entropy.csv
        └── boundary_distance_entropy_groups.csv
```

## Some Plots

**Uncertainty actually works** — wrong predictions are more uncertain:

![Entropy comparison](figures/entropy_correct_vs_wrong_v2.png)

**But not enough** — 34% of errors were high-confidence:

![High confidence errors](figures/high_confidence_errors_class_dist_v2.png)

**Calibration fixes it** — before and after temperature scaling:

![Before](figures/reliability_diagram_before.png)
![After](figures/reliability_diagram_after.png)

## What I'd Do Differently

- Try **class-specific temperature scaling** to address the uneven calibration across categories
- Try more sophisticated calibration methods (Platt scaling, isotonic regression)
- Experiment with different uncertainty metrics (mutual information, variation ratios)
- Test on a more balanced dataset—the class imbalance here definitely hurt

## Tools

Python, PyTorch, NumPy, Pandas, Matplotlib, SciPy. Nothing fancy.

## Background Reading

- Gal & Ghahramani (2016) — MC Dropout paper
- Guo et al. (2017) — temperature scaling
- Aljanaki et al. (2017) — DEAM dataset

---

*Done as an independent project exploring uncertainty in ML. Not perfect, but I learned a lot about why "accuracy" isn't the whole story.*
