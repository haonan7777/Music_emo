# Music Emotion Classification - Week 4 Calibration Analysis

This repository contains the code and results for my Week 4 project on calibration analysis in music emotion classification. The main Python scripts are in the `code/` folder.

## What This Project Does

I trained a music emotion classifier and then looked at two things:
1. Whether the model knows when it's uncertain (using MC Dropout)
2. Whether the model's confidence scores are actually reliable (using temperature scaling)

The main finding was that the model was overconfident (mean confidence 73% vs accuracy 64%), and temperature scaling with T=1.60 fixed this without hurting performance.

## Main Code Files

All code is in the `code/` folder:

| File | What It Does |
|------|--------------|
| `code/week4_day2_corrected.py` | Calculates confidence bins and ECE (Expected Calibration Error) |
| `code/week4_day3_corrected.py` | Implements temperature scaling to fix overconfidence |
| `code/week4_day4_final.py` | Generates final figures and summary statistics |

## How to Run

```bash
# Week 4 Day 2: confidence bins and reliability diagram
python code/week4_day2_corrected.py

# Week 4 Day 3: temperature scaling
python code/week4_day3_corrected.py

# Week 4 Day 4: final summary
python code/week4_day4_final.py
```

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- SciPy (for statistical tests)

## Data

The code expects data from the DEAM dataset with MC Dropout predictions already saved. The main files used are:
- `test_mean_probs.npy` - MC Dropout predictions (30 samples)
- `test_y.npy` - true labels
- `calibration_input_table.csv` - metadata including entropy and borderline flags

## Key Results

| Metric | Before | After (T=1.60) |
|--------|--------|----------------|
| Accuracy | 64.2% | 64.2% |
| Mean Confidence | 73.3% | 61.8% |
| ECE | 9.8% | 5.0% |
| High-confidence errors | 34% | - |

## Figures Generated

All figures are saved to the `figures/` folder:
- `reliability_diagram_before.png` / `after.png` - calibration before and after
- `entropy_correct_vs_wrong_v2.png` - uncertainty for correct vs wrong predictions
- `entropy_borderline_vs_nonborderline_v2.png` - uncertainty for borderline songs
- `high_confidence_errors_class_dist_v2.png` - where high-confidence errors occur

## Notes

- I used a fixed test set of 271 samples throughout
- Temperature T=1.60 was chosen by grid search on validation data
- Statistical tests (Mann-Whitney U) confirmed significance (p < 0.001)

## Contact

If you have questions about the code, feel free to ask.
