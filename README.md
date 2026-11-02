# Influence-Aware Reliability Evaluation of Medical Image Segmentation
Influence-Aware Reliability Analysis provides a statistical framework for examining the stability of medical image segmentation results at the individual-case level. Rather than relying exclusively on aggregate performance metrics (Dice Coefficient), it investigates how variations across test samples affect the estimated mean Dice coefficient. The framework combines jackknife estimation, bootstrap comparison, and leave-one-out analysis to characterize uncertainty and identify samples with a substantial impact on overall performance.

## Key Features
- **Jackknife Resampling**: Estimates bias, standard error, and 95% confidence intervals of the mean Dice coefficient.
- **Consistency Analysis**: Examines the empirical convergence behavior of the jackknife estimator using mean squared error (MSE).
- **Jackknife vs. Bootstrap**: Compares global performance and uncertainty estimates obtained using both methods.
- **Leave-One-Out Sensitivity Analysis**: Quantifies the influence of individual test samples on the mean Dice coefficient.
- **High-Influence Sample Analysis**: Investigates the sensitivity of reported performance through progressive removal of influential samples.


## Experimental Setup
The proposed framework is evaluated using **nnU-Net** segmentation results from two medical imaging datasets:
- **BUSI**: Breast ultrasound tumor segmentation.
- **BraTS2020**: Brain tumor segmentation (Whole Tumor region).

The reliability analysis is performed on **case-wise Dice coefficients** obtained from held-out test sets.

## Repository Structure

```text
repository-name/
├── folder1/
│   ├── file1.py
│   └── file2.py
├── folder2/
│   └── results/
├── requirements.txt
└── README.md
