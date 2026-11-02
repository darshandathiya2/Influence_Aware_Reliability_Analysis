# Influence-Aware Reliability Evaluation of Medical Image Segmentation

This repository contains the implementation of the Influence-Aware Reliability Evaluation Framework proposed in the paper "Beyond Average Dice: Influence-Aware Reliability Evaluation of Medical Image Segmentation"

## Overview
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
Influence_Aware_Reliability_Analysis/
├── data/
│   └── summary_BUSI.json   #Per-sample segmentation performance metrics for the BUSI test set
├── influence_aware_evaluation.py   #Evaluates the influence of individual test samples on overall segmentation performance and quantifies the uncertainty and stability of performance estimates
├── LICENSE
├── requirements.txt
└── README.md
```

## Requirements

Install the required dependencies:

```
pip install numpy matplotlib json
```

## Usage

An example implementation of the proposed framework is provided in the Jupyter Notebook:

```bash
Influence_Aware_Evaluation_Example.ipynb
```
Run the notebook cells sequentially to reproduce the results and visualizations.
