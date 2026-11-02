# Influence-Aware Reliability Evaluation of Medical Image Segmentation
This repository contains the implementation of an influence-aware reliability evaluation framework for medical image segmentation. The framework uses jackknife resampling and leave-one-out sensitivity analysis to assess the statistical reliability of segmentation performance and quantify the influence of individual test samples on the reported mean Dice coefficient.

## Overview

Deep learning-based medical image segmentation models are commonly evaluated using average performance metrics, such as the Dice coefficient. However, a single average value may conceal substantial variability across individual test samples.

This work introduces a jackknife-based evaluation approach that provides:

- Estimation of bias and standard error of the mean Dice coefficient.

- Approximate 95% confidence intervals for segmentation performance.

- Leave-one-out sensitivity analysis to identify influential test samples.

- Empirical assessment of estimator convergence with increasing sample size.

- Comparison of jackknife and bootstrap uncertainty estimates.

- Progressive removal analysis to examine the influence of high-impact test samples.

The framework provides both global uncertainty estimates and sample-level insights into segmentation performance.


