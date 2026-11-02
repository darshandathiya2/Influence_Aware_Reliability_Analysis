import numpy as np
import matplotlib.pyplot as plt


def jackknife_analysis(dice_scores):
    """
    Perform jackknife analysis on per-sample Dice scores.

    Parameters
    ----------
    dice_scores : array-like
        Per-sample Dice scores.

    Returns
    -------
    dict
        Full-sample mean, jackknife estimate, bias, standard error,
        95% confidence interval, CI width, leave-one-out means,
        influence scores, and ranked influential sample indices.
    """

    # Convert input to a NumPy array and remove invalid values
    dice_scores = np.asarray(dice_scores, dtype=float).flatten()
    dice_scores = dice_scores[np.isfinite(dice_scores)]

    n = len(dice_scores)

    if n < 2:
        raise ValueError("At least two valid Dice scores are required.")

    # Step 1: Full-sample mean
    full_mean = np.mean(dice_scores)

    # Step 2: Leave-one-out means
    jackknife_means = np.array([
        np.mean(np.delete(dice_scores, i))
        for i in range(n)
    ])

    # Step 3: Jackknife estimate
    jackknife_mean = np.mean(jackknife_means)

    # Step 4: Jackknife bias estimate
    jackknife_bias = (n - 1) * (jackknife_mean - full_mean)

    # Step 5: Jackknife standard error
    jackknife_se = np.sqrt(
        ((n - 1) / n) *
        np.sum((jackknife_means - jackknife_mean) ** 2)
    )

    # Step 6: 95% Confidence interval (normal approximation)
    ci_lower = jackknife_mean - 1.96 * jackknife_se
    ci_upper = jackknife_mean + 1.96 * jackknife_se

    # Step 7: Confidence interval width
    ci_width = ci_upper - ci_lower

    # Step 8: Absolute influence scores
    influence_scores = np.abs(full_mean - jackknife_means)

    # Step 9: Rank samples by influence (highest first)
    top_influential_indices = np.argsort(influence_scores)[::-1]

    return {
        "n": n,
        "full_mean": full_mean,
        "jackknife_mean": jackknife_mean,
        "jackknife_bias": jackknife_bias,
        "jackknife_se": jackknife_se,
        "ci": (ci_lower, ci_upper),
        "ci_width": ci_width,
        "jackknife_means": jackknife_means,
        "influence_scores": influence_scores,
        "top_influential_indices": top_influential_indices
    }


def plot_jackknife_convergence(
    dice_scores,
    max_sample_size=None,
    step=5,
    runs=300,
    seed=42,
    save_path=None
):
    """
    Evaluate the empirical convergence of the jackknife estimator
    using per-sample segmentation Dice scores.
    """

    # Prepare Dice scores
    dice_scores = np.asarray(dice_scores, dtype=float).flatten()
    dice_scores = dice_scores[np.isfinite(dice_scores)]

    n_cases = len(dice_scores)

    if n_cases < 2:
        raise ValueError("At least two valid Dice scores are required.")

    if max_sample_size is None:
        max_sample_size = n_cases

    max_sample_size = min(max_sample_size, n_cases)

    if step < 1 or runs < 1:
        raise ValueError("step and runs must be positive.")

    print("Number of valid cases:", n_cases)

    # Full-sample reference estimate
    true_theta = np.sqrt(np.mean(dice_scores))

    # Sampling configuration
    rng = np.random.default_rng(seed)
    sample_sizes = np.arange(2, max_sample_size + 1, step)

    jackknife_mse = []

    for n in sample_sizes:

        squared_errors = []

        for _ in range(runs):

            # Bootstrap sampling
            x = rng.choice(dice_scores, size=n, replace=True)

            # Ordinary estimator
            theta = np.sqrt(np.mean(x))

            # Leave-one-out estimates
            loo_estimates = np.array([
                np.sqrt(np.mean(np.delete(x, i)))
                for i in range(n)
            ])

            # Bias-corrected jackknife estimator
            theta_jack = (
                n * theta - (n - 1) * np.mean(loo_estimates)
            )

            # Squared estimation error
            squared_errors.append((theta_jack - true_theta) ** 2)

        jackknife_mse.append(np.mean(squared_errors))

    jackknife_mse = np.asarray(jackknife_mse)

    # Plot
    plt.figure(figsize=(7, 5))

    plt.plot(
        sample_sizes,
        jackknife_mse,
        linewidth=2,
        label="Jackknife MSE"
    )

    plt.xlabel("Sample Size (n)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return sample_sizes, jackknife_mse
