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
    max_sample_size=500,
    step=5,
    runs=300,
    seed=42,
    save_path=None
):
    """
    Evaluate the empirical convergence of the jackknife estimator
    using bootstrap samples of per-case Dice scores.
    """

    # Prepare Dice scores
    dice_scores = np.asarray(dice_scores, dtype=float).flatten()
    dice_scores = dice_scores[np.isfinite(dice_scores)]

    if len(dice_scores) < 2:
        raise ValueError("At least two valid Dice scores are required.")

    print("Number of available cases:", len(dice_scores))

    # Full-dataset reference estimate
    true_theta = np.sqrt(np.mean(dice_scores))

    # Bootstrap sampling configuration
    rng = np.random.default_rng(seed)
    sample_sizes = np.arange(5, max_sample_size + 1, step)

    jackknife_mse = []

    for n in sample_sizes:

        jack_errors = []

        for _ in range(runs):

            # Bootstrap sampling with replacement
            x = rng.choice(dice_scores, size=n, replace=True)

            # Ordinary estimator
            theta = np.sqrt(np.mean(x))

            # Leave-one-out estimates
            jackknife_estimates = np.array([
                np.sqrt(np.mean(np.delete(x, i)))
                for i in range(n)
            ])

            # Bias-corrected jackknife estimator
            theta_jack = (
                n * theta - (n - 1) * np.mean(jackknife_estimates)
            )

            # Squared estimation error
            jack_errors.append((theta_jack - true_theta) ** 2)

        jackknife_mse.append(np.mean(jack_errors))

    jackknife_mse = np.asarray(jackknife_mse)

    # Plot convergence curve
    plt.figure(figsize=(7, 5))

    plt.plot(
        sample_sizes,
        jackknife_mse,
        linewidth=2,
        label="Jackknife Error"
    )

    plt.xlabel("Sample Size (n)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return sample_sizes, jackknife_mse


def plot_loo_sensitivity(
    dice_scores,
    save_path=None
):
    """
    Perform leave-one-out sensitivity analysis using per-sample Dice scores.
    """

    # Prepare Dice scores
    dice_scores = np.asarray(dice_scores, dtype=float).flatten()
    dice_scores = dice_scores[np.isfinite(dice_scores)]

    n = len(dice_scores)

    if n < 2:
        raise ValueError("At least two valid Dice scores are required.")

    # Original mean Dice
    theta = np.mean(dice_scores)

    # Leave-one-out means
    theta_loo = np.array([
        np.mean(np.delete(dice_scores, i))
        for i in range(n)
    ])

    # LOO sensitivity scores
    sensitivity = theta_loo - theta

    # Assign colors
    colors = ["red" if s > 0 else "blue" for s in sensitivity]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(n), sensitivity, color=colors)
    plt.axhline(0, linestyle="--")

    plt.xlabel("Sample Index")
    plt.ylabel("Δ Dice")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {
        "original_mean": theta,
        "jackknife_means": theta_loo,
        "sensitivity_scores": sensitivity
    }


def plot_k_sample_sensitivity(
    dice_scores,
    max_k=5,
    save_path=None
):
    """
    Perform influence-based k-sample sensitivity analysis and
    plot the change in mean Dice after removing influential samples.

    Parameters
    ----------
    dice_scores : array-like
        Per-sample Dice scores.
    max_k : int
        Maximum number of influential samples to remove.
    save_path : str or None
        Optional path to save the stability plot.

    Returns
    -------
    dict
        Full mean, influence scores, ranked sample indices,
        k values, and mean Dice after sample removal.
    """

    # Prepare Dice scores
    dice_scores = np.asarray(dice_scores, dtype=float).flatten()
    valid_mask = np.isfinite(dice_scores)
    dice_scores = dice_scores[valid_mask]

    n = len(dice_scores)

    if n < 2:
        raise ValueError("At least two valid Dice scores are required.")

    if max_k < 1 or max_k >= n:
        raise ValueError("max_k must be between 1 and n-1.")

    # Step 1: Compute full-sample mean
    full_mean = np.mean(dice_scores)

    # Step 2: Compute leave-one-out means
    jackknife_means = np.array([
        np.mean(np.delete(dice_scores, i))
        for i in range(n)
    ])

    # Step 3: Compute absolute influence scores
    influence_scores = np.abs(full_mean - jackknife_means)

    # Step 4: Rank samples by influence
    sorted_indices = np.argsort(influence_scores)[::-1]

    # Step 5: Progressive top-k sample removal
    k_values = np.arange(1, max_k + 1)
    stability_results = []

    for k in k_values:

        remove_indices = sorted_indices[:k]
        remaining_scores = np.delete(dice_scores, remove_indices)

        new_mean = np.mean(remaining_scores)
        stability_results.append(new_mean)

        print(
            f"k={k} | "
            f"Removed indices: {remove_indices} | "
            f"New Mean Dice: {new_mean:.4f}"
        )

    stability_results = np.asarray(stability_results)

    # Step 6: Plot stability curve
    k_plot = np.concatenate(([0], k_values))
    dice_plot = np.concatenate(([full_mean], stability_results))

    plt.figure(figsize=(7, 5))
    plt.plot(k_plot, dice_plot, marker="o", linewidth=2)

    plt.xlabel("Number of Removed Samples (k)")
    plt.ylabel("Mean Dice Score")
    plt.title("Influence-Based k-Sample Sensitivity Analysis")
    plt.xticks(k_plot)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {
        "full_mean": full_mean,
        "influence_scores": influence_scores,
        "ranked_indices": sorted_indices,
        "k_values": k_values,
        "stability_results": stability_results
    }
