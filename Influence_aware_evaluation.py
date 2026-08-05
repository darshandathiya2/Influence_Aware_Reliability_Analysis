import numpy as np

def jackknife_analysis(dice_scores):
    dice_scores = np.array(dice_scores)
    n = len(dice_scores)

    # Step 1: Full mean
    full_mean = np.mean(dice_scores)

    # Step 2: Leave-one-out means
    jackknife_means = np.array([
        np.mean(np.delete(dice_scores, i)) for i in range(n)
    ])

    # Step 3: Jackknife estimate of mean
    jk_mean = np.mean(jackknife_means)

    # jackknife bias estimate
    bias_jackknife = (n - 1) * (jk_mean - full_mean)

    # Step 4: Jackknife standard error
    jk_se = np.sqrt((n - 1) / n * np.sum((jackknife_means - jk_mean) ** 2))

    # Step 5: 95% Confidence Interval
    ci_lower = jk_mean - 1.96 * jk_se
    ci_upper = jk_mean + 1.96 * jk_se

    # Step 6: CI width
    ci_width = ci_upper - ci_lower

    # Step 7: Influence scores (IMPORTANT)
    influence_scores = np.abs(full_mean - jackknife_means)

    # Step 8: Top influential samples
    top_indices = np.argsort(influence_scores)[::-1]

    return {
        "full_mean": full_mean,
        "jackknife_mean": jk_mean,
        "jackknife_bias": bias_jackknife,
        "jackknife_se": jk_se,
        "ci": (ci_lower, ci_upper),
        "ci_width": ci_width,
        "jackknife_means": jackknife_means,
        "influence_scores": influence_scores,
        "top_influential_indices": top_indices
    }
