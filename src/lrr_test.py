import numpy as np
from scipy.optimize import root
from scipy.special import gamma


# Step by step - functions to calculate log likelihood ratia
def calculate_slope(
    x,
    y,
    x_mean,
    y_mean,
):
    return (
        (np.sum((y - y_mean) * (x - x_mean), axis=1)) / (np.sum((x - x_mean) ** 2))
    ).reshape(-1, 1)


def calculate_intercept(
    x,
    y,
    x_mean,
    y_mean,
    b,
):
    return y_mean - (b * x_mean)


def calculate_variance(
    x,
    y,
    b,
    b_0,
    n,
):
    return ((np.sum((y - b_0 - b * x) ** 2, axis=1)) / n).reshape(-1, 1)


def calculate_loglikelihood(
    x,
    y,
    b,
    b_0,
    n,
    var,
):
    return (n / 2) * np.log(var) + (1 / (2 * var)) * np.sum(
        (y - b_0 - b * x) ** 2, axis=1, keepdims=True
    )


def calculate_fully_log_likelihood(
    x: np.array,
    y: np.array,
    n: int,
) -> np.array:
    """
    Function which calculate the log likelihood of linearly distributed channels (vectorized calculate).
    Args:
        x, y: the data.
    Return:
       array of log likelihoods.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y, axis=1, keepdims=True)

    b = calculate_slope(x, y, x_mean, y_mean)
    b_0 = calculate_intercept(x, y, x_mean, y_mean, b)
    var = calculate_variance(x, y, b, b_0, n)

    return calculate_loglikelihood(x, y, b, b_0, n, var)


def calculate_log_likelihood_ratio(
    channels: np.array,
    edges: int = 10,
    method: str = "EMULLR",
) -> list:
    """
    Function which calculate the log likelihood ratio (llr) of linearly distributed channels (vectorized calculate).
    Args:
        channels: array of the channels, shape - (number of channel, time bins).
        edges: time steps to ignore from in the beginning and the end of the channels (To avoid noise from too close change points).
    Return:
        all_llr: log likelihood ratio in every time bin of each channel, shape - (number of channel, time bins).
    """

    # Initialize variables:
    channels_num, time_bins = channels.shape
    all_llr = np.zeros((channels_num, time_bins))
    x = np.arange(1, time_bins + 1)
    y = channels

    if method == "EMULLR":
        # Null hypothesis calculation (H_0)
        x_mean = np.mean(x)
        y_mean = np.mean(y, axis=1, keepdims=True)
        b = calculate_slope(x, y, x_mean, y_mean)
        b_0 = calculate_intercept(x, y, x_mean, y_mean, b)
        var = calculate_variance(x, y, b, b_0, time_bins)
        s = calculate_loglikelihood(x, y, b, b_0, time_bins, var)

        # For each time bin calculate the alternative hypothesis (H_1)
        for k in range(edges, time_bins - edges):
            # Division into segments - before and after k (not including k)
            s_until_k = calculate_fully_log_likelihood(x=x[:k], y=y[:, :k], n=k)
            s_from_k = calculate_fully_log_likelihood(
                x=x[k:], y=y[:, k:], n=time_bins - k
            )

            # Update total array
            score = s - s_until_k - s_from_k
            all_llr[:, k] = score.T

    elif method == "MULLR":
        # Repeated calculations
        LL = lambda y, mu: np.sum((y - mu) ** 2, axis=1, keepdims=True)
        mu = np.mean(y, axis=1, keepdims=True)
        sigma = np.std(y, axis=1, keepdims=True)
        s = LL(y, mu)

        for k in range(edges, time_bins - edges):
            mu_until_k = np.mean(y[:, :k], axis=1, keepdims=True)
            mu_from_k = np.mean(y[:, k:], axis=1, keepdims=True)
            s_until_k = LL(y[:, :k], mu_until_k)
            s_from_k = LL(y[:, k:], mu_from_k)
            # calculate log likelihood ratio
            score = (s - s_until_k - s_from_k) / ((sigma**2))
            # Update  total array
            all_llr[:, k] = score.T

    return all_llr


def calculate_threshold(
    channels_num: int,
    time_bins: int,
    alpha: float = 0.01,
    method: str = "EMULLR",
) -> float:
    """
    Function which calculate threshold for algorithm of detecting change points in multi-channels time
    series data using log likelihood ratio test - Calculation according to the formulas in the article.
    Args:
        channels_num: number of channels.
        time bins: time bins.
        alpha: (1-confidence).
    Returns:
        threshold value for log likelihood ratio test.
    """

    if method == "EMULLR":
        d = 3 * channels_num  # degree_of_freedom
    elif method == "MULLR":
        d = channels_num  # degree_of_freedom
    h = (np.log(time_bins) ** 1.5) / time_bins
    T = np.log(((1 - h) ** 2) / (h**2))

    def f(x):
        return (
            (x**d) * np.exp((-(x**2)) / 2.0) / ((2.0 ** (d / 2.0)) * gamma(d / 2.0))
        ) * (T - ((d / (x**2.0)) * T) + (4 / (x**2.0))) - alpha

    if channels_num < 5:
        tries = 6
        xrootVec = np.empty(tries)
        xrootVec[:] = np.NaN

        for i in range(1, tries + 1):
            xrootVec[i - 1] = root(f, i).x

        xrootVec = xrootVec[xrootVec < 6]
        max_root = np.max(xrootVec)  # assign the largest found root
        threshold = (max_root**2) / 2

    else:
        tries = 7
        xrootVec = np.empty(tries)
        xrootVec[:] = np.NaN

        for i in range(1, tries + 1):
            xrootVec[i - 1] = root(f, i).x

        xrootVec = xrootVec[xrootVec < 10]
        max_root = np.max(xrootVec)  # assign the largest found root
        threshold = (max_root**2) / 2

    return threshold
