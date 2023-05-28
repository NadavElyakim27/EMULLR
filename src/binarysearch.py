import numpy as np

from src.lrr_test import calculate_log_likelihood_ratio, calculate_threshold
from src.plot import step_plot


def BS(
    model,
    channels: np.array,
    alpha: float = 0.01,
    edges: int = 10,
    source: str = None,
    to_plot: bool = True,
) -> list:
    """
    A recursive binary search algorithm for detecting change points in multi-channels time series data using log likelihood ratio test.
    Args:
        model: the model
        channels: array of the channels, shape - (number of channel, time bins).
        alpha: (1-confidence).
        edges: time steps to ignore from in the beginning and the end of the channels (To avoid noise from too close change points).
        source: the parent of the current node with additional direction to differ which successor.
    Returns:
        change_points_list: list of the change points which founded.
    """

    # Initialize variables
    change_points_list = []
    channels_num, time_bins = channels.shape
    method=model.method

    all_llr = calculate_log_likelihood_ratio(
        channels=channels,
        edges=edges,
        method=method
    )

    multi_channels_LLR = np.sum(all_llr, axis=0)
    max_llr = np.max(multi_channels_LLR)
    pred_time_bins = np.argmax(multi_channels_LLR)

    threshold = calculate_threshold(
        channels_num,
        time_bins,
        alpha=alpha,
        method=method
    )

    if to_plot:
        # if root
        if model.plots_keys == []:
            source = "Root"
        print(f"Step - {source}:")
        model.plots[source] = step_plot(
            channels,
            multi_channels_LLR,
            all_llr,
            pred_time_bins,
            threshold,
        )
        model.plots_keys.append(source)

    # Recursive step if a change point is found
    if max_llr > threshold:

        cp_list_left = BS(
            model=model,
            channels=channels[:, :pred_time_bins],
            alpha=alpha,
            edges=edges,
            source=f"{source}_L",
            to_plot=to_plot,
        )
        cp_list_right = BS(
            model=model,
            channels=channels[:, pred_time_bins + 1 :],
            alpha=alpha,
            edges=edges,
            source=f"{source}_R",
            to_plot=to_plot,
        )

        if cp_list_right != []:
            cp_list_right = cp_list_right + pred_time_bins

        change_points_list.extend(cp_list_left)
        change_points_list.extend([pred_time_bins])
        change_points_list.extend(cp_list_right)

    if source == "Root" and to_plot:
        print(
            f"""**************************
All predicted change points: {change_points_list})
**************************
______________________________________________________________________

"""
        )

    return change_points_list
