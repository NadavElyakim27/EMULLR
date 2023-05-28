import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from moviepy.video.io.bindings import mplfig_to_npimage
from sklearn.linear_model import LinearRegression


def inner_LinearRegression(
    X_train,
    y_train,
):
    """
    Linear regression calculation
    Args:
        X_train: array of x train data
        y_train: array of y train data
    Returns:
        array of y predictions
    """
    regressor = LinearRegression()
    regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    return regressor.predict(X_train.reshape(-1, 1))


def step_plot(
    channels: np.array,
    multi_channels_LLR: np.array,
    all_llr: np.array,
    k: int,
    treshold: float,
) -> mplfig_to_npimage:
    """
    Plot algorithm step - channels, their log likelihood ratio, the treshold and ect.
    Args:
        channels: array of the channels, shape - (number of channel, time bins).
        multi_channels_LLR: array of all log likelihood ratio of the channels in each time bin.
        all_llr: log likelihood ratio in every time bin of each channel, shape - (number of channel, time bins).
        k: The index of the maximum log likelihood ratio which found.
        treshold: threshold value.
    Returns:
        fig: npimage of the fig to plot latter
    """

    # Initializetion
    channels_num, time_bins = channels.shape
    color = [
        "aquamarine",
        "limegreen",
        "brown",
        "salmon",
        "deepskyblue",
        "lightpink",
        "orange",
        "gold",
        "plum",
        "cyan",
        "aqua",
        "gray",
    ]

    fig = plt.figure(
        constrained_layout=True,
        figsize=(12, 6),
    )
    gs = GridSpec(
        channels_num,
        4,
        figure=fig,
    )

    # Each channel plot with his linear regression line:
    for ch, i in zip(channels, range(channels_num)):
        ax = fig.add_subplot(gs[i, 0:2])
        ax.scatter(np.arange(time_bins), ch, color=color[i])
        if k != 0:
            ax.plot(
                np.arange(time_bins)[:k],
                inner_LinearRegression(np.arange(time_bins)[:k], ch[:k]),
                color="midnightblue",
                alpha=0.8,
                linestyle="-",
            )
            ax.plot(
                np.arange(time_bins)[k:],
                inner_LinearRegression(np.arange(time_bins)[k:], ch[k:]),
                color="midnightblue",
                alpha=0.8,
                linestyle="-",
            )
        ax.set_ylabel(f"Ch {i+1}", fontsize=10)

    axbig = fig.add_subplot(gs[0:, 2:])

    # Each chennel log likelihood ratio plot:
    for i, llr in zip(range(channels_num), all_llr):
        axbig.plot(np.arange(time_bins), llr, color=color[i], label=f"ch {i+1}")

    # Multichennels log likelihood ratio plot:
    axbig.plot(
        np.arange(time_bins),
        multi_channels_LLR,
        color="midnightblue",
        label="multichannel",
    )

    # Add treshold line:
    if np.sum(multi_channels_LLR) == 0:
        axbig.text(
            x=0.01,
            y=0.99,
            transform=axbig.transAxes,
            ha="left",
            va="top",
            s="segments are too small",
            color="red",
            fontsize=12,
        )
    elif treshold < np.max(multi_channels_LLR):
        axbig.axhline(y=treshold, color="green", linestyle="-", alpha=0.5)
        axbig.text(x=0, y=treshold + 0.1, s="Treshold", color="green", fontsize=12, va="bottom", ha="left")
    elif treshold >= np.max(multi_channels_LLR):
        axbig.axhline(y=treshold, color="red", linestyle="-", alpha=0.5)
        axbig.text(x=0, y=treshold + 0.1, s="Treshold", color="red", fontsize=12, va="bottom", ha="left")

    # Add change point line:
    if k != 0:
        axbig.axvline(x=k, color="k", linestyle="dashed")

    # More designs:
    axbig.legend(loc=1)
    for ax in fig.axes:
        ax.tick_params(labelbottom=False, labelleft=False)
    fig.axes[0].set_title("Channels", fontsize=18, fontweight="bold")
    fig.axes[-1].set_title("Log likelihood ratio", fontsize=18, fontweight="bold")
    fig.axes[-1].tick_params(labelbottom=True, labelleft=False, labelright=True)
    fig.axes[-1].set_ylabel("LLR", fontsize=14)
    fig.axes[-1].yaxis.set_label_position("right")
    fig.axes[-2].tick_params(labelbottom=True, labelleft=False)
    fig.axes[-2].set_xlabel("Time bins", fontsize=14)
    fig.axes[-1].set_xlabel("Time bins", fontsize=14)

    plt.show()
    return mplfig_to_npimage(fig)


def plot_experiment_results(memory_result, memory_std, list_parameter_changed = None, x_label = None, param_text = ""): 

#   plt.figure(figsize=(12,4))

  for mu, std, lb in zip(memory_result, memory_std, ["precision","recall","F1"]):
    if list_parameter_changed:
      ci = 0.1 * (std/mu)
      plt.fill_between(list_parameter_changed, (mu-ci), (mu+ci), alpha=0.1)
      plt.plot(list_parameter_changed, mu, label=lb) 
  plt.ylim((0.0,1))
  plt.xlabel(x_label, fontsize = 12)                 
  plt.xticks(list_parameter_changed)  
  plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))            
  plt.ylabel('score', fontsize = 14)
  plt.grid()
  plt.legend(loc='upper left', bbox_to_anchor=(0.13, 1.13), ncol=3, fancybox=True, shadow=True)
  props = dict(boxstyle='round', facecolor='white', alpha=0.5)
  plt.text(list_parameter_changed[-3], 0, param_text, multialignment = 'left', verticalalignment = 'bottom', fontsize=10,  bbox=props) 
