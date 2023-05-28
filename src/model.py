import cv2
import numpy as np

from src.binarysearch import *
from src.plot import *


class ChangePointDetection:
    # Multi-channels change points detection

    def __init__(
        self,
        channels: np.array,
        alpha: float = 0.01,
        edges: int = 10,
        method: str = "EMULLR",
    ):
        """
        Initialization.
        Args:
            channels: array of the channels, size - (number of channel, time bins).
            alpha: (1-confidence).
            edge: time steps to ignore from in the beginning and the end of the channels
        """

        self.channels = channels
        self.alpha = alpha
        self.edge = edges
        self.method = method
        self.change_points = []
        self.plots = {}
        self.plots_keys = []

    def fit(self, to_plot: bool = True) -> list:
        """
        Using binary search and log likelihood ratio test to find change point
        at the multi-channels input.
        Returns:
            change_points: List, time bins of change points
        """

        self.change_points = BS(
            model=self,
            channels=self.channels,
            alpha=self.alpha,
            edges=self.edge,
            to_plot=to_plot,
        )
        return self.change_points

    def save_plot(
        self,
        path: str = "",
        data_set_name="",
    ):
        """
        Save the algorithm steps plots (if any)
        Args:
            path: path to save the plots
            data_set_name: the name of the dateset which the plots releted
        """

        if self.plots == []:
            print("fit not done yet")
        else:
            for key, fig in self.plots.items():
                path_to_save = f"{path}{data_set_name}_{key}.jpg"
                cv2.imwrite(path_to_save, fig[:, :, (2, 1, 0)])
