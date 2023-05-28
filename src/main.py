import numpy as np
import pandas as pd

from src.model import ChangePointDetection

# Change according to the location of the data
PATH_TO_DATA = "data/"
PATH_TO_SAVE_PLOTS = "plots/"


def import_data(path_to_data_set):

    df = pd.read_csv(path_to_data_set, index_col=0)
    channels = np.zeros((df.shape[1] - 1, df.shape[0]))
    for i in range(1, df.shape[1]):
        channels[i - 1] = df.iloc[:, i]

    return channels

def run_VIA_datasets(method = "EMULLR"):
    for i in range(1, 7):

        print("**************************")
        print(f"VIA Data set {i}")
        print("**************************")

        data_set_name = f"data_set_{i}"
        path_to_data_set = f"{PATH_TO_DATA}{data_set_name}.csv"
        channels = import_data(path_to_data_set)
        model_via = ChangePointDetection(channels, method = method)
        change_points = model_via.fit()        
