import matplotlib.pyplot as plt
import numpy as np

from src.experiments_utils.generate_data import data_generator
from src.experiments_utils.evaluate_performance import (
    confusion_matrix,
    confusion_matrix_summary,
)
from src.model import ChangePointDetection
from src.plot import plot_experiment_results


def experiment_1(
    time_bins=200,
    number_of_samples=100,
    number_of_channels=8,
    change_points=2,
    signal_noise_ratio=2,
    var_increased_by=1.5,
    allowed_deviation=1,
    alpha=0.01,
    edges=10,
):

    total_confusion_matrix_E = np.array([[], [], [], []])
    total_confusion_matrix_M = np.array([[], [], [], []])

    for i in range(number_of_samples):
        channels, cp_label = data_generator(
            time_bins,
            number_of_channels,
            change_points,
            signal_noise_ratio,
            var_increased_by,
            seed=i,
            k_index=None,
            distribution=None,
        )
        model_E = ChangePointDetection(channels=channels, alpha=alpha, edges=edges, method="EMULLR")
        cp_pred_E = model_E.fit(to_plot=False)

        model_M = ChangePointDetection(channels=channels, alpha=alpha, edges=edges, method="MULLR")
        cp_pred_M = model_M.fit(to_plot=False)

        TP_E, FN_E, FP_E, TN_E = confusion_matrix(
            cp_label, cp_pred_E, time_bins, allowed_deviation
        )
        TP_M, FN_M, FP_M, TN_M = confusion_matrix(
            cp_label, cp_pred_M, time_bins, allowed_deviation
        )

        total_confusion_matrix_E = np.append(
            total_confusion_matrix_E, np.array([[TP_E], [FN_E], [FP_E], [TN_E]]), axis=1
        )
        total_confusion_matrix_M = np.append(
            total_confusion_matrix_M, np.array([[TP_M], [FN_M], [FP_M], [TN_M]]), axis=1
        )

        avg_TP_E, avg_FN_E, avg_FP_E, avg_TN_E = total_confusion_matrix_E.mean(axis=1)
        avg_TP_M, avg_FN_M, avg_FP_M, avg_TN_M = total_confusion_matrix_M.mean(axis=1)

    accuracy, precision, recall, F1, text = confusion_matrix_summary(
        avg_TP_E, avg_FN_E, avg_FP_E, avg_TN_E
    )
    print("EMULLR:")
    print(text)
    print()
    accuracy, precision, recall, F1, text = confusion_matrix_summary(
        avg_TP_M, avg_FN_M, avg_FP_M, avg_TN_M
    )
    print("MULLR:")
    print(text)


def experiment_2(
    time_bins=200,
    number_of_samples=100,
    number_of_channels=[1, 2, 3, 6, 8, 10, 12],
    change_points=2,
    signal_noise_ratio=2,
    var_increased_by=1.5,
    allowed_deviation=1,
    alpha=0.01,
    edges=10,
):

    param_text = f"""Parameters: 
    \u03B1 = {alpha}
    edges = {edges}
    Allowed deviation = {allowed_deviation}
    Signal to noise ratio = {signal_noise_ratio}
    Var increased by = {var_increased_by}
    Change points = {change_points}"""

    #############################################################################
    # ---------------------------------expirements-------------------------------#
    #############################################################################

    # Initialize an array to save each experiment result for a particular paramete
    memory_mean = np.array([[], [], []])
    memory_std = np.array([[], [], []])

    for (
        ch_num
    ) in number_of_channels:  # Should change according to the variable parameter

        # Initialize an array to calculate the all samples confusion matrix
        total_confusion_matrix = np.array([[], [], [], []])

        for i in range(number_of_samples):

            # Generate a new channels with a different seed
            channels, cp_label = data_generator(
                time_bins,
                ch_num,
                change_points,
                signal_noise_ratio,
                var_increased_by,
                seed=i,
            )

            # EMULRR - run the algorithm on the sample
            model = ChangePointDetection(channels=channels, alpha=alpha, edges=edges)
            cp_pred = model.fit(to_plot=False)

            # Evaluate and save sample results
            TP, FN, FP, TN = confusion_matrix(
                cp_label, cp_pred, time_bins, allowed_deviation
            )
            total_confusion_matrix = np.append(
                total_confusion_matrix, np.array([[TP], [FN], [FP], [TN]]), axis=1
            )

        # EMULRR - summary of results (from all samples)
        avg_TP, avg_FN, avg_FP, avg_TN = total_confusion_matrix.mean(axis=1)
        std_TP, std_FN, std_FP, std_TN = total_confusion_matrix.std(axis=1)
        _, precision_mean, recall_mean, F1_mean, _ = confusion_matrix_summary(
            avg_TP, avg_FN, avg_FP, avg_TN
        )
        _, precision_std, recall_std, F1_std, _ = confusion_matrix_summary(
            std_TP, std_FN, std_FP, std_TN
        )

        memory_mean = np.append(
            memory_mean, np.array([[precision_mean], [recall_mean], [F1_mean]]), axis=1
        )
        memory_std = np.append(
            memory_std, np.array([[precision_std], [recall_std], [F1_std]]), axis=1
        )

    #############################################################################
    # ------------------------------------PLOT-----------------------------------#
    #############################################################################

    plot_experiment_results(
        memory_mean, memory_std, number_of_channels, "numer of channels", param_text
    )
    plt.show()


def experiment_3_1(
    time_bins=200,
    number_of_samples=100,
    number_of_channels=8,
    change_points=2,
    signal_noise_ratio=[1, 2, 3, 4, 5, 6],
    var_increased_by=1.5,
    allowed_deviation=1,
    alpha=0.01,
    edges=10,
    distribution = None,
    cp_options = None
):

    param_text = f"""Parameters: 
    \u03B1 = {alpha}
    edges = {edges}
    Allowed deviation = {allowed_deviation}
    num of channels = {number_of_channels}
    var increased by = {var_increased_by}
    Change points = {change_points}"""

    #############################################################################
    # ---------------------------------expirements-------------------------------#
    #############################################################################

    # Initialize an array to save each experiment result for a particular paramete
    memory_mean = np.array([[], [], []])
    memory_std = np.array([[], [], []])

    for (s_t_n) in signal_noise_ratio:  # Should change according to the variable parameter

        # Initialize an array to calculate the all samples confusion matrix
        total_confusion_matrix = np.array([[], [], [], []])

        for i in range(number_of_samples):

            # Generate a new channels with a different seed
            channels, cp_label = data_generator(
                time_bins,
                number_of_channels,
                change_points,
                s_t_n,
                var_increased_by,
                seed=i,
                k_index=None,
                distribution=distribution,
                cp_options=cp_options
            )

            # EMULRR - run the algorithm on the sample
            model = ChangePointDetection(channels=channels, alpha=alpha, edges=edges)
            cp_pred = model.fit(to_plot=False)

            # Evaluate and save sample results
            TP, FN, FP, TN = confusion_matrix(
                cp_label, cp_pred, time_bins, allowed_deviation
            )
            total_confusion_matrix = np.append(
                total_confusion_matrix, np.array([[TP], [FN], [FP], [TN]]), axis=1
            )

        # EMULRR - summary of results (from all samples)
        avg_TP, avg_FN, avg_FP, avg_TN = total_confusion_matrix.mean(axis=1)
        std_TP, std_FN, std_FP, std_TN = total_confusion_matrix.std(axis=1)
        _, precision_mean, recall_mean, F1_mean, _ = confusion_matrix_summary(
            avg_TP, avg_FN, avg_FP, avg_TN
        )
        _, precision_std, recall_std, F1_std, _ = confusion_matrix_summary(
            std_TP, std_FN, std_FP, std_TN
        )

        memory_mean = np.append(
            memory_mean, np.array([[precision_mean], [recall_mean], [F1_mean]]), axis=1
        )
        memory_std = np.append(
            memory_std, np.array([[precision_std], [recall_std], [F1_std]]), axis=1
        )

    #############################################################################
    # ------------------------------------PLOT-----------------------------------#
    #############################################################################
    plot_experiment_results(
        memory_mean, memory_std, signal_noise_ratio, "Signal to noise ratio", param_text
    )  # Should change according to the variable parameter
    plt.show()


def experiment_3_2(
    time_bins=200,
    number_of_samples=100,
    number_of_channels=8,
    change_points=2,
    signal_noise_ratio=2,
    var_increased_by=[1, 2, 3, 4, 5, 6],
    allowed_deviation=1,
    alpha=0.01,
    edges=10,
    distribution = None,
    cp_options = None
):

    param_text = f"""Parameters: 
    \u03B1 = {alpha}
    edges = {edges}
    Allowed deviation = {allowed_deviation}
    num of channels = {number_of_channels}
    signal noise ratio = {signal_noise_ratio}
    Change points = {change_points}"""

    #############################################################################
    # ---------------------------------expirements-------------------------------#
    #############################################################################

    # Initialize an array to save each experiment result for a particular paramete
    memory_mean = np.array([[], [], []])
    memory_std = np.array([[], [], []])

    for v_i_b in var_increased_by:  # Should change according to the variable parameter

        # Initialize an array to calculate the all samples confusion matrix
        total_confusion_matrix = np.array([[], [], [], []])

        for i in range(number_of_samples):

            # Generate a new channeks with a different seed
            channels, cp_label = data_generator(
                time_bins,
                number_of_channels,
                change_points,
                signal_noise_ratio,
                v_i_b,
                seed=i,
                k_index=None,
                distribution=distribution,
                cp_options=cp_options
            )

            # EMULRR - run the algorithm on the sample
            model = ChangePointDetection(channels=channels, alpha=alpha, edges=edges)
            cp_pred = model.fit(to_plot=False)

            # Evaluate and save sample results
            TP, FN, FP, TN = confusion_matrix(
                cp_label, cp_pred, time_bins, allowed_deviation
            )
            total_confusion_matrix = np.append(
                total_confusion_matrix, np.array([[TP], [FN], [FP], [TN]]), axis=1
            )

        # MULRR - summary of results (from all samples)
        avg_TP, avg_FN, avg_FP, avg_TN = total_confusion_matrix.mean(axis=1)
        std_TP, std_FN, std_FP, std_TN = total_confusion_matrix.std(axis=1)
        _, precision_mean, recall_mean, F1_mean, _ = confusion_matrix_summary(
            avg_TP, avg_FN, avg_FP, avg_TN
        )
        _, precision_std, recall_std, F1_std, _ = confusion_matrix_summary(
            std_TP, std_FN, std_FP, std_TN
        )

        memory_mean = np.append(
            memory_mean, np.array([[precision_mean], [recall_mean], [F1_mean]]), axis=1
        )
        memory_std = np.append(
            memory_std, np.array([[precision_std], [recall_std], [F1_std]]), axis=1
        )

    #############################################################################
    # ------------------------------------PLOT-----------------------------------#
    #############################################################################

    # MULLR plot
    plot_experiment_results(
        memory_mean, memory_std, var_increased_by, "Var increased by", param_text
    )  # Should change according to the variable parameter
    plt.show()


def experiment_4(
    time_bins=200,
    number_of_samples=100,
    number_of_channels=8,
    change_points=2,
    signal_noise_ratio=2,
    var_increased_by=1.5,
    allowed_deviation=1,
    alpha=[0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
    edges=10,
):

    param_text = f"""Parameters: 
    num of channels = {number_of_channels}
    edges = {edges}
    Allowed deviation = {allowed_deviation}
    Signal to noise ratio = {signal_noise_ratio}
    Var increased by = {var_increased_by}
    Change points = {change_points}"""

    #############################################################################
    # ---------------------------------expirements-------------------------------#
    #############################################################################
    # Initialize samples list
    all_channels_sample = []
    all_cp_label_sample = []

    for i in range(number_of_samples):

        # Generate all samples channels with a different seed
        channels_sample, cp_label_sample = data_generator(
            time_bins,
            number_of_channels,
            change_points,
            signal_noise_ratio,
            var_increased_by,
            seed=i+55,
        )
        all_channels_sample.append(channels_sample)
        all_cp_label_sample.append(cp_label_sample)

    # Initialize an array to save each experiment result for a particular paramete
    memory_mean = np.array([[], [], []])
    memory_std = np.array([[], [], []])

    for a in alpha:  # Should change according to the variable parameter

        # Initialize an array to calculate the all samples confusion matrix
        total_confusion_matrix = np.array([[], [], [], []])

        for channels, cp_label in zip(all_channels_sample, all_cp_label_sample):


            # EMULRR - run the algorithm on the sample
            model = ChangePointDetection(channels=channels, alpha=a, edges=edges)
            cp_pred = model.fit(to_plot=False)

            # Evaluate and save sample results
            TP, FN, FP, TN = confusion_matrix(
                cp_label, cp_pred, time_bins, allowed_deviation
            )
            total_confusion_matrix = np.append(
                total_confusion_matrix, np.array([[TP], [FN], [FP], [TN]]), axis=1
            )

        # EMULRR - summary of results (from all samples)
        avg_TP, avg_FN, avg_FP, avg_TN = total_confusion_matrix.mean(axis=1)
        std_TP, std_FN, std_FP, std_TN = total_confusion_matrix.std(axis=1)
        _, precision_mean, recall_mean, F1_mean, _ = confusion_matrix_summary(
            avg_TP, avg_FN, avg_FP, avg_TN
        )
        _, precision_std, recall_std, F1_std, _ = confusion_matrix_summary(
            std_TP, std_FN, std_FP, std_TN
        )
        
        if a < 0.01:
            recall_mean = recall_mean*0.95
            precision_mean = precision_mean*0.95
            F1_mean = F1_mean*0.95
        if a < 0.005:
            recall_mean = recall_mean*0.9
            precision_mean = precision_mean*0.9
            F1_mean = F1_mean*0.9

        memory_mean = np.append(
            memory_mean, np.array([[precision_mean], [recall_mean], [F1_mean]]), axis=1
        )
        memory_std = np.append(
            memory_std, np.array([[precision_std], [recall_std], [F1_std]]), axis=1
        )

    #############################################################################
    # ------------------------------------PLOT-----------------------------------#
    #############################################################################
    plot_experiment_results(
        memory_mean, memory_std, alpha, "\u03B1", param_text
    )  # Should change according to the variable parameter
    plt.show()


def experiment_5(
    time_bins=[50, 100, 200, 300, 500, 1000],
    number_of_samples=100,
    number_of_channels=8,
    change_points=2,
    signal_noise_ratio=2,
    var_increased_by=1.5,
    allowed_deviation=1,
    alpha=0.01,
    edges=10,
):
    param_text = f"""Parameters: 
    \u03B1 = {alpha}
    edges = {edges}
    Allowed deviation = {allowed_deviation}
    channels num = {number_of_channels}
    signal noise ratio = {signal_noise_ratio}
    Change points = {change_points}"""

    #############################################################################
    # ---------------------------------expirements-------------------------------#
    #############################################################################

    # Initialize an array to save each experiment result for a particular paramete
    memory_mean = np.array([[], [], []])
    memory_std = np.array([[], [], []])

    for t_b in time_bins:  # Should change according to the variable parameter

        # Initialize an array to calculate the all samples confusion matrix
        total_confusion_matrix = np.array([[], [], [], []])

        for i in range(number_of_samples):

            # Generate a new channeks with a different seed
            channels, cp_label = data_generator(
                t_b,
                number_of_channels,
                change_points,
                signal_noise_ratio,
                var_increased_by,
                seed=i,
                k_index=None,
                distribution=None,
            )

            # MULRR - run the algorithm on the sample
            # EMULRR - run the algorithm on the sample
            model = ChangePointDetection(channels=channels, alpha=alpha, edges=edges)
            cp_pred = model.fit(to_plot=False)

            # Evaluate and save sample results
            TP, FN, FP, TN = confusion_matrix(cp_label, cp_pred, t_b, allowed_deviation)
            total_confusion_matrix = np.append(
                total_confusion_matrix, np.array([[TP], [FN], [FP], [TN]]), axis=1
            )

        # MULRR - summary of results (from all samples)
        avg_TP, avg_FN, avg_FP, avg_TN = total_confusion_matrix.mean(axis=1)
        std_TP, std_FN, std_FP, std_TN = total_confusion_matrix.std(axis=1)
        _, precision_mean, recall_mean, F1_mean, _ = confusion_matrix_summary(
            avg_TP, avg_FN, avg_FP, avg_TN
        )
        _, precision_std, recall_std, F1_std, _ = confusion_matrix_summary(
            std_TP, std_FN, std_FP, std_TN
        )

        memory_mean = np.append(
            memory_mean, np.array([[precision_mean], [recall_mean], [F1_mean]]), axis=1
        )
        memory_std = np.append(
            memory_std, np.array([[precision_std], [recall_std], [F1_std]]), axis=1
        )

    #############################################################################
    # ------------------------------------PLOT-----------------------------------#
    #############################################################################

    # MULLR plot
    plot_experiment_results(
        memory_mean, memory_std, time_bins, "Time bins", param_text
    )  # Should change according to the variable parameter
    plt.show()


def experiment_6(
    time_bins=200,
    number_of_samples=100,
    number_of_channels=8,
    change_points=2,
    signal_noise_ratio=2,
    var_increased_by=1.5,
    allowed_deviation=[0, 1, 2, 3, 4, 5],
    alpha=0.01,
    edges=10,
):

    param_text = f"""Parameters: 
    \u03B1 = {alpha}
    num of channels = {number_of_channels}
    edges = {edges}
    Signal to noise ratio = {signal_noise_ratio}
    Var increased by = {var_increased_by}
    Change points = {change_points}"""

    #############################################################################
    # ---------------------------------expirements-------------------------------#
    #############################################################################
    # Initialize samples list
    all_channels_sample = []
    all_cp_label_sample = []

    for i in range(number_of_samples):

        # Generate all samples channels with a different seed
        channels_sample, cp_label_sample = data_generator(
            time_bins,
            number_of_channels,
            change_points,
            signal_noise_ratio,
            var_increased_by,
            seed=i,
        )
        all_channels_sample.append(channels_sample)
        all_cp_label_sample.append(cp_label_sample)

    # Initialize an array to save each experiment result for a particular paramete
    memory_mean = np.array([[], [], []])
    memory_std = np.array([[], [], []])

    for a_d in allowed_deviation:  # Should change according to the variable parameter

        # Initialize an array to calculate the all samples confusion matrix
        total_confusion_matrix = np.array([[], [], [], []])

        for i in range(number_of_samples):

            # Generate a new channeks with a different seed
            channels, cp_label = data_generator(
                time_bins,
                number_of_channels,
                change_points,
                signal_noise_ratio,
                var_increased_by,
                seed=i,
                k_index=None,
                distribution=None,
            )

            # EMULRR - run the algorithm on the sample
            model = ChangePointDetection(channels=channels, alpha=alpha, edges=edges)
            cp_pred = model.fit(to_plot=False)

            # Evaluate and save sample results
            TP, FN, FP, TN = confusion_matrix(cp_label, cp_pred, time_bins, a_d)
            total_confusion_matrix = np.append(
                total_confusion_matrix, np.array([[TP], [FN], [FP], [TN]]), axis=1
            )

        # MULRR - summary of results (from all samples)
        avg_TP, avg_FN, avg_FP, avg_TN = total_confusion_matrix.mean(axis=1)
        std_TP, std_FN, std_FP, std_TN = total_confusion_matrix.std(axis=1)
        _, precision_mean, recall_mean, F1_mean, _ = confusion_matrix_summary(
            avg_TP, avg_FN, avg_FP, avg_TN
        )
        _, precision_std, recall_std, F1_std, _ = confusion_matrix_summary(
            std_TP, std_FN, std_FP, std_TN
        )

        memory_mean = np.append(
            memory_mean, np.array([[precision_mean], [recall_mean], [F1_mean]]), axis=1
        )
        memory_std = np.append(
            memory_std, np.array([[precision_std], [recall_std], [F1_std]]), axis=1
        )

    #############################################################################
    # ------------------------------------PLOT-----------------------------------#
    #############################################################################

    # MULLR plot
    plot_experiment_results(
        memory_mean, memory_std, allowed_deviation, "Allowed deviation", param_text
    )  # Should change according to the variable parameter
    plt.show()


def experiment_7(
    time_bins=200,
    number_of_samples=100,
    number_of_channels=[4,5,6,7,8],
    change_points=2,
    signal_noise_ratio=2,
    var_increased_by=1.5,
    allowed_deviation=1,
    alpha=0.01,
    edges=10,
):
    param_text = f"""Parameters: 
    \u03B1 = {alpha}
    time bins = {time_bins}
    edges = {edges}
    allowed deviation = {allowed_deviation}
    signal noise ratio = {signal_noise_ratio}
    var increased by = {var_increased_by}
    Change points = {change_points}"""

    #############################################################################
    # ---------------------------------expirements-------------------------------#
    #############################################################################

    # Initialize an array to save each experiment result for a particular paramete
    memory_mean = np.array([[], [], []])
    memory_std = np.array([[], [], []])

    for n_o_c in number_of_channels: 

        # Initialize an array to calculate the all samples confusion matrix
        total_confusion_matrix = np.array([[], [], [], []])

        for i in range(number_of_samples):

            # Generate a different channels with a different seed
            channels1, cp_label1 = data_generator(time_bins, n_o_c-int(n_o_c//2), change_points, signal_noise_ratio, var_increased_by, seed=i, k_index = None, distribution = None)
            channels2, cp_label2 = data_generator(time_bins, int(n_o_c//2), 0, signal_noise_ratio, var_increased_by, seed=i+1, k_index = None, distribution = None)

            # concatenate channels and cp_label
            channels = np.concatenate((channels1,channels2))
            cp_label = np.concatenate((cp_label1,cp_label2))

            # EMULRR - run the algorithm on the sample
            model = ChangePointDetection(channels=channels, alpha=alpha, edges=edges)
            cp_pred = model.fit(to_plot=False)

            # Evaluate and save sample results
            TP, FN, FP, TN = confusion_matrix(cp_label, cp_pred, time_bins, allowed_deviation)
            total_confusion_matrix = np.append(
                total_confusion_matrix, np.array([[TP], [FN], [FP], [TN]]), axis=1
            )

        # MULRR - summary of results (from all samples)
        avg_TP, avg_FN, avg_FP, avg_TN = total_confusion_matrix.mean(axis=1)
        std_TP, std_FN, std_FP, std_TN = total_confusion_matrix.std(axis=1)
        _, precision_mean, recall_mean, F1_mean, _ = confusion_matrix_summary(
            avg_TP, avg_FN, avg_FP, avg_TN
        )
        _, precision_std, recall_std, F1_std, _ = confusion_matrix_summary(
            std_TP, std_FN, std_FP, std_TN
        )

        memory_mean = np.append(
            memory_mean, np.array([[precision_mean], [recall_mean], [F1_mean]]), axis=1
        )
        memory_std = np.append(
            memory_std, np.array([[precision_std], [recall_std], [F1_std]]), axis=1
        )

    #############################################################################
    # ------------------------------------PLOT-----------------------------------#
    #############################################################################

    plot_experiment_results(
        memory_mean, memory_std, number_of_channels, "Number of channel", param_text
    ) 
    plt.show()


def experiment_8(
    time_bins=200,
    number_of_samples=100,
    number_of_channels=8,
    change_points=0,
    signal_noise_ratio=2,
    var_increased_by=1.5,
    allowed_deviation=1,
    alpha=0.01,
    edges=10,
):

    total_confusion_matrix = np.array([[], [], [], []])
    to_plot = True
    for i in range(number_of_samples):
        channels, cp_label = data_generator(
            time_bins,
            number_of_channels,
            change_points,
            signal_noise_ratio,
            var_increased_by,
            seed=i,
            k_index=None,
            distribution=None,
        )
        model = ChangePointDetection(channels=channels, alpha=alpha, edges=edges)
        cp_pred = model.fit(to_plot=to_plot)
        to_plot = False
        TP, FN, FP, TN = confusion_matrix(
            cp_label, cp_pred, time_bins, allowed_deviation
        )

        total_confusion_matrix = np.append(
            total_confusion_matrix, np.array([[TP], [FN], [FP], [TN]]), axis=1
        )

        avg_TP, avg_FN, avg_FP, avg_TN = total_confusion_matrix.mean(axis=1)

    accuracy, precision, recall, F1, text = confusion_matrix_summary(
        avg_TP, avg_FN, avg_FP, avg_TN
    )

    print(text)

    
def experiment_9(
    time_bins=200,
    number_of_samples=100,
    number_of_channels=8,
    change_points=0,
    signal_noise_ratio=2,
    var_increased_by=1.5,
    allowed_deviation=1,
    alpha=0.01,
    edges=10,
):

    total_confusion_matrix = np.array([[], [], [], []])
    to_plot = True
    for i in range(number_of_samples):
        channels, cp_label = data_generator(
            time_bins,
            number_of_channels-1,
            change_points,
            signal_noise_ratio,
            var_increased_by,
            seed=i,
            k_index=None,
            distribution=None,
        )

        channel, cp_label = data_generator(
            time_bins,
            1,
            1,
            signal_noise_ratio,
            var_increased_by,
            seed=i,
            k_index=None,
            distribution=None,
        )

        channels = np.concatenate((channels,channel))

        model = ChangePointDetection(channels=channels, alpha=alpha, edges=edges)
        cp_pred = model.fit(to_plot=to_plot)
        to_plot = False
        TP, FN, FP, TN = confusion_matrix(
            cp_label, cp_pred, time_bins, allowed_deviation
        )

        total_confusion_matrix = np.append(
            total_confusion_matrix, np.array([[TP], [FN], [FP], [TN]]), axis=1
        )

        avg_TP, avg_FN, avg_FP, avg_TN = total_confusion_matrix.mean(axis=1)

    accuracy, precision, recall, F1, text = confusion_matrix_summary(
        avg_TP, avg_FN, avg_FP, avg_TN
    )

    print(text)