import random
import numpy as np

def data_generator(
    n=365,
    number_of_channels=1,
    num_of_cp=2,
    signal_noise_ratio=2,
    var_increse_by=1.5,
    seed=1,
    k_index=None,
    distribution=None,
    cp_options=None,

):    # sourcery skip: low-code-quality
    """
    The function generates an array of channels that are distributed according to Linear regression or according to Gaussian distribution
    Args:
        n: time bins.
        number_of_channels: Number of channels to generete.
        num_of_cp: The number of change points to create in each channel.
        signal_noise_ratio: Signal to noise ratio - relevant to changes in mean and slope.
        var_increse_by: How much to multiply or divide variance after a change point.
        seed: Must be changed with each call to the function to get complete randomness.
        k_index: Change points can be defined in advance.
        distribution: Distribution can be defined in advance - "gaussian" / "linear_regression".
    Returns
        channels: The array of channels - (number_of_channels,n).
        index_of_cp_locations:  The index (time bin-1) of the true change points.
    """

    # Initializations
    np.random.seed(seed)  # Seed initialization
    channels = np.zeros((number_of_channels, n))  # Channels array initialization
    distributions = ["gaussian", "linear_regression"]  # Optionality distributions
    cp_options_reg = [
        "trend",
        "var",
        "trend&var",
    ]  # Linear Regression change point options
    cp_options_gaussian = ["mean", "var", "mean&var"]  # Gussian change point options

    # If the change points are given:
    if k_index:
        cp_locations = [0] + k_index + [n]
        num_of_cp = len(k_index)
    # else if not pick them at random
    else:
        cp_locations = [0] + random.sample(range(9, n - 9), num_of_cp) + [n]
    cp_locations.sort()

    # If distribution is given:
    if distribution:
        distributions = [distribution]
    # If cp_options is given:
    if cp_options:
      cp_options_gaussian = cp_options
      cp_options_reg = cp_options

    # --------------------------------------Channels generator:---------------------------------------------------#

    for j in range(number_of_channels):
        channel = np.array([[]])  # Initializations
        channel_dist = random.choice(
            distributions
        )  # For each channel the distribution is randomly selected

        # --------------------------------------Gaussian channel generator:---------------------------------------------------#
        if channel_dist == "gaussian":

            # First segment:
            var = np.random.randint(20, 100) / 100  # range: 0.2-1, jump: 0.01
            mean = np.random.randint(1, 20)  # range: 1-20, jump: 1
            size = cp_locations[1] - cp_locations[0]  # Segmet length
            channel = np.random.normal(mean, var, size=size)  # generate first segment

            # The other segments
            for i in range(1, num_of_cp + 1):
                np.random.seed(seed * 100 + j * 10 + i)  # change seed for every segment
                size = cp_locations[i + 1] - cp_locations[i]  # Segmet length

                # Gussian change point options (mean, var, both)
                # In each segment the change can be done on mean, variance or both and chosen at random
                cp_options = random.choice(cp_options_gaussian)
                if cp_options == "mean":
                    mean = mean + var * signal_noise_ratio
                    channel = np.append(channel, np.random.normal(mean, var, size=size))
                elif cp_options == "var":
                    var = var * var_increse_by
                    channel = np.append(channel, np.random.normal(mean, var, size=size))
                    var_increse_by = 1 / var_increse_by
                elif cp_options == "mean&var":
                    mean = mean + var * signal_noise_ratio
                    var = var * var_increse_by
                    channel = np.append(channel, np.random.normal(mean, var, size=size))
                    var_increse_by = 1 / var_increse_by

        # --------------------------------------Linear regression channel generator:---------------------------------------------------#

        elif channel_dist == "linear_regression":

            # First segment:
            var = np.random.randint(6, 15)  # range: 6-14, jump: 1
            trend = np.random.randint(10, 100) / 100  # range: 0.1-1, jump: 0.01
            size = cp_locations[1] - cp_locations[0]  # Segmet length
            channel = np.linspace(trend, size * trend, num=size) + np.random.normal(
                0, var, size
            )  # generate first segment

            # The other segments
            for i in range(1, num_of_cp + 1):
                np.random.seed(seed * 100 + j * 10 + i)  # change seed for every segment
                size = cp_locations[i + 1] - cp_locations[i]  # Segmet length
                last_y = channel[-1]

                # Linear Regression change point options (trend, var, both)
                # In each segment the change can be done on trend, variance or both and chosen at random
                cp_options = random.choice(cp_options_reg)
                if cp_options == "trend":
                    trend = trend + signal_noise_ratio * (1 / var)
                    segment = np.linspace(
                        last_y + trend, size * trend + last_y, num=size
                    ) + np.random.normal(0, var, size)
                elif cp_options == "var":
                    var = var * var_increse_by
                    segment = np.linspace(
                        last_y + trend, size * trend + last_y, num=size
                    ) + np.random.normal(0, var, size)
                    var_increse_by = 1 / var_increse_by
                elif cp_options == "trend&var":
                    trend = trend + signal_noise_ratio * (1 / var)
                    var = var * var_increse_by
                    segment = np.linspace(
                        last_y + trend, size * trend + last_y, num=size
                    ) + np.random.normal(0, var, size)
                    var_increse_by = 1 / var_increse_by
                channel = np.append(channel, segment)

        # Add the channel to the channels array
        channels[j] = channel

    # Get the indexes of the change point
    index_of_cp_locations = [x - 1 for x in cp_locations[1:-1]]

    return channels, index_of_cp_locations
