import numpy as np


def confusion_matrix(labels, preds, n, allowed_deviation=1):

    """
    Calculate the confusion matrix values - TP, FN, FP, TN
    Args:
      labels: True labels.
      preds: Preds.
      n: time bins (The size of the population in terms of the confusion matrix).
      allowed_deviation: The deviation of error allowed for each side.
    Returns:
      TP, FN, FP, TN: TP, FN, FP, TN.
    """

    TP, FN, FP = 0, 0, 0
    preds_with_deviation = []
    labels_with_deviation = []

    for i in preds:
        preds_with_deviation.extend(
            np.arange(i - allowed_deviation, i + allowed_deviation + 1).tolist()
        )
    for i in labels:
        labels_with_deviation.extend(
            np.arange(i - allowed_deviation, i + allowed_deviation + 1).tolist()
        )

    for l1 in labels:
        if l1 in preds_with_deviation:
            TP += 1
        else:
            FN += 1

    for l2 in preds:
        if l2 not in labels_with_deviation:
            FP += 1

    TN = n - (FP + FN + TP)

    return TP, FN, FP, TN


def confusion_matrix_summary(TP, FN, FP, TN):

    """
    Calculate the accuracy, precision, recall, F1 from confusion matrix values.
    Args:
        TP, FN, FP, TN: confusion matrix values.
    Returns
        accuracy, precision, recall, F1: accuracy, precision, recall, F1
        text: Text that summarizes the results and ready for print.
    """
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * (precision * recall)) / (precision + recall)
    text = (
        "\033[1mAccuracy:\033[0;0m {:.0%}".format(accuracy)
        + "\n\033[1mPrecision:\033[0;0m {:.0%}".format(precision)
        + "\n\033[1mRecall:\033[0;0m {:.0%}".format(recall)
        + "\n\033[1mF1:\033[0;0m {:.0%}".format(F1)
    )

    return accuracy, precision, recall, F1, text
