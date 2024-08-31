import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DeepAllele import tools


# TODO: Clean and comment these functions
def influence_ablation(
    ablation_prediction_path, pearson_correlation, original_pearson, n_kernel=256
):

    for kernel in range(n_kernel):
        # metric for all ablations B6_counts, Cast_counts, ratio for pearson correlation

        ablation_predictions = np.load(
            ablation_prediction_path + "predictions_kernel_{}.npy".format(kernel)
        )
        #FIXME: labels should be passed as an argument
        B6_counts_pearson = tools.pearson_r(labels[:, 0], ablation_predictions[:, 0])
        Cast_counts_pearson = tools.pearson_r(labels[:, 1], ablation_predictions[:, 1])
        ratio_pearson = tools.pearson_r(labels[:, 2], ablation_predictions[:, 2])

        pearson_correlation["B6_counts"].append(B6_counts_pearson)
        pearson_correlation["Cast_counts"].append(Cast_counts_pearson)
        pearson_correlation["ratio"].append(ratio_pearson)

    pearson_correlation["B6_counts"] = np.array(pearson_correlation["B6_counts"])
    pearson_correlation["Cast_counts"] = np.array(pearson_correlation["Cast_counts"])
    pearson_correlation["ratio"] = np.array(pearson_correlation["ratio"])

    influence_B6 = original_pearson["B6_counts"] - pearson_correlation["B6_counts"]
    influence_Cast = (
        original_pearson["Cast_counts"] - pearson_correlation["Cast_counts"]
    )
    influence_ratio = original_pearson["ratio"] - pearson_correlation["ratio"]

    influence_B6 = (influence_B6 - influence_B6.min()) / (
        influence_B6.max() - influence_B6.min()
    )
    influence_Cast = (influence_Cast - influence_Cast.min()) / (
        influence_Cast.max() - influence_Cast.min()
    )
    influence_ratio = (influence_ratio - influence_ratio.min()) / (
        influence_ratio.max() - influence_ratio.min()
    )

    # save the influence as dataframe
    influence_df = pd.DataFrame(
        {
            "kernel": range(256),
            "influence_B6": influence_B6,
            "influence_Cast": influence_Cast,
            "influence_ratio": influence_ratio,
        }
    )

    # convert kernel from int to filter+kernel
    influence_df["kernel"] = influence_df["kernel"].apply(
        lambda x: "filter{}".format(x)
    )

    return influence_df


def contribution_ablation(prediction, ablation_prediction, percpetile=95):
    diff = prediction - ablation_prediction
    # only calculate the mean of diff for the absolute value top 5% of the difference
    diff_abs = np.abs(diff)
    diff_select = diff[diff_abs > np.percentile(diff_abs, percpetile)]

    return diff_select.mean()


def ablation_prediction_diff_OCR(
    ablation_path,
    original_predictions,
    peakname,
    n_kernel=256,
):

    diff_OCR_df = {}
    diff_OCR_df["peakname"] = peakname

    for i in range(n_kernel):
        print("The %dth kernel" % i)
        ablation_predictions = np.load(
            ablation_path + "predictions_kernel_{}.npy".format(i)
        )
        diff_B6 = original_predictions[:, 0] - ablation_predictions[:, 0]
        diff_Cast = original_predictions[:, 1] - ablation_predictions[:, 1]
        diff_ratio = original_predictions[:, 2] - ablation_predictions[:, 2]

        diff_OCR_df["filter{}_diff_ratio".format(i)] = diff_ratio
        diff_OCR_df["filter{}_diff_B6".format(i)] = diff_B6
        diff_OCR_df["filter{}_diff_Cast".format(i)] = diff_Cast

    diff_OCR_df = pd.DataFrame(diff_OCR_df)

    return diff_OCR_df
