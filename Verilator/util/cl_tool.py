import copy
import numpy as np
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import (
    compute_confident_joint,
    estimate_latent,
)

def remove_noise_threshold(labels, pre, X, y, threshold):

    label_new1 = copy.deepcopy(labels)
    pre_new1 = copy.deepcopy(pre)

    label_1 = label_new1.ravel()
    y_train2 = label_1.astype(np.int16)
    confident_joint = compute_confident_joint(
        s=y_train2,
        psx=pre_new1,  # P(s = k|x)
        thresholds=[threshold, 1]
    )

    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=y_train2,
        py_method='cnt',
        converge_latent_estimates=False
    )

    ordered_label_errors = get_noise_indices(
        s=y_train2,
        psx=pre_new1,
        inverse_noise_matrix=inv_noise_matrix,
        confident_joint=confident_joint,
        prune_method='prune_by_noise_rate',
    )
    # Adjust filtering criteria to be more stringent
    # Filter based on ordered_label_errors and threshold
    x_mask = ~ordered_label_errors
    all_labels = np.array(labels)

    print("数组中正样本个数", np.sum(all_labels == 1))
    print("数组中负样本个数", np.sum(all_labels == 0))

    ids_positive = np.where(y_train2 == 1)[0]
    # ids = np.concatenate((ids_positive, x_mask, ids_positive))

    new_labels = all_labels[x_mask]


    print("过滤之后的数组中正样本个数", np.sum(new_labels == 1))
    print("过滤之后的数组中负样本个数", np.sum(new_labels == 0))

    temp_x = X[x_mask]
    temp_y = y[x_mask]

    now_x = temp_x[new_labels != 1]
    now_y = temp_y[new_labels != 1]

    train_instances = np.concatenate((now_x, X[ids_positive]), axis=0)
    train_labels = np.concatenate((now_y, y[ids_positive]), axis=0)

    return train_instances, train_labels


