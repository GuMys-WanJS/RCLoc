from model import MLP_CL as mlp_cl
from util.rank_file import test_file
from util.rank_statement import test_statement
import pandas as pd

# main run driver
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def main(threshold, dataset):
    cvae_file = "../Datasets/" + dataset + "Datasets/"
    bugs = ["bug_02", "bug_03", "bug_05", "bug_06", "bug_07", "bug_08", "bug_09", "bug_10", "bug_12"]

    # bugs = ["bug_02", "bug_03", "bug_05", "bug_06", "bug_07", "bug_08", "bug_09", "bug_10", "bug_12",
    #         "bug_13", "bug_14", "bug_15", "bug_16", "bug_17", "bug_18", "bug_19", "bug_21", "bug_23",
    #         "bug_24", "bug_25", "bug_26", "bug_27", "bug_29", "bug_31", "bug_32", "bug_36", "bug_38",
    #         "bug_39", "bug_43", "bug_45", "bug_46", "bug_49", "bug_51", "bug_52", "bug_61", "bug_67",
    #         "bug_68", "bug_69", "bug_70", "bug_71", "bug_77", "bug_79", "bug_81", "bug_83", "bug_84"]
    if dataset == 'if':
        test = test_file
    else:
        test = test_statement

    for bug_id in bugs:
        print("============================= {}  Threshold: {}==================================".format(bug_id, threshold))
        train_path = cvae_file + bug_id + "/Train.csv"
        train_label_path = cvae_file + bug_id + "/TrainLabel.csv"
        test_path = cvae_file + bug_id + "/Test.csv"
        test_label_path = cvae_file + bug_id + "/TestLabel.csv"

        mlp_cl.run(train_path, train_label_path, test_path, test_label_path, 1, bug_id, threshold)

    res = test("./Result/", cvae_file, bugs)
    return res
# main function execution
if __name__=='__main__':
    res = main(0.0007, "if")
    print(res)

