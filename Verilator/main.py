from model import MLP_CL as mlp_cl
from util.rank_file import test_file
from util.rank_statement import test_statement
import pandas as pd

# main run driver
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def main(threshold, dataset):
    cvae_file = "../Datasets/" + dataset + "Datasets/"
    bugs = ["bug_01", "bug_02", "bug_03", "bug_04", "bug_06", "bug_07", "bug_09", "bug_10", "bug_11",
            "bug_12", "bug_14", "bug_15", "bug_16", "bug_18", "bug_19", "bug_20", "bug_21", "bug_22",
            "bug_23", "bug_24", "bug_25", "bug_26", "bug_28", "bug_29", "bug_30", "bug_31", "bug_32",
            "bug_33", "bug_34", "bug_36", "bug_37", "bug_38", "bug_39", "bug_41", "bug_43", "bug_44",
            "bug_45", "bug_46", "bug_47", "bug_48", "bug_49", "bug_51", "bug_52", "bug_53", "bug_54",
            "bug_55", "bug_56", "bug_57", "bug_58", "bug_59", "bug_64", "bug_69", "bug_71", "bug_72",
            "bug_75", "bug_76", "bug_77", "bug_78", "bug_79", "bug_80", "bug_81", "bug_82", "bug_83",
            "bug_84", "bug_85", "bug_87", "bug_88", "bug_89", "bug_90", "bug_91", "bug_92", "bug_94",
            "bug_95", "bug_96", "bug_97", "bug_98", "bug_99", "bug_100", "bug_101", "bug_102", "bug_103",
            "bug_104", "bug_105", "bug_106", "bug_107", "bug_115", "bug_116", "bug_117", "bug_118", "bug_119",
            "bug_123", "bug_126", "bug_127", "bug_129", "bug_136", "bug_140"]

    if dataset == 'vf':
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
    res = main(0.0001, "vf")
    print(res)


