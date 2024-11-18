import numpy as np
import sys
import os.path

def readScore(pred_label_path, true_label_path):
    p = open(pred_label_path)
    pred_list = [line.rstrip('\n') for line in p]
    l = open(true_label_path)
    label_list =[line.rstrip('\n').split(',')[0] for line in l]
    pred_list = np.asarray(pred_list, np.float32)
    label_list = np.asarray(label_list, np.int32)

    return pred_list, label_list

def getTopN_MAR_MFR(pred_list, label_list):
    bug_site = np.where(label_list == 1)[0]
    sort_pred = np.argsort(pred_list)[::-1]

    bug_set = set(bug_site)
    top1_set = set(sort_pred[:1])
    top3_set = set(sort_pred[:3])
    top5_set = set(sort_pred[:5])
    top10_set = set(sort_pred[:10])

    top1 = int(not bug_set.isdisjoint(top1_set))
    top3 = int(not bug_set.isdisjoint(top3_set))
    top5 = int(not bug_set.isdisjoint(top5_set))
    top10 = int(not bug_set.isdisjoint(top10_set))

    ranks = [np.where(sort_pred == element)[0][0] for element in bug_site]

    if len(ranks) == 0:
        return top1, top3, top5, top10, -1, -1

    average_rank = np.mean([rank + 1 for rank in ranks])
    first_rank = ranks[0] + 1.0

    return top1, top3, top5, top10, average_rank, first_rank

def calculate(pred_label, true_label, bugs):
    tops = np.zeros(4)
    ranks = np.zeros(2)
    actual_ver = 0
    for bug_id in bugs:
        pred_label_path = pred_label + bug_id
        true_label_path = true_label + bug_id + '/TestLabel.csv'
        pred_list, label_list = readScore(pred_label_path, true_label_path)
        top1, top3, top5, top10, average_rank, first_rank = getTopN_MAR_MFR(pred_list, label_list)
        if average_rank == -1:
            continue
        tops[0] += top1
        tops[1] += top3
        tops[2] += top5
        tops[3] += top10
        ranks[0] += average_rank
        ranks[1] += first_rank
        actual_ver += 1

    ranks = ranks / actual_ver
    result = (int(tops[0]),
              int(tops[1]),
              int(tops[2]),
              int(tops[3]),
              round(float(ranks[0]), 2),
              round(float(ranks[1]), 2))
    result = np.array(result, dtype=object)

    return result

def test_file(pred_label, true_label, bugs):
    resultMatrix = calculate(pred_label, true_label, bugs)
    print("Top-1   Top-3   Top-5   Top-10   MFR   MAR")
    for metric in range(0, 6):
        sys.stdout.write(str(resultMatrix[metric]) + "\t")
    print('')
    return list(resultMatrix)




