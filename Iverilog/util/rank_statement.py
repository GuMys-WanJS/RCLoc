import joblib
import pandas as pd
import sys


def get_method_list(pred_label, path, bugs):
    rank_list_map = dict()
    for bug in bugs:
        f = open(pred_label + bug, 'r')
        lines = f.readlines()
        f.close()

        method_list = pd.read_csv(path + "method_name/{}.csv".format(bug))['method_name'].to_list()
        sort_list = []
        for i in range(len(method_list)):
            sort_list.append((method_list[i], float(lines[i].strip())))
        rank_list_map[bug] = dict()
        sorted_list = sorted(sort_list, key=lambda x: x[1], reverse=True)
        rank_list_map[bug]['DeepFL'] = sorted_list

    return rank_list_map


def get_rank_list_statement(pred_label, path, bugs, bug_items):
    rank_list_all = get_method_list(pred_label, path, bugs)
    # coveraged_methods = joblib.load("bug_coveraged_method.dump")
    valid_lines = joblib.load(path + "valid_lines_completion.dump")

    statement_rank_list = dict()
    for bug in bugs:
        id = int(bug.split('_')[1])
        if bug not in rank_list_all:
            continue
        rank_list = rank_list_all[bug]['DeepFL']
        bug_valid_lines = valid_lines[bug]
        rank_list_for_statement = []
        for each in rank_list:
            file, start, end = each[0].split("#")
            for i in range(int(start), int(end) + 1):
                line_name = file + ":" + str(i)
                if line_name in bug_valid_lines:
                    rank_list_for_statement.append((line_name, each[1]))
        rank_list = sorted(rank_list_for_statement, key=lambda x: x[1], reverse=True)

        bug_lines = bug_items[bug_items.id == id]['line'].to_list()[0].split(",")

        statement_rank_list_for_bug = []
        for bug_line in bug_lines:
            rank = -1
            for each in rank_list:
                if bug_line == each[0]:
                    first_rank,last_rank = -1,-1
                    for idx,item in enumerate(rank_list):
                        if item[1] == each[1]:
                            if first_rank == -1:
                                first_rank = idx + 1
                            last_rank = idx + 1
                        if item[1] != each[1] and first_rank != -1:
                            break
                    rank = (first_rank + last_rank) // 2
                    break
            if rank == -1:
                rank = len(bug_valid_lines)//2
            statement_rank_list_for_bug.append((bug_line,rank))
        if len(statement_rank_list_for_bug) == 0:
            continue
        statement_rank_list[bug] = dict()
        statement_rank_list[bug]['DeepFL'] = statement_rank_list_for_bug

    return statement_rank_list

def test_statement(pred_label, path, bugs):
    path = '../Datasets/replenish/iverilog/'
    bug_items = pd.read_csv(path + "method_lines.csv", encoding='gb18030')
    rank_list = get_rank_list_statement(pred_label, path, bugs, bug_items)

    method_list, top1_list, top3_list, top5_list, top10_list, mfr_list, mar_list = [], [],[],[],[],[],[]
    method_names = rank_list[list(rank_list.keys())[0]].keys()
    for method_name in method_names:
        top1, top3, top5, top10 = 0, 0, 0, 0
        mar = 0
        mfr, bug_num = 0, 0

        for bug in bugs:
            bug_mar, bug_element = 0, 0
            if bug not in rank_list:
                continue
            rank_min = 1e9
            for item in rank_list[bug][method_name]:
                bug_mar += item[1]
                rank_min = min(rank_min, item[1])
                bug_element += 1
            if rank_min == 1:
                top1 += 1
                top3 += 1
                top5 += 1
                top10 += 1
            elif rank_min <= 3:
                top3 += 1
                top5 += 1
                top10 += 1
            elif rank_min <= 5:
                top5 += 1
                top10 += 1
            elif rank_min <= 10:
                top10 += 1
            mfr += rank_min
            bug_num += 1
            mar += bug_mar / bug_element
        method_list.append(method_name)
        top1_list.append(top1)
        top3_list.append(top3)
        top5_list.append(top5)
        top10_list.append(top10)
        mfr_list.append("{:.2f}".format(mfr / bug_num))
        mar_list.append("{:.2f}".format(mar / bug_num))


    resultMatrix = []
    resultMatrix.append(top1_list[0])
    resultMatrix.append(top3_list[0])
    resultMatrix.append(top5_list[0])
    resultMatrix.append(top10_list[0])
    resultMatrix.append(mfr_list[0])
    resultMatrix.append(mar_list[0])

    print("Top-1   Top-3   Top-5   Top-10   MFR   MAR")
    for metric in range(0, 6):
        sys.stdout.write(str(resultMatrix[metric]) + "\t")
    print('')
    return resultMatrix


