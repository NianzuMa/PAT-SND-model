from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score


def get_auc_score(y_true, pred_score):
    auc_score = roc_auc_score(y_true, pred_score)
    return auc_score


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1_multiclass(preds, labels, label_str_list):
    acc = simple_accuracy(preds, labels)
    f1_micro = f1_score(y_true=labels, y_pred=preds, average='micro')
    f1_macro = f1_score(y_true=labels, y_pred=preds, average='macro')
    f1_weighted = f1_score(y_true=labels, y_pred=preds, average='weighted')
    f1_none = f1_score(y_true=labels, y_pred=preds,
                       average=None).tolist()  # If None, the scores for each class are returned
    return {
        "acc": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f_None": f1_none,
        "classification_report": classification_report(labels, preds, digits=4, labels=list(range(len(label_str_list))),
                                                       target_names=label_str_list)
    }


def in_top_k_multiclass(top_k_preds, labels, label_str_list):
    """
    as long as golden label is in the top k the prediction, it is counted as correct
    :param top_k_preds:
    :param labels:
    :param label_str_list:
    :return:
    """
    new_preds_list = []
    for i in range(top_k_preds.shape[0]):
        pred_range = set(list(top_k_preds[i]))
        l = int(labels[i])
        if l in pred_range:
            new_preds_list.append(l)
        else:
            new_preds_list.append(-1)
        # endif
    # endfor
    new_preds_list = np.array(new_preds_list)
    result = acc_and_f1_multiclass(new_preds_list, labels, label_str_list=label_str_list)
    return result


def novelty_evalution(unique_id_list, top_k_preds, labels):
    new_preds_list = []
    for i in range(top_k_preds.shape[0]):
        pred_range = set(list(top_k_preds[i]))
        l = int(labels[i])
        if l in pred_range:
            new_preds_list.append(0)
        else:
            new_preds_list.append(1)
        # endif
    # endfor
    # new_preds_list = np.array(new_preds_list)
    # result = acc_and_f1_multiclass(new_preds_list, labels, label_str_list=label_str_list)

    sent_value_list_dict = defaultdict(list)
    for i in range(len(unique_id_list)):
        unique_id = unique_id_list[i]
        sent_id = "_".join(unique_id.split("_")[:-1])
        sent_value_list_dict[sent_id].append(new_preds_list[i])
    # endfor

    sent_sum_value_dict = defaultdict(int)
    for sent_id in sent_value_list_dict:
        sent_sum_value_dict[sent_id] = sum(sent_value_list_dict[sent_id])
    # endfor

    count = 0
    for sent_id, v in sent_sum_value_dict.items():
        if v == 0:
            count += 1

    return count * 1.0 / len(sent_sum_value_dict)


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def pretty_cm_str(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    result_str = ""

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    result_str += "     " + fst_empty_cell
    # End CHANGES

    for label in labels:
        result_str += "%{0}s ".format(columnwidth) % label

    result_str += '\n'
    # Print rows
    for i, label1 in enumerate(labels):
        result_str += "    %{0}s ".format(columnwidth) % label1
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            result_str += cell + " "
        result_str += "\n"
    # endfor
    return result_str


def detailed_report(pred_id_list, true_id_list, label_list):
    idx_to_label_dict = {i: lb for i, lb in enumerate(label_list)}
    y_pred_label = [idx_to_label_dict[idx] for idx in pred_id_list]
    y_gold_label = [idx_to_label_dict[idx] for idx in true_id_list]
    result_json = acc_and_f1_multiclass(pred_id_list, true_id_list, label_list)

    matrix = confusion_matrix(y_gold_label, y_pred_label, label_list)
    confusion_mx_str = pretty_cm_str(matrix, label_list)
    acc_list = matrix.diagonal() / matrix.sum(axis=1)
    class_acc_dict = dict(zip(label_list, acc_list))

    return result_json, confusion_mx_str, class_acc_dict


def Test_in_top_k_multiclass():
    preds = np.random.randint(0, high=10, size=(20, 10), dtype="int")
    preds_top_k = preds[:, -5:]
    labels = np.random.randint(0, high=5, size=20, dtype="int")

    new_preds_list = []
    for i in range(preds_top_k.shape[0]):
        pred_range = set(list(preds_top_k[i]))
        l = int(labels[i])
        if l in pred_range:
            new_preds_list.append(l)
        else:
            new_preds_list.append(-1)
        # endif
        print("hello")
    # endfor

    new_preds_list = np.array(new_preds_list)
    result = acc_and_f1_multiclass(new_preds_list, labels, label_str_list=[str(item) for item in list(range(5))])
    print("hello")

    pass


if __name__ == '__main__':
    Test_in_top_k_multiclass()
