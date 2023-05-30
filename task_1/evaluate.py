import fastwer
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def compute_metrics(true_labels, pred_labels, average="macro"):
    acc_sc = accuracy_score(true_labels, pred_labels)
    f1_sc = f1_score(true_labels, pred_labels, average=average)
    prec_sc = precision_score(true_labels, pred_labels, average=average)
    rec_sc = recall_score(true_labels, pred_labels, average=average)

    print(f"accuracy score: {acc_sc:.4f}")
    print(f"precision score: {prec_sc:.4f}")
    print(f"recall score: {rec_sc:.4f}")
    print(f"f1 score: {f1_sc:.4f}")

    return

def find_closes_true_label(pred_label, list_true_labels):
    closest_true_label = None
    # compute character error scores with all true labels
    cer_scores = np.array([])
    for true_label in list_true_labels:
        cer_score = fastwer.score_sent(pred_label, true_label, char_level=True)
        cer_scores = np.append(cer_scores, cer_score)
    # find the closest true label with the smalles character error score
    index_with_min_cer_score = np.argmin(cer_scores)
    return index_with_min_cer_score

def convert_pred_string_labels_to_int_labels(string_pred_labels, list_true_labels, delimiter="\t"):
    num_samples = len(string_pred_labels)
    num_labels = len(list_true_labels)
    int_labels = np.zeros((num_samples, num_labels))

    for sample_index in range(len(string_pred_labels)):
        list_pred_labels = string_pred_labels[sample_index].split(delimiter)
        for pred_label in list_pred_labels:
            closest_true_label_index = find_closes_true_label(pred_label, list_true_labels)
            int_labels[sample_index, closest_true_label_index] = 1
    return int_labels

def main():
    # example to compute metrics
    # 5 samples, 3 class labels
    true_labels = [[1, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]]
    pred_labels = [[1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1]]
    compute_metrics(true_labels, pred_labels)

    # example to convert prediction string labels to int labels
    # predictions can be wrong
    # 2 samples, first one with incorrect predictions, second one with correct predictions
    delimiter = "\t"
    pred_labels = [
        f"Self-directio: thought{delimiter}Sef-diretion: action{delimiter}Stimltion",
        f"Self-direction: thought{delimiter}Self-direction: action{delimiter}Stimulation{delimiter}Universalism: tolerance{delimiter}Universalism: objectivity"
    ]
    df_train = pd.read_csv("dataset/labels-training.tsv", sep=delimiter)
    list_true_labels = list(df_train.columns)[1:]
    int_labels = convert_pred_string_labels_to_int_labels(pred_labels, list_true_labels)
    print("\nTrue string labels")
    print(list_true_labels)
    print("Predicted string labels")
    print(pred_labels)
    print("Predicted int labels")
    print(int_labels)
    return


main()
