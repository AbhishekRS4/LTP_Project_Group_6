import os
import random
import numpy as np
import pandas as pd

from datasets import load_from_disk

from get_active_category_per_author import get_dominant_categories_from_json

def main():
    top_N = 5
    dir_author_datasets = "author_datasets_orig_prompt/"
    list_authors = os.listdir(dir_author_datasets)
    list_authors = [author.split("_")[1] for author in list_authors]
    print(list_authors)


    file_author_subreddit_json = "../../author_subreddit_category.jsonl"
    dict_authors_dom_cat = get_dominant_categories_from_json(
        file_author_subreddit_json,
        list_authors
    )
    #print(dict_authors_dom_cat)


    file_labels_tsv = "../task_1/dataset/labels-training.tsv"
    df_train = pd.read_csv(file_labels_tsv, sep="\t")
    list_true_labels = list(df_train.columns)[1:]
    print(list_true_labels)

    df_predictions = None
    df_results = None

    for author in list_authors:
        if author in list(dict_authors_dom_cat.keys()):
            file_pred_csv = f"arguments_{author}_body.csv"
            if os.path.isfile(file_pred_csv):
                try:
                    df_predictions = pd.read_csv(file_pred_csv)
                except:
                    print("not found")

                if df_predictions is not None:
                    #print(df_predictions.shape)
                    #print(df_predictions.head())

                    try:
                        dataset_path = f'author_datasets_orig_prompt/arguments_{author}/'
                        dataset = load_from_disk(dataset_path)
                        #print(len(dataset['test']))
                        #print(list(dataset['test'][:].values())[1])
                        list_scores = list(dataset['test'][:].values())[1]
                        list_delta = list(dataset['test'][:].values())[2]
                        df_predictions["score"] = list_scores
                        num_samples = len(list_scores)
                        df_predictions["successful"] = (df_predictions["score"] > 0)
                        df_predictions["delta"] = list_delta

                        list_top_N = list(reversed(sorted(dict_authors_dom_cat[author].keys())))[:top_N]
                        #print(list_top_N, num_samples)
                        list_random_cats_counts = random.choices(list_top_N, k=num_samples)
                        list_random_cats = [dict_authors_dom_cat[author][count] for count in list_random_cats_counts]
                        #print(len(list_random_cats))
                        df_predictions["active_subreddit"] = list_random_cats


                        list_overall_values = []
                        list_int_labels = df_predictions["pred_int_labels"].to_list()
                        #print(len(list_int_labels))
                        for int_values_as_char in list_int_labels:
                            list_string_values = ""
                            int_values_as_char = int_values_as_char[1:-1].split(".")[:-1]

                            int_values = []
                            for x in int_values_as_char:
                                int_values.append(int(x.strip()))
                            int_value_indices = np.nonzero(int_values)[0]
                            #print(int_value_indices)
                            for index in int_value_indices:
                                if list_string_values != "":
                                    list_string_values = list_string_values + "," + list_true_labels[index]
                                else:
                                    list_string_values = list_true_labels[index]
                            list_overall_values.append(list_string_values)

                        df_predictions["values"] = list_overall_values

                        if df_results is None:
                            df_results = df_predictions.copy()
                        else:
                            df_results = pd.concat([df_results, df_predictions], ignore_index=True)

                        print(df_results.shape)
                    except:
                        print("mismatch in samples")

    print(df_results.head())
    df_results.to_csv("final_results.csv", index=False)
    return

if __name__ == "__main__":
    main()
