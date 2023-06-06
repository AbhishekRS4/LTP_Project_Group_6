import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_dominant_categories_from_json(file_json, list_required_authors):
    """
    ---------
    Arguments
    ---------
    file_json : str
        full path of json file to be saved
    list_required_authors : list
        list of authors for whom dominant categories need to be collated

    -------
    Returns
    -------
    dict_data : dict
        dictionary of params loaded from the json file
    """
    #print(list_required_authors)
    dict_data = {}
    with open(file_json) as fh:
        list_json = list(fh)
        #print(len(list_json))
        for json_str_index in range(len(list_json)):
            author_data = json.loads(list_json[json_str_index])
            author = author_data["author"]
            if author in list_required_authors:
                #print(author)
                if author not in dict_data:
                    dict_data[author] = {}

                categories = list(author_data['subreddits'].keys())

                for category in categories:
                    if category not in dict_data[author]:
                        dict_data[author][category] = 0
                    count_per_category = sum(list(author_data['subreddits'][category].values()))
                    dict_data[author][category] = count_per_category

    #print(f"result: {result['subreddits']}")
    #print(f"result: {len(result['subreddits'])}")
    for author in list(dict_data.keys()):
        dict_orig = dict_data[author]
        dict_new = {y: x for x, y in dict_orig.items()}
        dict_data[author] = dict_new
    return dict_data

"""
def plot_histogram(dict_category_counts):
    fig = plt.figure(0)
    plt.bar(list(dict_category_counts.keys()), np.array(list(dict_category_counts.values())))
    plt.grid()
    plt.title("Distribution of subreddit categories ", fontsize=16)
    plt.ylabel("Count", fontsize=20)
    plt.xlabel(f"Subreddit category", fontsize=20)
    plt.xticks(fontsize=16, rotation=30, ha='right')
    plt.yticks(fontsize=16)
    plt.show()
    #plt.save("histogram.png", dpi=fig.dpi)
    return
"""

def get_dominant_categories(ARGS):
    list_required_authors = os.listdir(ARGS.dir_author_datasets)
    list_required_authors = [author.split('_')[1] for author in list_required_authors]
    dict_category_data = get_dominant_categories_from_json(ARGS.file_json, list_required_authors)
    print(dict_category_data)
    print(len(dict_category_data))
    #plot_histogram(dict_category_data)
    return

def main():
    file_json = "../../author_subreddit_category.jsonl"
    dir_author_datasets = "author_datasets_orig_prompt"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--file_json", default=file_json,
        type=str, help="full path to the jsonl file")
    parser.add_argument("--dir_author_datasets", default=dir_author_datasets,
        type=str, help="full path to the directory containing author datasets")
    ARGS, unparsed = parser.parse_known_args()
    get_dominant_categories(ARGS)
    return

if __name__ == "__main__":
    main()
