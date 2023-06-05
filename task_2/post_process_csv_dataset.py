import argparse
import numpy as np
import pandas as pd

def get_author_comment_count(ARGS):
    df_subreddits = pd.read_csv(ARGS.file_csv)
    #print(df_subreddits.head())

    # replace all empties with Nans
    df_subreddits['author'].replace('', np.nan, inplace=True)
    df_subreddits['subreddit'].replace('', np.nan, inplace=True)
    df_subreddits['body'].replace('', np.nan, inplace=True)
    #print(df_subreddits.head())

    # drop duplicates
    df_subreddits.drop_duplicates(keep='first', inplace=True)
    #print(df_subreddits.head())

    # drop Nans
    df_subreddits = df_subreddits.dropna()
    #print(df_subreddits.head())
    print(df_subreddits.value_counts("author").head(ARGS.top_k_authors))

    df_subreddits.to_csv("subreddit_threads.csv", index=False)
    return

def main():
    file_csv = "subreddit_threads.csv"
    top_k_authors = 20

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--file_csv", default=file_csv,
        type=str, help="full path to the csv file")
    parser.add_argument("--top_k_authors", default=top_k_authors,
        type=int, help="display counts for top k authors")

    ARGS, unparsed = parser.parse_known_args()

    get_author_comment_count(ARGS)
    return

if __name__ == "__main__":
    main()
