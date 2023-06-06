import os
import argparse
import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from transformers import T5Tokenizer, AutoTokenizer

SEED = 42

class DataPreprocessor():
    def __init__(self,
                 dataset_load_path: str = "author_datasets/arguments_Amablue",
                 dataset_save_path: str = "author_datasets/arguments_Amablue",
                 model_checkpoint: str = "google/flan-t5-small",
                 input_col_name: str = "body",
                 max_source_length: int = 512,
                 max_target_length: int = 128,
                 long_T5: int = 0,) -> None:

        self.dataset_load_path = dataset_load_path
        self.dataset_save_path = dataset_save_path + "_" + input_col_name
        if long_T5:
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
        self.input_col_name = input_col_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def load_dataset(self):
        self.datasets = load_from_disk(self.dataset_load_path)

    def tokenize_dataset(self):
        raw_dataset_columns = self.datasets["test"].column_names

        for split in self.datasets.keys():
            self.datasets[split] = self.datasets[split].map(
                self._tokenize,
                batched=True,
                remove_columns=raw_dataset_columns,
            )
        self.datasets.set_format(type="torch")

    def _tokenize(self, examples):
        model_inputs = self.tokenizer(
            examples[self.input_col_name], truncation=True, max_length=self.max_source_length)
        return model_inputs

    def save_dataset(self):
        self.datasets.save_to_disk(self.dataset_save_path)

    def report(self):
        for split in self.datasets.keys():
            print(split + " data:")
            print(self.datasets[split])


def preprocess_argument_dataset(dataset_load_path, dataset_save_path, long_T5=False):
    if long_T5:
        model_checkpoint = "google/long-t5-local-base"
        max_source_length = 2048
    else:
        model_checkpoint = "google/flan-t5-base"
        max_source_length = 512

    preprocessor = DataPreprocessor(dataset_load_path=dataset_load_path,
                                    dataset_save_path=dataset_save_path,
                                    model_checkpoint=model_checkpoint)
    preprocessor.load_dataset()
    print("BEFORE")
    preprocessor.report()
    preprocessor.tokenize_dataset()

    print("AFTER:")
    preprocessor.report()
    preprocessor.save_dataset()
    return

def get_author_comment_count(ARGS):
    df_subreddits = pd.read_csv(ARGS.file_csv)
    #print(df_subreddits.value_counts("author").head(ARGS.top_k_authors))

    # drop empty
    df_subreddits['author'].replace('', np.nan, inplace=True)
    df_subreddits['score'].replace('', 0, inplace=True)
    df_subreddits['body'].replace('', np.nan, inplace=True)

    # drop duplicates
    df_subreddits.drop_duplicates(keep='first', inplace=True)

    # drop Nans
    df_subreddits = df_subreddits.dropna()

    if not os.path.isdir(ARGS.dir_path_orig):
        os.makedirs(ARGS.dir_path_orig)

    if not os.path.isdir(ARGS.dir_path_preprocessed):
        os.makedirs(ARGS.dir_path_preprocessed)

    print("Creating original datasets..........")
    for author, count in df_subreddits.value_counts("author").head(ARGS.top_k_authors).items():
        print(author, count)

        dataset_dict = DatasetDict()
        df_author = df_subreddits[df_subreddits.author == author]
        dataset_dict["test"] = Dataset.from_pandas(df_author)
        dataset_dict.save_to_disk(os.path.join(ARGS.dir_path_orig, f"arguments_{author}"))
    print("Completed creating original datasets\n")

    print("Creating preprocessed datasets..........")
    list_orig_datasets = sorted(os.listdir(ARGS.dir_path_orig))
    for author_dataset_orig in list_orig_datasets:
        preprocess_argument_dataset(
            os.path.join(ARGS.dir_path_orig, author_dataset_orig),
            os.path.join(ARGS.dir_path_preprocessed, author_dataset_orig),
            long_T5=bool(ARGS.long_T5)
        )
    print("Completed creating preprocessed datasets")

    return

def main():
    file_csv = "subreddit_threads_prompt.csv"
    dir_path_orig = "author_datasets_orig_prompt"
    dir_path_preprocessed = "author_datasets_preprocessed_prompt"
    top_k_authors = 30
    long_T5 = 0

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dir_path_orig", default=dir_path_orig,
        type=str, help="full dir path to save the original individual author datasets")
    parser.add_argument("--dir_path_preprocessed", default=dir_path_preprocessed,
        type=str, help="full dir path to save the preprocessed individual author datasets")
    parser.add_argument("--file_csv", default=file_csv,
        type=str, help="full path to the csv file")
    parser.add_argument("--top_k_authors", default=top_k_authors,
        type=int, help="display counts for top k authors")
    parser.add_argument("--long_T5", default=long_T5,
        type=int, choices=[0, 1], help="to indicate long_T5 (1=True) or not (0=False)")

    ARGS, unparsed = parser.parse_known_args()

    get_author_comment_count(ARGS)
    return

if __name__ == "__main__":
    main()
