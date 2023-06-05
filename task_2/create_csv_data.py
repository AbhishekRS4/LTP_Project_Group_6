import io
import os
import sys
import csv
import json
import ijson
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm


class CSVWriter:
    """
    for writing tabular data to a csv file
    """
    def __init__(self, file_name, column_names):
        self.file_name = file_name
        self.column_names = column_names

        self.file_handle = open(self.file_name, "w")
        self.writer = csv.writer(self.file_handle)

        self.write_header()
        print(f"{self.file_name} created successfully with header row")

    def write_header(self):
        """
        writes header into csv file
        """
        self.write_row(self.column_names)
        return

    def write_row(self, row):
        """
        writes a row into csv file
        """
        self.writer.writerow(row)
        return

    def close(self):
        """
        close the file
        """
        self.file_handle.close()
        return


def load_cmv_threads_data(ARGS):
    list_keys_required = ["author", "subreddit", "body"]
    list_author = []
    list_subreddit = []
    list_body = []

    csv_writer = CSVWriter("subreddit_threads.csv", list_keys_required)
    author, subreddit, body = (None, None, None)

    with open(ARGS.file_json, encoding="UTF-8") as file_handler:
        # print(f"num lines: {sum(1 for _ in file_handler)}")

        for line_number, line in enumerate((file_handler)):
            line_as_file = io.StringIO(line)
            json_parser = ijson.parse(line_as_file)
            for prefix, type, value in json_parser:
                if prefix == f"comments.item.{list_keys_required[0]}":
                    author = value
                elif prefix == f"comments.item.{list_keys_required[1]}":
                    subreddit = value
                elif prefix == f"comments.item.{list_keys_required[2]}":
                    body = value
                else:
                    pass

                if (author != "[deleted]") and (subreddit != "[deleted]") and (body != "[deleted]"):
                    csv_writer.write_row([author, subreddit, body])

    return

def main():
    file_json = "../../threads.jsonl"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--file_json", default=file_json,
        type=str, help="full path to the jsonl file")

    ARGS, unparsed = parser.parse_known_args()
    load_cmv_threads_data(ARGS)
    return

if __name__ == "__main__":
    main()
