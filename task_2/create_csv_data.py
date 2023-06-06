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
    list_keys_required = ["author", "score", "delta", "body",]
    num_keys = len(list_keys_required)

    csv_writer = CSVWriter("subreddit_threads.csv", list_keys_required)
    author, body, score, delta = (None, None, None, None)
    counter = 0

    with open(ARGS.file_json, encoding="UTF-8") as file_handler:
        # print(f"num lines: {sum(1 for _ in file_handler)}")

        for line_number, line in enumerate((file_handler)):
            line_as_file = io.StringIO(line)
            json_parser = ijson.parse(line_as_file)
            #print(json_parser)
            #print("="*100)
            #print("\n\n\n\n")
            #print("="*100)
            for prefix, type, value in json_parser:
                #print("prefix:", prefix , "type:", type, "value:", value)
                if prefix == f"comments.item.{list_keys_required[3]}":
                    body = value
                    #print(f"{counter} body: {body}")
                    counter += 1
                elif prefix == f"comments.item.{list_keys_required[1]}":
                    score = value
                    #print(f"{counter} score: {score}")
                    counter += 1
                elif prefix == "delta":
                    delta = value
                    counter += 1
                    #print(f"{counter} delta: {delta}")
                elif prefix == f"comments.item.{list_keys_required[0]}":
                    author = value
                    #print(f"{counter} author: {author}")
                    counter += 1
                else:
                    pass

                if (counter % num_keys) == 0:
                    if (author != "[deleted]") and (score != "[deleted]") and (body != "[deleted]"):
                        csv_writer.write_row([author, score, delta, body])

            """
            if line_number == 2:
                break
            """

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
