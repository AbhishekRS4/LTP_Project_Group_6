import os
import argparse
import pandas as pd
from datasets import Dataset, concatenate_datasets, DatasetDict

def convert_binary_labels_to_string(df):
    label_names = df.columns[1:]
    string_labels = []

    for _, row in df.iterrows():
        binary_values = row.values[1:]
        string_labels.append([label_names[i] for i, value in enumerate(binary_values) if value == 1])

    df['String Labels'] = string_labels
    return df

def add_prompt_to_df(df):
    prompt_format = "Premise: {}\nStance: {}\nConclusion: {}\nWhich value category does it support?"
    preprocessed_arguments = []

    for _, row in df.iterrows():
        premise = row['Premise']
        stance = row['Stance']
        conclusion = row['Conclusion']
        prompt = prompt_format.format(premise, stance, conclusion)
        preprocessed_arguments.append(prompt)

    df['Prompt'] = preprocessed_arguments
    return df

def combine_columns(df_arguments, df_labels):
    """Combines the two DataFrames on column 'Argument ID'"""
    df_labels = df_labels[['Argument ID', 'String Labels']]
    df_labels.columns = ['Argument ID', 'Labels']
    return pd.merge(df_arguments, df_labels, on='Argument ID')

def labels_to_multi_choice(labels):
    """Converts the labels to a multi-choice format"""
    multi_choice_format = "{}: {}"
    multi_choice_options = []

    for index, label in enumerate(labels):
        multi_choice_option = multi_choice_format.format(chr(65 + index), label)
        multi_choice_options.append(multi_choice_option)

    return multi_choice_options

def main():
    parser = argparse.ArgumentParser(description="Data Loading and Preprocessing Script")

    parser.add_argument('--data-dir', type=str, help='Path to the directory containing the dataset')
    parser.add_argument('--export-path', type=str, help='Path to export the preprocessed data')

    args = parser.parse_args()

    data_dir = args.data_dir
    export_path = args.export_path

    arguments_files = [
        'arguments-training.tsv',
        'arguments-validation.tsv',
        'arguments-validation-zhihu.tsv',
        'arguments-test.tsv'
    ]

    labels_files = [
        'labels-training.tsv',
        'labels-validation.tsv',
        'labels-validation-zhihu.tsv',
        'labels-test.tsv'
    ]

    arguments_dfs = []
    labels_dfs = []

    # Load argument data
    for file in arguments_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            arguments_dfs.append(pd.read_csv(file_path, encoding='utf-8', sep='\t', header=0))
        else:
            print(f"File not found: {file_path}")

    # Load label data
    for file in labels_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            labels_dfs.append(pd.read_csv(file_path, encoding='utf-8', sep='\t', header=0))
        else:
            print(f"File not found: {file_path}")

    # Preprocess argument data
    preprocessed_arguments_dfs = []
    for df in arguments_dfs:
        df = add_prompt_to_df(df)
        preprocessed_arguments_dfs.append(df)

    # Preprocess label data
    preprocessed_labels_dfs = []
    for df in labels_dfs:
        df = convert_binary_labels_to_string(df)
        preprocessed_labels_dfs.append(df)

    # Combine argument and label data
    combined_dfs = []
    for i in range(len(preprocessed_arguments_dfs)):
        argument_df = preprocessed_arguments_dfs[i]
        label_df = preprocessed_labels_dfs[i]

        combined_df = combine_columns(argument_df, label_df)
        combined_dfs.append(combined_df)

    # Split data into train, validation, and test sets
    train_data = combined_dfs[0]
    val_data = combined_dfs[1]
    val_data_z = combined_dfs[2]
    test_data = combined_dfs[3]
    print(len(combined_dfs))
    # Convert the DataFrames to Hugging Face dataset format
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    val_dataset_z = Dataset.from_pandas(val_data_z)
    test_dataset = Dataset.from_pandas(test_data)

    # Create a DatasetDict object
    dataset_dict = DatasetDict()

    # Add the individual datasets to the DatasetDict with their respective names
    dataset_dict["train"] = train_dataset
    dataset_dict["validation"] = val_dataset
    dataset_dict["validation_zhihu"] = val_dataset_z
    dataset_dict["test"] = test_dataset

    # export the DatasetDict object in Hugging Face dataset format
    # combined_dataset = concatenate_datasets([dataset_dict["train"], dataset_dict["validation"],dataset_dict["validation_zhihu"], dataset_dict["test"]])

    dataset_dict.save_to_disk(os.path.join(export_path, 'dataset_dict'))
    
if __name__ == '__main__':
    main()