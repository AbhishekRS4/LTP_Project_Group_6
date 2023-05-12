import os
import argparse
import pandas as pd
from datasets import Dataset, concatenate_datasets, DatasetDict
import random

LABELS = ['Self-direction thought', 'Self-direction action', 'Stimulation', 'Hedonism', 'Achievement', 'Power dominance', 'Power resources', 'Face', 'Security personal', 'Security societal', 'Tradition', 'Conformity rules', 'Conformity interpersonal', 'Humility', 'Benevolence caring', 'Benevolence dependability', 'Universalism concern', 'Universalism nature', 'Universalism tolerance', 'Universalism objectivity']
PROMPT_FORMATS = ["The premise: '{}' is '{}'. The conclusion is '{}'. Value category: {}\n Question: Which value category does the argument belong to?\n",
                  "Premise: {}\nStance: {}\nConclusion: {}. Value category: {}\n Question: Which value category does the argument belong to?\n",
                  "Argument: {}. {}. {}. Value category: {}\n Question: Which value category does the argument belong to?\n"]


def convert_binary_labels_to_string(df):
    label_names = df.columns[1:]
    string_labels = []

    for index, row in df.iterrows():
        binary_values = row.values[1:]
        string_labels.append([label_names[i] for i, value in enumerate(binary_values) if value == 1])

    df['String Labels'] = string_labels
    return df

def ensemble_prompt(df):
    prompts = [
        [
            prompt.format(row['Premise'], row['Stance'], row['Conclusion'], ', '.join(LABELS))
            for prompt in PROMPT_FORMATS
        ]
        for _, row in df.iterrows()
    ]

    df['ensemble_prompt'] = prompts
    return df

def few_shot_prompt(df, num_shots=1, prompt_format=0, random_seed=46):
    """Creates a few shot prompt for each argument"""

    prompt_format = PROMPT_FORMATS[prompt_format]
    
    selected_arguments = df.sample(n=num_shots, random_state=random_seed)
    few_shot_prompts = [
        prompt_format.format(row['Premise'], row['Stance'], row['Conclusion'], ', '.join(LABELS)) + f"Answer: {random.choice(LABELS)}\n"
        for _, row in selected_arguments.iterrows()
    ]
    # prompts = [
    #     df.apply(lambda row: ''.join(few_shot_prompts) + prompt_format.format(row['Premise'], row['Stance'], row['Conclusion'], ', '.join(LABELS)) + f"Answer: \n", axis=1)
    # ]

    df['few_shot_prompt'] = df.apply(lambda row: ''.join(few_shot_prompts) + prompt_format.format(row['Premise'], row['Stance'], row['Conclusion'], ', '.join(LABELS)) + f"Answer: \n", axis=1)
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
    data_dir = 'dataset'
    export_path = 'dataset/processed'

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
        df = ensemble_prompt(df)
        df = few_shot_prompt(df)
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

    
    train_data = combined_dfs[0]
    val_data = combined_dfs[1]
    val_data_z = combined_dfs[2]
    test_data = combined_dfs[3]

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

    dataset_dict.save_to_disk(os.path.join(export_path, 'processed_dataset'))
    print(f"Dataset succesfully saved to {export_path} as processed_dataset")
    
if __name__ == '__main__':
    main()