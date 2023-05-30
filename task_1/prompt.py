import os
import pandas as pd
from datasets import Dataset, DatasetDict
import random

LABELS = ['Self-direction thought', 'Self-direction action', 'Stimulation', 'Hedonism', 'Achievement', 'Power dominance', 'Power resources', 'Face', 'Security personal', 'Security societal', 'Tradition', 'Conformity rules', 'Conformity interpersonal', 'Humility', 'Benevolence caring', 'Benevolence dependability', 'Universalism concern', 'Universalism nature', 'Universalism tolerance', 'Universalism objectivity']
PROMPT_FORMATS = ["The premise: '{}' is '{}'. The conclusion is '{}'. Value category: {}\n Question: Which value category does the argument belong to?\n",
                  "Premise: {}\nStance: {}\nConclusion: {}. Value category: {}\n Question: Which value category does the argument belong to?\n",
                  "Argument: {}. {}. {}. Value category: {}\n Question: Which value category does the argument belong to?\n"]


def label_to_vector(df):
    """Converts the labels to a vector"""
    label_names = df.iloc[:, 1:]
    return label_names.values.tolist()

def convert_binary_labels_to_string(df):
    label_names = df.columns[1:]
    string_labels = []

    for index, row in df.iterrows():
        binary_values = row.values[1:]
        string_labels.append([label_names[i] for i, value in enumerate(binary_values) if value == 1])

    return string_labels

def single_shot_prompt(df):
    """Creates a single shot prompt for each argument with the first prompt format"""
    
    template = PROMPT_FORMATS[0] # use the first template 
    prompts = [
                template.format(row['Premise'], row['Stance'], row['Conclusion'], ', '.join(LABELS))
                for _, row in df.iterrows()
    ]
    df['single_shot_prompt'] = prompts
    return df

def few_shot_prompt(df, num_shots=1, prompt_format=0, random_seed=46):
    """Creates a few shot prompt for each argument"""

    prompt_format = PROMPT_FORMATS[prompt_format]
    
    selected_arguments = df.sample(n=num_shots, random_state=random_seed)
    few_shot_prompts = [
        prompt_format.format(row['Premise'], row['Stance'], row['Conclusion'], ', '.join(LABELS)) + f"Answer: {random.choice(LABELS)}\n"
        for _, row in selected_arguments.iterrows()
    ]
    df['few_shot_prompt'] = df.apply(lambda row: ''.join(few_shot_prompts) + prompt_format.format(row['Premise'], row['Stance'], row['Conclusion'], ', '.join(LABELS)) + f"Answer: \n", axis=1)
    return df


def main():
    data_dir = 'dataset'
    export_path = '../datasets/'

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

    arguments = []
    labels = []

    # Load argument data
    for file in arguments_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            arguments.append(pd.read_csv(file_path, encoding='utf-8', sep='\t', header=0))
        else:
            print(f"File not found: {file_path}")

    # Load label data
    for file in labels_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            labels.append(pd.read_csv(file_path, encoding='utf-8', sep='\t', header=0))
        else:
            print(f"File not found: {file_path}")

    # Preprocess argument data
    out = []
    for df, label in zip(arguments, labels):
        df['label_vector'] = label_to_vector(label)
        df['label_string'] = convert_binary_labels_to_string(label)
        df = single_shot_prompt(df)
        df = few_shot_prompt(df)
        out.append(df)

    # Create a DatasetDict object
    dataset_dict = DatasetDict()

    # Add the individual datasets to the DatasetDict with their respective names
    dataset_dict["train"] = Dataset.from_pandas(out[0])
    dataset_dict["validation"] = Dataset.from_pandas(out[1])
    dataset_dict["validation_zhihu"] = Dataset.from_pandas(out[2])
    dataset_dict["test"] = Dataset.from_pandas(out[3])

    dataset_dict.save_to_disk(os.path.join(export_path, 'touche23_prompt'))
    print(f"Dataset succesfully saved to {export_path} as touche23_prompt")
    
if __name__ == '__main__':
    main()