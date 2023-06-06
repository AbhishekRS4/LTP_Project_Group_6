from datasets import load_dataset, Dataset, load_from_disk

def main():
    dataset_path = 'author_datasets_orig_prompt/arguments_DeltaBot/'
    dataset = load_from_disk(dataset_path)
    print(len(dataset['test']))
    print(dataset['test'][0].keys())
    print(dataset['test'][0])
    print(dataset['test'][1])
    print(dataset['test'][2])
    print(dataset['test'][3])
    print(list(dataset['test'][:].values())[1])
    print(list(dataset['test'][:].values())[2])
    return

main()
