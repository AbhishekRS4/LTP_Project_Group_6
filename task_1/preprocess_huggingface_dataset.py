from transformers import T5Tokenizer, AutoTokenizer
from datasets import load_dataset, load_from_disk, concatenate_datasets

SEED = 42


class DataPreprocessor():

    def __init__(self,
                 dataset_load_path: str = 'datasets/touche23_prompt',
                 dataset_save_path: str = '../datasets/touche23',
                 model_checkpoint: str = 'google/flan-t5-small',
                 input_col_name: str = 'single_shot_prompt',
                 max_source_length: int = 512,
                 max_target_length: int = 128,
                 long_T5: int = 0,) -> None:

        self.dataset_load_path = dataset_load_path
        self.dataset_save_path = dataset_save_path + '_' + input_col_name
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
        raw_dataset_columns = self.datasets['train'].column_names

        for split in self.datasets.keys():
            self.datasets[split] = self.datasets[split].map(
                self._tokenize,
                batched=True,
                remove_columns=raw_dataset_columns,
            )
        self.datasets.set_format(type='torch')

    def _tokenize(self, examples):
        model_inputs = self.tokenizer(
            examples[self.input_col_name], truncation=True, max_length=self.max_source_length)

        targets = self.tokenizer(
            examples['label_string'], truncation=True, max_length=self.max_target_length)

        model_inputs["labels"] = targets["input_ids"]
        model_inputs["int_vec_labels"] = examples["label_vector"]
        return model_inputs

    def save_dataset(self):
        self.datasets.save_to_disk(self.dataset_save_path)

    def report(self):
        for split in self.datasets.keys():
            print(split + ' data:')
            print(self.datasets[split])

    def combine_augmented_datasets(self, proportion_data_to_keep=0.20):
        lm = self.datasets['augmented_lm'].shuffle(seed=SEED)
        noise = self.datasets['augmented_noise'].shuffle(seed=SEED)
        thesaurus = self.datasets['augmented_thesaurus'].shuffle(seed=SEED)

        lm = lm.remove_columns('Unnamed: 0')
        noise = noise.remove_columns('Unnamed: 0')
        thesaurus = thesaurus.remove_columns('Unnamed: 0')

        lm = lm.train_test_split(train_size=proportion_data_to_keep)
        noise = noise.train_test_split(train_size=proportion_data_to_keep)
        thesaurus = thesaurus.train_test_split(train_size=proportion_data_to_keep)
        # print(lm['train'])
        # print(noise['train'])
        # print(thesaurus['train'])

        train_data = concatenate_datasets([self.datasets['train'], lm['train'], noise['train'], thesaurus['train']])
        self.datasets['train'] = train_data
        self.datasets.pop('augmented_lm')
        self.datasets.pop('augmented_noise')
        self.datasets.pop('augmented_thesaurus')


def preprocess_dataset():
    # model_checkpoint = "google/flan-t5-base"
    model_checkpoint = "google/long-t5-local-base"
    dataset_save_path = 'datasets/touche23_long'
    max_source_length = 2048
    # max_source_length = 2048
    # input_col_name = 'single_shot_prompt'
    input_col_name = 'few_shot_prompt'
    preprocessor = DataPreprocessor(dataset_save_path=dataset_save_path,
                                    model_checkpoint=model_checkpoint,
                                    max_source_length=max_source_length,
                                    input_col_name=input_col_name
                                    )
    preprocessor.load_dataset()
    preprocessor.tokenize_dataset()
    preprocessor.save_dataset()


def preprocess_augmented_dataset(long_T5=False, single_shot=True):
    dataset_load_path = 'datasets/touche23_prompt_aug'

    if long_T5:
        model_checkpoint = "google/long-t5-local-base"
        dataset_save_path = 'datasets/touche23_prompt_aug_long'
        max_source_length = 2048
    else:
        model_checkpoint = "google/flan-t5-base"
        dataset_save_path = 'datasets/touche23_prompt_aug_large'
        max_source_length = 512

    if single_shot:
        input_col_name = 'single_shot_prompt'
    else:
        input_col_name = 'few_shot_prompt'

    preprocessor = DataPreprocessor(dataset_load_path=dataset_load_path,
                                    dataset_save_path=dataset_save_path,
                                    model_checkpoint=model_checkpoint,
                                    max_source_length=max_source_length,
                                    input_col_name=input_col_name
                                    )
    preprocessor.load_dataset()
    print('BEFORE')
    preprocessor.report()
    preprocessor.combine_augmented_datasets()
    preprocessor.tokenize_dataset()

    print('AFTER:')
    preprocessor.report()
    preprocessor.save_dataset()


if __name__ == "__main__":
    preprocess_augmented_dataset(long_T5=True, single_shot=False)
