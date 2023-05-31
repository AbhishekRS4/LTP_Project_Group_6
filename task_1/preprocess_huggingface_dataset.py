from transformers import T5Tokenizer
from datasets import load_dataset, load_from_disk


class DataPreprocessor():

    def __init__(self,
                 dataset_load_path: str = 'datasets/touche23_prompt',
                 dataset_save_path: str = '../datasets/touche23',
                 model_checkpoint: str = 'google/flan-t5-small',
                 input_col_name: str = 'single_shot_prompt',
                 max_source_length: int = 512,
                 max_target_length: int = 128,) -> None:

        self.dataset_load_path = dataset_load_path
        self.dataset_save_path = dataset_save_path + '_' + input_col_name
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


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.load_dataset()
    preprocessor.tokenize_dataset()
    print(preprocessor.datasets['train'])
    preprocessor.save_dataset()
