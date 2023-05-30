from transformers import T5Tokenizer
from datasets import load_dataset, load_from_disk


class DataPreprocessor():

    def __init__(self,
                 dataset_load_path: str = 'datasets/touche23_prompt',
                 dataset_save_path: str = 'datasets/touche23_prompt_V1',
                 model_checkpoint: str = 'google/flan-t5-small',
                 max_source_length: int = 512,
                 max_target_length: int = 128,) -> None:

        self.dataset_load_path = dataset_load_path
        self.dataset_save_path = dataset_save_path
        self.tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def set_dummy_dataset(self):
        self.datasets = load_from_disk('datasets/touche23/dummy')

    def tokenize_dataset(self):
        raw_dataset_columns = self.datasets['training'].column_names

        for split in self.datasets.keys():
            self.datasets[split] = self.datasets[split].map(
                self._tokenize,
                batched=True,
                remove_columns=raw_dataset_columns,
            )
        self.datasets.set_format(type='torch')

    def _tokenize(self, examples):
        model_inputs = self.tokenizer(
            examples['Premise'], truncation=True, max_length=self.max_source_length)

        targets = self.tokenizer(
            examples['Stance'], truncation=True, max_length=self.max_target_length)

        model_inputs["labels"] = targets["input_ids"]
        return model_inputs

    def save_dataset(self):
        self.datasets.save_to_disk(self.dataset_load_path)


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.set_dummy_dataset()
    preprocessor.tokenize_dataset()
    preprocessor.save_dataset()
