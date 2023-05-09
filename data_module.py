from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset, load_from_disk


class Touche23DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str = None,
        train_batch_size: int = 16,
        eval_batch_size: int = 64,
        max_source_length: int = 512,
        max_target_length: int = 128,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.dataset_path = dataset_path

        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, padding=True, max_length=self.max_source_length, label_pad_token_id=-100)

        self.save_hyperparameters(ignore=['dataset_path'])

        self.datasets = self._load_dataset()

    def _load_dataset(self, stage: str = None) -> None:
        return load_from_disk(self.dataset_name_or_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['train'],
            shuffle=True,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['validation'],
            batch_size=self.eval_batch_size,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['test'],
            batch_size=self.eval_batch_size,
            collate_fn=self.data_collator,
        )

    def _load_and_process():
        datasets = load_dataset('webis/Touche23-ValueEval')

        for split in datasets.keys():
            datasets[split] = datasets[split].map(
                self._preprocess_function,
                batched=True,
            )

        datasets.set_format(type='torch')
        return datasets

    def _load_dataset(self):

        if self.dataset_name_or_path == 'webis/Touche23-ValueEval':
            datasets = self.load_and_process()
        else:
            try:
                datasets = load_from_disk(self.dataset_name_or_path)
            except FileNotFoundError:
                print('Incorrect path. Dataset will be downloaded, processed, and saved to: ' +
                      self.dataset_name_or_path)
                datasets = self._load_and_process()
                datasets.save_to_disk(self.dataset_name_or_path)

        return datasets

    def _preprocess_function(self, examples, training=True):
        # Create input text by combining premise and hypothesis
        input_text = [
            f"premise: {premise}\n" f"hypothesis: {hypothesis}"
            for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])
        ]

        # Tokenize input text
        model_inputs = self.tokenizer(
            examples['input_text'], truncation=True, max_length=self.max_source_length)

        # If we request the int labels for the classification task
        if self.classify:
            model_inputs["int_labels"] = examples["label"]
            return model_inputs

        # Tokenize first explanation and add as "labels" to model inputs
        targets = self.tokenizer(
            examples['explanation_1'], truncation=True, max_length=self.max_target_length)

        model_inputs["labels"] = targets["input_ids"]

        # Tokenize all explanations and assign to explanation_i
        if not training:
            for i in range(1, 4):
                key_explanation = f"explanation_{i}"
                targets = self.tokenizer(
                    examples[key_explanation], truncation=True, padding='max_length', max_length=self.max_target_length)
                model_inputs[key_explanation] = targets["input_ids"]
                # Note that these are zero padded and not -100 padded

        return model_inputs
