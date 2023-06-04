from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, T5Tokenizer, AutoTokenizer
from datasets import load_dataset, load_from_disk


class Touche23DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str = None,
        model_checkpoint: str = 'google/flan-t5-small',
        max_source_length: int = 512,
        train_batch_size: int = 16,
        eval_batch_size: int = 64,
        num_workers: int = 1,
        long_T5: int = 0,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.dataset_path = dataset_path

        self.model_checkpoint = model_checkpoint
        self.num_workers = num_workers
        self.long_T5 = long_T5

        tokenizer = self._get_tokenizer()
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100)

        self.save_hyperparameters(ignore=['dataset_path'])

        self.datasets = self._load_dataset()

    def _load_dataset(self) -> dict:
        return load_from_disk(self.dataset_path)

    def _get_tokenizer(self):
        if self.long_T5:
            tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        else:
            tokenizer = T5Tokenizer.from_pretrained(self.model_checkpoint)
        return tokenizer

    def report(self):
        print('Training data:')
        print(self.datasets['train'])
        print('Validation data:')
        print(self.datasets['validation'])
        print('Testing data:')
        print(self.datasets['test'])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['train'],
            shuffle=True,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['validation'],
            batch_size=self.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['test'],
            batch_size=self.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )


if __name__ == '__main__':
    data_module = Touche23DataModule('datasets/touche23/dummy-tokenized')
    train = data_module.train_dataloader()
    # Prints the first batch of the training set
    print(next(iter(train)))
