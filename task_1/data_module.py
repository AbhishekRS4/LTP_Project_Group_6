from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, T5Tokenizer
from datasets import load_dataset, load_from_disk


class Touche23DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str = None,
        model_checkpoint: str = 'google/flan-t5-small',
        max_source_length: int = 512,
        train_batch_size: int = 16,
        eval_batch_size: int = 64,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.dataset_path = dataset_path

        tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer, padding=True, max_length=max_source_length, label_pad_token_id=-100)

        self.save_hyperparameters(ignore=['dataset_path'])

        self.datasets = self._load_dataset()

    def _load_dataset(self) -> dict:
        return load_from_disk(self.dataset_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['training'],
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


if __name__ == '__main__':
    data_module = Touche23DataModule('datasets/touche23/dummy-tokenized')
    train = data_module.train_dataloader()
