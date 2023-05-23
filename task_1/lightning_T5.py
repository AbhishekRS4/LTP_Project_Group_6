import numpy as np
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import GenerationConfig, T5ForConditionalGeneration, T5Tokenizer
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall


NUM_CLASSES = 53


class LightningT5(LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        num_classes: int = 53,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = "google/flan-t5-small"

        # Load model, generation config, and tokenizer
        self.model = self._get_model()
        self.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path)
        self.generation_config.max_new_tokens = 128
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            model_name_or_path)

        # Instanciate metrics
        self.num_classes = num_classes
        self.f1_score = MultilabelF1Score(
            num_labels=self.num_classes,
            average='macro',)
        self.precision_score = MultilabelPrecision(
            num_labels=self.num_classes,
            average='macro',)
        self.recall_score = MultilabelRecall(
            num_labels=self.num_classes,
            average='macro',)
        # List to keep track of training loss
        self.train_loss_history = []

    def _get_model(self):
        model = T5ForConditionalGeneration.from_pretrained(
            self.model_name_or_path)
        return model

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # Pass the batch to the model
        outputs = self(**batch)
        loss = outputs.loss

        # Log the loss
        self.log('train/loss_epoch', loss, on_step=False, on_epoch=True)
        self.log('train/loss_step', loss, on_step=True,
                 on_epoch=False, prog_bar=True)

        self.train_loss_history.append(loss.item())

        # Every log_every_n_steps compute running loss and log
        if self.global_step % self.trainer.log_every_n_steps == 0 and self.global_step != 0:
            step_metrics = self.train_loss_history
            reduced = sum(step_metrics) / len(step_metrics)
            self.log('train/loss_step_reduced', reduced,
                     on_step=True, on_epoch=False, prog_bar=True)
            self.train_loss_history = []

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Generate outputs given a batch
        generated_out = self.model.generate(
            inputs=batch['input_ids'],
            generation_config=self.generation_config)

        # Convert the input, generated, and reference tokens to text
        input_text = self.tokenizer.batch_decode(
            batch['input_ids'], skip_special_tokens=True)
        generated_text = self.tokenizer.batch_decode(
            generated_out, skip_special_tokens=True)
        # reference_texts = self.tokenizer.batch_decode(
        #     batch['labels'], skip_special_tokens=True)

        # @TODO convert output text to a numerical array
        # And compute metrics to log

        # Also pass input to the model to compute loss
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])

        val_loss = outputs.loss

        self.log_dict({'val/loss': val_loss}, prog_bar=True)

        return {'val_loss': val_loss,
                'input_text': input_text,
                'generated_text': generated_text, }

    def configure_optimizers(self):
        # Might also add lr_scheduler
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer


if __name__ == '__main__':
    model = LightningT5()
