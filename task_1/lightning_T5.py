import numpy as np
from evaluate import convert_pred_string_labels_to_int_labels
from pytorch_lightning import LightningModule
from torch.optim import AdamW
import torch
from torchmetrics.classification import (MultilabelF1Score,
                                         MultilabelPrecision, MultilabelRecall)
from transformers import (GenerationConfig, T5ForConditionalGeneration,
                          T5Tokenizer, LongT5ForConditionalGeneration)
from data_module import Touche23DataModule
import pandas as pd


class LightningT5(LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        num_classes: int = 20,
        gt_string_labels: list = [],
        learning_rate: float = 1e-4,
        long_T5: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_classes = num_classes
        self.gt_string_labels = gt_string_labels
        self.learning_rate = learning_rate
        self.long_T5 = long_T5

        # Load model, generation config, and tokenizer
        self.model = self._get_model()
        self.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path)
        self.generation_config.max_new_tokens = 128
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            model_name_or_path)

        # Instanciate metrics
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

        self.save_hyperparameters()

    def _get_model(self):
        if self.long_T5:
            model = LongT5ForConditionalGeneration.from_pretrained(
                self.model_name_or_path)
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                self.model_name_or_path)
        return model

    def forward(self, **inputs):
        return self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels'])

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

        # And compute metrics to log
        pred_int_labels = convert_pred_string_labels_to_int_labels(
            generated_text,
            self.gt_string_labels,
            delimiter=','
        )
        pred_int_labels = torch.from_numpy(pred_int_labels)
        pred_int_labels = pred_int_labels.to(device=self.device.type)
        target_int_labels = batch['int_vec_labels']

        self.f1_score.update(pred_int_labels, target_int_labels)
        self.precision_score.update(pred_int_labels, target_int_labels)
        self.recall_score.update(pred_int_labels, target_int_labels)

        # Also pass input to the model to compute loss
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])

        val_loss = outputs.loss

        self.log_dict({'val/loss': val_loss,
                       'val/f1': self.f1_score,
                       'val/precision': self.precision_score,
                       'val/recall': self.recall_score, },
                      prog_bar=True)

        return {'val_loss': val_loss,
                'input_text': input_text,
                'generated_text': generated_text, }

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        return optimizer


if __name__ == '__main__':

    df_train = pd.read_csv("task_1/dataset/labels-training.tsv", sep="\t")
    list_true_labels = list(df_train.columns)[1:]
    model = LightningT5(gt_string_labels=list_true_labels)

    data_module = Touche23DataModule('datasets/touche23_single_shot_prompt')
    val = data_module.val_dataloader()
    # Prints the first batch of the training set
    batch = next(iter(val))
    model.validation_step(batch, 0)
