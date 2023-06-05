import torch
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule
from transformers import (GenerationConfig, T5ForConditionalGeneration,
                          T5Tokenizer, LongT5ForConditionalGeneration, AutoTokenizer)


from create_csv_data import CSVWriter
from data_module import ChangeMyViewDataModule
from utils import convert_pred_string_labels_to_int_labels

class LightningT5(LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        num_classes: int = 20,
        gt_string_labels: list = [],
        long_T5: int = 0,
        author_dataset: str = "arguments_Amablue_body",
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_classes = num_classes
        self.gt_string_labels = gt_string_labels
        self.long_T5 = long_T5

        # Load model, generation config, and tokenizer
        self.model = self._get_model()
        self.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path)
        self.generation_config.max_new_tokens = 128
        self.tokenizer = self._get_tokenizer()
        self.dict_labels = {}
        self.author_dataset = author_dataset
        self.csv_writer = CSVWriter(self.author_dataset+".csv", ["body", "pred_int_labels"])

    def _get_model(self):
        if self.long_T5:
            model = LongT5ForConditionalGeneration.from_pretrained(
                self.model_name_or_path)
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                self.model_name_or_path)
        return model

    def _get_tokenizer(self):
        if self.long_T5:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        else:
            tokenizer = T5Tokenizer.from_pretrained(self.model_name_or_path)
        return tokenizer

    def forward(self, **inputs):
        return self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels'])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        all_predictions = None

        # Generate outputs given a batch
        generated_out = self.model.generate(
            inputs=batch['input_ids'],
            generation_config=self.generation_config)

        # Convert the input, generated, and reference tokens to text
        input_text = self.tokenizer.batch_decode(
            batch['input_ids'], skip_special_tokens=True)
        generated_text = self.tokenizer.batch_decode(
            generated_out, skip_special_tokens=True)

        #print(len(input_text))
        #print(input_text[0])
        #print(input_text[1])

        pred_int_labels = convert_pred_string_labels_to_int_labels(
            generated_text,
            self.gt_string_labels,
            delimiter=','
        )

        for label_index in range(len(pred_int_labels)):
            self.csv_writer.write_row([input_text[label_index], pred_int_labels[label_index]])
        #print(len(pred_int_labels))
        #print(pred_int_labels[0])
        #print(pred_int_labels[1])

        pred_int_labels = np.sum(pred_int_labels, axis=0)
        for label_index in range(len(pred_int_labels)):
            if label_index not in self.dict_labels:
                self.dict_labels[f"{label_index}"] = pred_int_labels[label_index]
            else:
                temp = self.dict_labels[f"{label_index}"]
                temp += pred_int_labels[label_index]
                self.dict_labels[f"{label_index}"] = temp

        self.log_dict(self.dict_labels, reduce_fx="sum")

        return all_predictions


if __name__ == '__main__':
    df_train = pd.read_csv("../task_1/dataset/labels-training.tsv", sep="\t")
    list_true_labels = list(df_train.columns)[1:]
    model = LightningT5(gt_string_labels=list_true_labels)

    data_module = ChangeMyViewDataModule('author_dataset_orig/arguments_Amablue')
    test = data_module.test_dataloader()
    # Prints the first batch of the training set
    batch = next(iter(test))
    print(batch)
    #model.validation_step(batch, 0)
