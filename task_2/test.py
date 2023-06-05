import os
import torch
import argparse
import pandas as pd

from pytorch_lightning import Trainer

from lightning_T5 import LightningT5
from data_module import ChangeMyViewDataModule


def test(ARGS):
    accelerator = "cpu"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    df_train = pd.read_csv(ARGS.file_labels_tsv, sep="\t")
    list_true_labels = list(df_train.columns)[1:]

    data_module = ChangeMyViewDataModule(dataset_path=ARGS.dataset_path,
                                         eval_batch_size=ARGS.batch_size,
                                         num_workers=ARGS.num_workers)

    model = LightningT5(model_name_or_path=ARGS.model,
                        num_classes=ARGS.num_classes,
                        gt_string_labels=list_true_labels,
                        author_dataset=ARGS.dataset_path.split("/")[1])

    trainer = Trainer(
        accelerator=accelerator,
        devices="auto",
        limit_test_batches=ARGS.limit_test_batches,
    )

    all_predictions = trainer.test(model=model, ckpt_path=ARGS.file_path_model, dataloaders=data_module)
    model.csv_writer.close()
    #print(all_predictions)
    return

def test_multi(ARGS):
    accelerator = "cpu"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    df_train = pd.read_csv(ARGS.file_labels_tsv, sep="\t")
    list_true_labels = list(df_train.columns)[1:]

    dir_datasets = "author_datasets_preprocessed_prompt"
    list_datasets = os.listdir(dir_datasets)

    for dataset_path in list_datasets:
        data_module = ChangeMyViewDataModule(dataset_path=os.path.join(dir_datasets, dataset_path),
                                             eval_batch_size=ARGS.batch_size,
                                             num_workers=ARGS.num_workers)

        model = LightningT5(model_name_or_path=ARGS.model,
                            num_classes=ARGS.num_classes,
                            gt_string_labels=list_true_labels,
                            author_dataset=dataset_path)

        trainer = Trainer(
            accelerator=accelerator,
            devices="auto",
            limit_test_batches=ARGS.limit_test_batches,
        )

        all_predictions = trainer.test(model=model, ckpt_path=ARGS.file_path_model, dataloaders=data_module)
        model.csv_writer.close()
    #print(all_predictions)
    return

def main():
    model = "google/flan-t5-base"
    dataset_path = "author_datasets_preprocessed_prompt/arguments_Amablue_body/"
    file_path_model = "/home/abhishek/Desktop/RUG/lang_tech_project/model_checkpoints/f1=0.76-v2.ckpt"
    file_labels_tsv = "../task_1/dataset/labels-training.tsv"

    parser = argparse.ArgumentParser(description="Training parameters")

    parser.add_argument("--model", type=str, default=model,
                        help="The name or path of the model to be trained.")
    parser.add_argument("--dataset_path", type=str, default=dataset_path,
                        help="The path to the dataset.")
    parser.add_argument("--file_labels_tsv", type=str, default=file_labels_tsv,
                        help="Full path to labels tsv file")
    parser.add_argument("--file_path_model", type=str, default=file_path_model,
                        help="Full path to trained model file")
    parser.add_argument("--num_classes", type=int, default=20,
                        help="The number of output classes.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers to use for data loading")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size to use for testing")
    parser.add_argument("--limit_test_batches", type=float, default=1.0,
                        help="fraction of test batches to be used")

    ARGS = parser.parse_args()
    test_multi(ARGS)
    return

if __name__ == "__main__":
    main()
