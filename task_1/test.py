import os
import torch
import argparse
import pandas as pd

from data_module import Touche23DataModule
from lightning_T5 import LightningT5
from pytorch_lightning import Trainer


def get_test_args():
    results_save_file = 'results_task1/flan-t5-base-few-shot'
    # results_save_file = 'results_task1/long_t5_base_augmented_single_shot'

    model_type = 'google/flan-t5-base'
    # model_type = 'google/long-t5-local-base'
    longT5_mode = 0
    dataset_path = "datasets/touche23_few_shot_prompt"
    file_path_model = "artifacts/T5_base_few_shot_0604-15:26:47/f1=0.79.ckpt"
    # file_path_model = "artifacts/T5_base_single_shot_0604-12:44:31/model.ckpt"
    file_labels_tsv = "task_1/dataset/labels-training.tsv"
    force_cpu = 1
    limit_test_batches = 1.0
    eval_batch_size = 32

    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--results_save_file", type=str, default=results_save_file,
                        help="File path of the results summary")
    parser.add_argument("--model_type", type=str, default=model_type,
                        help="Wether to load T5 or longT5 tokenizer.")
    parser.add_argument("--dataset_path", type=str, default=dataset_path,
                        help="The path to the dataset.")
    parser.add_argument("--file_labels_tsv", type=str, default=file_labels_tsv,
                        help="Full path to labels tsv file")
    parser.add_argument("--file_path_model", type=str, default=file_path_model,
                        help="Full path to trained model file")
    parser.add_argument("--num_classes", type=int, default=20,
                        help="The number of output classes.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers to use for data loading")
    parser.add_argument("--eval_batch_size", type=int, default=eval_batch_size,
                        help="batch size to use for testing")
    parser.add_argument("--limit_test_batches", type=float, default=limit_test_batches,
                        help="fraction of test batches to be used")
    parser.add_argument('--longT5_mode', type=int, default=longT5_mode,
                        help='Whether to use longT5')
    parser.add_argument('--force_cpu', type=int, default=force_cpu,
                        help='Forcing to use cpu when training when set to 1.')

    return parser.parse_args()


def test():
    test_args = get_test_args()
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    accelerator = "cpu" if test_args.force_cpu else "auto"

    df_train = pd.read_csv(test_args.file_labels_tsv, sep="\t")
    list_true_labels = list(df_train.columns)[1:]

    data_module = Touche23DataModule(dataset_path=test_args.dataset_path,
                                     eval_batch_size=test_args.eval_batch_size,
                                     num_workers=test_args.num_workers,
                                     long_T5=test_args.longT5_mode)
    data_module.report()

    model = LightningT5.load_from_checkpoint(test_args.file_path_model)

    trainer = Trainer(
        accelerator=accelerator,
        devices="auto",
        limit_test_batches=test_args.limit_test_batches,
    )

    test_results = trainer.test(model=model, dataloaders=data_module)

    with open(test_args.results_save_file + '.txt', 'w') as f:
        f.write('Test results:\n')
        for key, value in test_results[0].items():
            f.write('%s:%s\n' % (key, value))

        f.write('\n')
        f.write('Test params:\n')

        for key, value in vars(test_args).items():
            f.write('%s:%s\n' % (key, value))

        return


if __name__ == "__main__":
    test()
