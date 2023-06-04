import os
from datetime import datetime

import pandas as pd
from data_module import Touche23DataModule
from lightning_T5 import LightningT5
from lightning_GPTNeo import LightningGPTNeo
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from train_params import get_args_parser

time = datetime.now().strftime("%m%d-%H:%M:%S")


def get_logger(args):
    name = args.run_name + '_' + args.prompt_mode + '_' + time
    logger = WandbLogger(name=name,
                         project='Touche23',
                         save_dir="logs/",
                         log_model=True,
                         entity=args.wandb_entity,
                         config=args)
    return logger


def get_callbacks(args):
    checkpoint = ModelCheckpoint(
        monitor='val/f1',
        mode='max',
        dirpath=args.checkpoint_save_path,
        filename=time + '-' + 'touche23-{epoch:02d}-{val/f1:.2f}',
        every_n_train_steps=args.val_check_interval,
        save_top_k=3,
    )
    return [checkpoint]


def train():
    args = get_args_parser()

    print(args)

    if args.checkpoint_save_path:
        args.checkpoint_save_path = os.path.expanduser(
            args.checkpoint_save_path)

    df_train = pd.read_csv("task_1/dataset/labels-training.tsv", sep='\t')
    list_true_labels = list(df_train.columns)[1:]

    data_module = Touche23DataModule(dataset_path=args.data_path,
                                     train_batch_size=args.train_batch_size,
                                     eval_batch_size=args.eval_batch_size,
                                     num_workers=args.num_workers,
                                     long_T5=args.longT5_mode)
    data_module.report()

    model = LightningT5(model_name_or_path=args.model,
                        num_classes=args.num_classes,
                        gt_string_labels=list_true_labels,
                        learning_rate=args.learning_rate,
                        long_T5=args.longT5_mode)

    callbacks = get_callbacks(args)
    logger = get_logger(args)

    accelerator = 'cpu' if args.force_cpu else 'auto'

    trainer = Trainer(
        accelerator=accelerator,
        devices='auto',
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        callbacks=callbacks,
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    train()
