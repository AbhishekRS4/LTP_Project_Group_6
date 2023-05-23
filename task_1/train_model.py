from datetime import datetime

from data_module import Touche23DataModule
from lightning_T5 import LightningT5
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

time = datetime.now().strftime("%m%d-%H:%M:%S")
checkpoint_save_path = '~/models/'
val_check_interval = 100
max_epochs = 100
log_every_n_steps = 100
limit_val_batches = 16
data_path = 'datasets/touche23/dummy-tokenized'
force_cpu = True


def get_logger():
    name = 'T5_Touche23_' + time
    logger = WandbLogger(name=name,
                         project='T5_Touche23',
                         save_dir="logs/",
                         log_model="all",)
    return logger


def get_callbacks():
    checkpoint = ModelCheckpoint(
        monitor='val/loss',
        dirpath=checkpoint_save_path,
        filename=time + 'touche23-{epoch:02d}-{val/loss:.2f}',
        every_n_train_steps=val_check_interval,
    )
    return [checkpoint]


def train():
    data_module = Touche23DataModule(dataset_path=data_path)
    model = LightningT5()
    callbacks = get_callbacks()
    logger = get_logger()

    accelerator = 'cpu' if force_cpu else 'auto'

    trainer = Trainer(
        accelerator=accelerator,
        devices='auto',
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        # Do validation every 50 steps
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        callbacks=callbacks,
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    train()
