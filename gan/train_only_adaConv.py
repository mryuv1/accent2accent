import argparse
from argparse import ArgumentParser
from pathlib import Path
import sys
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import os
import wandb
sys.path.append('..//lib_')
#from lib_.lightning.datamodule import DataModule
from lib_.lightning.lightningmodel import LightningModel
from lib_.DataLoader import AccentHuggingBasedDataLoader
import pickle
PC = "Macbook"

if PC=="Macbook":
    print("MPS - TZLIL WORKING")
    import subprocess
    # Set the environment variable
  #  os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1"

    # Execute your command
   # subprocess.run(["./webui.sh", "--no-half"])

class TensorBoardImageLogger(TensorBoardLogger):
    """
    Wrapper for TensorBoardLogger which logs images to disk,
        instead of the TensorBoard log file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        exp = self.experiment

        # if not hasattr(exp, 'add_image'):
        exp.add_image = self.add_image

    def add_image(self, tag, img_tensor, global_step):
        dir = Path(self.log_dir, 'images')
        dir.mkdir(parents=True, exist_ok=True)

        file = dir.joinpath(f'{tag}_{global_step:09}.pickle')

        # Serialize and save img_tensor using pickle
        with open(file, 'wb') as f:
            pickle.dump(img_tensor, f)


def parse_args():
    # Init parser
    parser = ArgumentParser()
    parser.add_argument('--iterations', type=int, default=160_000,
                        help='The number of training iterations.')
    parser.add_argument('--log-dir', type=str, default='./',
                        help='The directory where the logs are saved to.')
    parser.add_argument('--checkpoint', type=str,
                        help='Resume training from a checkpoint file.')
    parser.add_argument('--val-interval', type=int, default=1000,
                        help='How often a validation step is performed. '
                             'Applies the model to several fixed images and calculate the loss.')

    parser = AccentHuggingBasedDataLoader.add_argparse_args(parser)
    parser = LightningModel.add_argparse_args(parser)

    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    return vars(parser.parse_args())


if __name__ == '__main__':
    wandb.init(project="AdaCONV")
    args = parse_args()
    print("The arguments are", args)
    if args['checkpoint'] is None:
        max_epochs = 1
        model = LightningModel(**args)
    else:
        # We need to increment the max_epoch variable, because PyTorch Lightning will
        #   resume training from the beginning of the next epoch if resuming from a mid-epoch checkpoint.
        max_epochs = torch.load(args['checkpoint'])['epoch'] + 1
        model = LightningModel.load_from_checkpoint(checkpoint_path=args['checkpoint'])
    datamodule = AccentHuggingBasedDataLoader(**args)
    logger = TensorBoardImageLogger(args['log_dir'], name='logs')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # Create save directory if it doesn't exist
    os.makedirs(args['save_dir'], exist_ok=True)
    wandb.watch(model)
    #Move model to GPU
    model.cuda()
    checkpoint_callback = ModelCheckpoint(dirpath=args['save_dir'], filename='{epoch}-{step}',save_top_k = 3, monitor="loss", mode="min", every_n_train_steps=30)
    trainer = pl.Trainer(max_epochs=args['epochs'], callbacks=[checkpoint_callback, lr_monitor], logger=logger, max_steps=args['iterations'])


    trainer.fit(model, datamodule=datamodule)
