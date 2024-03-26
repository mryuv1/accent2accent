import argparse
from argparse import ArgumentParser
from pathlib import Path
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append('lib_')
# from lib_.data_loader import DataModule
# from lib_.lightning.datamodule import DataModule
from ganlightningmodel import GAN
import os
from DataLoader import AccentHuggingBasedDataLoader
import wandb
import pickle


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
    parser.add_argument('--iterations', type=int, default=250_000,
                        help='The number of training iterations.')
    parser.add_argument('--log-dir', type=str, default='./',
                        help='The directory where the logs are saved to.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume training from a checkpoint file.')
    parser.add_argument('--val-interval', type=int, default=1000,
                        help='How often a validation step is performed. '
                             'Applies the model to several fixed images and calculate the loss.')
    parser.add_argument('--workarea', type=str, default='/home/labs/training/class41/PycharmProjects/accent2accent', help="Work are for checkpoints")
    parser.add_argument('--prefix', type=str, default="prefix", help="Prefix for checkpoints saving")
    parser.add_argument('--step', type=str, default=3000, help="How often to save")

    parser = AccentHuggingBasedDataLoader.add_argparse_args(parser)
    parser = GAN.add_argparse_args(parser)

    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    return vars(parser.parse_args())

import os

def find_with_prefix(directory, prefix):
    matched_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                matched_files.append(os.path.join(root, file))
    matched_files.sort()
    if len(matched_files)>=2:
        return matched_files[-2]
    else:
        return matched_files[-1]




if __name__ == '__main__':
    os.environ['HF_HOME'] = "dataset"
    # set torch seed as 42
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(43)
    args = parse_args()

    # checkPoint = os.path.join("NewVGGWeights", "CHECKPOINT-step=3000.ckpt")

    if args['checkpoint'] is not None:
        checkPoint = os.path.join(args["workarea"], args["save_dir"], args['checkpoint'])
    else:
        checkPoint = None
    #Check if checkpoint exists in the path
    if os.path.exists(checkPoint):
       # checkPoint = find_with_prefix(os.path.join(args["workarea"],args["save_dir"]), args['checkpoint'])
        #:wcheckPoint = os.path.join(args["workarea"], checkPoint)
        print("Checkpoint exists", checkPoint)
    else:
        checkPoint = None
    if checkPoint is None:
        max_epochs = 1
        model = GAN(**args)
    else:
        # We need to increment the max_epoch variable, because PyTorch Lightning will
        #   resume training from the beginning of the next epoch if resuming from a mid-epoch checkpoint.
        max_epochs = torch.load(checkPoint)
        model = GAN.load_from_checkpoint(checkpoint_path=checkPoint)
    # TODO - MODIFY IT SO IT WILL TAKE MAYBE THE LAST CHECKPOINT, if not , enter as a flag python -checkpoint, give PATH to CHECKPOINT, it will resume from checkpoint
    wandb.init(project="AdaCONV", name=args['prefix'])
    logger = TensorBoardImageLogger(args['log_dir'], name='logs')
    # datamodule = DataModule(**args)
    datamodule = AccentHuggingBasedDataLoader(**args)
    os.makedirs(args['save_dir'], exist_ok=True)
    #    wandb.watch(model)
    args = parse_args()
    lr_monitor = LearningRateMonitor(logging_interval='step')
   # checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args["workarea"],args['save_dir']), filename=f'CHECKPOINT-{args["prefix"]}-{args["step"]}', save_top_k=4,
                                      #    monitor="TheShit", mode="min", every_n_train_steps=500)
    # checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args["workarea"],args['save_dir']), filename=f'CHECKPOINT-{args["prefix"]}', save_last=True,
    #                                       every_n_train_steps=1000)
    if checkPoint is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args["workarea"], args['save_dir']),
            filename=checkPoint.split("/")[-1].split(".")[0],
            save_last=True,
            every_n_epochs=1  # Save at the end of every epoch
        )
        print("HEY THERE IM LOADING THIS ", checkPoint.split("/")[-1])
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args["workarea"], args['save_dir']),
            filename=f'CHECKPOINT-{args["prefix"]}',

            save_last=True,
            every_n_epochs=1  # Save at the end of every epoch
        )

    wandb.watch(model)
    # Move model to cuda
    trainer = pl.Trainer(max_epochs=args['epochs'], callbacks=[checkpoint_callback, lr_monitor], logger=logger,
                         max_steps=args['iterations'], check_val_every_n_epoch=1)
    # accelerator="cpu")

    if torch.cuda.is_available():
        model = model.cuda()

    trainer.fit(model, datamodule=datamodule, ckpt_path=checkPoint)
    # trainer.save_checkpoint("./model.ckpt")
