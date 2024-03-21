from argparse import ArgumentParser
from math import sqrt
from statistics import mean
import gc
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import sys

sys.path.append('lib_')
import wandb
import torch
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
# from lib_.lightning.datamodule import DataModule
from adaconv.adaconv_model import AdaConvModel
from adain.adain_model import AdaINModel
from loss import MomentMatchingStyleLoss, GramStyleLoss, CMDStyleLoss, MSEContentLoss
import os
import wandb
from discriminator import Discriminator


class GAN(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Add params of other models
        parser = AdaConvModel.add_argparse_args(parser)
        parser.add_argument('--model-type', type=str, default='adaconv', choices=['adain', 'adaconv'])

        # Losses
        # mm = Moment Matching, gram = Gram matrix based, cmd = Central Moment Discrepancy
        parser.add_argument('--style-loss', type=str, default='mm', choices=['mm', 'gram', 'cmd'],
                            help="The style loss")
        parser.add_argument('--style-weight', type=float, default=0.3, help="The weight of the style loss")
        parser.add_argument('--content-loss', type=str, default='mse', choices=['mse'], help="The content loss")
        parser.add_argument('--content-weight', type=float, default=1.0, help="The weight of the content loss")
        parser.add_argument('--width-output', type=int, default=376, help="The width of the output image")
        parser.add_argument('--height-output', type=int, default=256, help="The height of the output image")
        parser.add_argument('--num-channels', type=int, default=1, help="The number of channels in the input image")
        parser.add_argument('--alpha', type=float, default=1.0, help="The alpha parameter for AdaIN")
        parser.add_argument('--fm-weight', type=float, default=1.0, help="The weight of the feature matching loss")
        parser.add_argument('--AdversionalLossWeight', type=float, default=0.5,
                            help="The weight of the adversarial loss")

        # Optimizer
        parser.add_argument('--lr', type=float, default=0.0001, help="The learning rate")
        parser.add_argument('--b1', type=float, default=0.5, help="The beta1 parameter for Adam")
        parser.add_argument('--b2', type=float, default=0.999, help="The beta2 parameter for Adam")
        return parser

    def __init__(self,
                 num_channels,
                 width_output,
                 height_output,
                 model_type,
                 alpha,
                 style_size, style_channels, kernel_size,
                 style_loss, style_weight,
                 content_loss, content_weight,
                 lr, b1, b2, prefix="prefix", PreTrainAdaConv=True,
                 **_):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        # networks
        data_shape = (num_channels, width_output, height_output)

        self.discriminator = Discriminator(input_shape=data_shape)
        self.train_cntr = 0

        self.prefix = prefix
        # Style loss
        if style_loss == 'mm':
            self.style_loss = MomentMatchingStyleLoss()
        elif style_loss == 'gram':
            self.style_loss = GramStyleLoss()
        elif style_loss == 'cmd':
            self.style_loss = CMDStyleLoss()
        else:
            raise ValueError('style_loss')

        # Content loss
        if content_loss == 'mse':
            self.content_loss = MSEContentLoss()
        else:
            raise ValueError('content_loss')

        # Content loss
        if content_loss == 'mse':
            self.content_loss = MSEContentLoss()
        else:
            raise ValueError('content_loss')

        # Model type
        if model_type == 'adain':
            self.generator = AdaINModel(alpha)
        elif model_type == 'adaconv':
            self.generator = AdaConvModel(style_size, style_channels, kernel_size)
        else:
            raise ValueError('model_type')

    #  self.init_weights()
    def forward(self, content, style):
        # Define the forward pass for the gan
        return self.generator(content, style)

    def init_weights(self):
        # List all the files in the directory
        files = os.listdir("NewVGGWeights")
        # Filter the files that are the weights of the generator
        generator_files = [file for file in files if "GeneratorWeights" in file]
        # Filter the files that are the weights of the discriminator
        discriminator_files = [file for file in files if "DiscriminatorWeights" in file]
        # Sort the files by the int after - in the file name
        generator_files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
        discriminator_files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
        # Load the weights
        if len(generator_files) > 0:
            self.generator.load_state_dict(torch.load(os.path.join("NewVGGWeights", generator_files[-1])))
        if len(discriminator_files) > 0:
            self.discriminator.load_state_dict(torch.load(os.path.join("NewVGGWeights", discriminator_files[-1])))
        print("The generator weights are", generator_files[-1], "The discriminator weights are",
              discriminator_files[-1])

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        inputs = batch["content"].squeeze(0)
        styles = batch["style"].squeeze(0)

        # Access optimizers
        optimizer_g, optimizer_d = self.optimizers()

        # Train the generator
        optimizer_g.zero_grad()
        self.generated_imgs, embeddings = self.generator(inputs, styles, return_embeddings=True)

        # print(f'generated images shape is: {self.generated_imgs.shape}')

        # Compute generator loss
        content_loss, style_loss = self.generator_loss(embeddings)
        self.log("Generator Style Loss", style_loss, prog_bar=True)
        self.log("Generator Content Loss", content_loss, prog_bar=True)
        g_loss = content_loss + style_loss
        wandb.log({"Content Loss": content_loss, "Style Loss": style_loss})

        if batch_idx % 20:
            log_input = inputs[0, 0, :, :]
            log_style = styles[0, 0, :, :]
            log_output = self.generated_imgs[0, 0, :, :]

            # Ensure all images have the same height
            min_height = min(log_input.shape[0], log_style.shape[0], log_output.shape[0])
            log_input = log_input[:min_height, :]
            log_style = log_style[:min_height, :]
            log_output = log_output[:min_height, :]
            #Normalize the log_output to the values of the log_input
            log_output = (log_output - log_output.min()) / (log_output.max() - log_output.min())
            log_input = (log_input - log_input.min()) / (log_input.max() - log_input.min())
            log_style = (log_style - log_style.min()) / (log_style.max() - log_style.min())
            # Concatenate images horizontally
            image_array = torch.cat((log_input, log_output, log_style), dim=1)

            images = wandb.Image(image_array, caption="Left: Input, Middle: Output, Right: Style")

            wandb.log({"examples": images})

        loss_of_folling_descriminator = self.hparams.AdversionalLossWeight * self.adversarial_loss(
            self.discriminator(self.generated_imgs.detach()), torch.ones(inputs.size(0), 1).type_as(inputs))
        # loss of the discriminator being fooled by the generated images
        if g_loss < 50:
            g_loss += loss_of_folling_descriminator
            print("g loss:", g_loss, "content loss:", content_loss,
                  "style loss:", style_loss, "Fooling discriminator:",
                  loss_of_folling_descriminator)
        else:
            print("g loss:", g_loss, "content loss:", content_loss,
                  "style loss:", style_loss, "Fooling discriminator:",
                  0)
        self.log("g_loss", g_loss, prog_bar=True)

        # Backward pass and optimization for generator
        # #NEED TO TEST THIS CONDITION TO PREVENT EXPLODING :!!!
        # if g_loss>1500:
        #     torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.0)
        self.manual_backward(g_loss)
        optimizer_g.step()
        # Check if generator loss is lower than 50 before updating discriminator
        if g_loss < 50:
            # Train the discriminator
            optimizer_d.zero_grad()
            valid = torch.ones(inputs.size(0), 1).type_as(inputs)
            real_loss = self.adversarial_loss(self.discriminator(styles), valid)

            fake = torch.zeros(inputs.size(0), 1).type_as(inputs)
            fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)
            wandb.log({"Fake Loss": fake_loss})
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            # Put both Discriminator and generator loss in one wandb graph and give them legend and title accordingly
            wandb.log({"Generator Loss": g_loss, "Discriminator Loss": d_loss})
            wandb.log({"Losses": {"Generator": g_loss, "Discriminator": d_loss}})

            # Backward pass and optimization for discriminator
            self.manual_backward(d_loss)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=4.0)
            optimizer_d.step()
            distance_between_g_to_d = torch.abs(d_loss - g_loss)
            self.log("TheShit", distance_between_g_to_d, prog_bar=True)
        else:
            print("g loss:", g_loss, "content loss:", content_loss,
                  "style loss:", style_loss, "Fooling discriminator:",
                  0)
            d_loss = 0
            self.log("g_loss", g_loss, prog_bar=True)
            self.log("TheShit", g_loss, prog_bar=True)
        return {"g_loss": g_loss, "d_loss": d_loss}

        # Backward pass and optimization for generator
        self.manual_backward(g_loss)
        optimizer_g.step()
        # Train the discriminator

        optimizer_d.zero_grad()
        # TODO - CHECK need to be done, added logits to the adversial, to prevent loss to explode.
        valid = torch.ones(inputs.size(0), 1).type_as(inputs)
        real_loss = self.adversarial_loss(self.discriminator(styles), valid)

        fake = torch.zeros(inputs.size(0), 1).type_as(inputs)
        fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)
        wandb.log({"Fake Loss": fake_loss})
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        # Put both Descriminator and generator loss in one wandb graph and give them legend and title accordinly
        wandb.log({"Generator Loss": g_loss, "Discriminator Loss": d_loss})
        wandb.log({"Losses": {"Generator": g_loss, "Discriminator": d_loss}})

        # Backward pass and optimization for discriminator
        self.manual_backward(d_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=4.0)
        optimizer_d.step()
        distance_between_g_to_d = torch.abs(d_loss - g_loss)
        self.log("TheShit", distance_between_g_to_d, prog_bar=True)
        return {"g_loss": g_loss, "d_loss": d_loss}

    def configure_optimizers(self):
        # Configure the optimizares
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def generator_loss(self, embeddings):
        # Content
        # content_loss = self.content_loss(embeddings['content'][-1], embeddings['output'][-1])

        content_loss = 0.0

        num_layers = len(embeddings['content'])

        # Define weights for each layer (you can adjust these)
        content_weights = [1.0 / num_layers] * num_layers

        for i, content_feature in enumerate(embeddings['content']):
            # Calculate content loss for each layer and multiply with weight
            content_loss += content_weights[i] * self.content_loss(content_feature, embeddings['output'][i])
        # Style
        style_loss = []
        for (style_features, output_features) in zip(embeddings['style'], embeddings['output']):
            style_loss.append(self.style_loss(style_features, output_features))
        style_loss = sum(style_loss) / len(style_loss)

        return self.hparams.content_weight * content_loss, self.hparams.style_weight * style_loss

    def on_train_batch_end(self, a=0, b=0, c=0, d=0, **_):
        # Check if cuda is available
        if torch.cuda.is_available():
            # Clear the cache
            torch.cuda.empty_cache()
            # Collect the garbage
            gc.collect()
            if self.global_step % 5000 == 0:
                # Save the weights for the generator and the discriminator
                current_dir = os.getcwd()
                torch.save(self.generator.state_dict(), os.path.join(current_dir, "NewVGGWeights",
                                                                     f'GeneratorWeights-{self.global_step}-{self.prefix}.pth'))
                torch.save(self.discriminator.state_dict(), os.path.join(current_dir, "NewVGGWeights",
                                                                         f'DiscriminatorWeights-{self.global_step}-{self.prefix}.pth'))
        else:
            if self.global_step % 300 == 0:
                # Save the weights for the generator and the discriminator
                current_dir = os.getcwd()
                torch.save(self.generator.state_dict(),
                           os.path.join(current_dir, "NewVGGWeights", f'GeneratorWeights-{self.global_step}.pth'))
                torch.save(self.discriminator.state_dict(),
                           os.path.join(current_dir, "NewVGGWeights", f'DiscriminatorWeights-{self.global_step}.pth'))

    def maintain_3_weights(self):
        # List all the files in the directory
        files = os.listdir("NewVGGWeights")
        # Filter the files that are the weights of the generator
        generator_files = [file for file in files if "GeneratorWeights" in file]
        # Filter the files that are the weights of the discriminator
        discriminator_files = [file for file in files if "DiscriminatorWeights" in file]
        # Sort the files by the int after - in the file name
        generator_files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
        discriminator_files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
        # Load the weights
        if len(generator_files) > 3:
            os.remove(os.path.join("NewVGGWeights", generator_files[0]))
        if len(discriminator_files) > 3:
            os.remove(os.path.join("NewVGGWeights", discriminator_files[0]))
        print("The generator weights are", generator_files[-1], "The discriminator weights are",
              discriminator_files[-1])
