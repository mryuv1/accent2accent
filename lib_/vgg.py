import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


import warnings

from torch import nn,save, load
from torchvision import models
from torchvision.transforms import transforms
import os
import torchvggish
import torchvggish

def LoadVGG19(path_to_weights=None,current_directory=None,TzlilSuccess=False,TzlilTrain=False,num_classes=2):
    # Load the VGGish model
    model = torchvggish.vggish()
    return model
  #  # Modify the first layer to accept 1 channel input
  #  vgg19.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

    # Modify the fully connected layers
    num_features = vgg19.classifier[0].in_features
    if TzlilTrain:
        vgg19.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),  # Change num_classes to the number of output classes
        )


    # Load weights if provided
    if TzlilSuccess:
        state_dict = load(os.path.join(current_directory, "vgg19_new.pth"))
        # Remove 'vgg19.' from keys in state_dict to match the model structure
        state_dict = {k.replace('vgg19.', ''): v for k, v in state_dict.items()}
        vgg19.load_state_dict(state_dict)
        #Save the weights for backup
        save(vgg19.state_dict(), os.path.join(current_directory, "vgg19_new_bckp.pth"))

   # vgg19.load_state_dict(torch.load(os.path.join(current_directory, "vgg19_new.pth")))

    return vgg19

class VGGEncoder(nn.Module):
    def __init__(self, path_to_weights=None,current_directory=None,normalize=True, post_activation=True, TzlilSucces=False, TzlilTrain=False, num_classes=2):
        super().__init__()
        if current_directory is None:
            current_directory = os.path.dirname(__file__)
        if path_to_weights is None:
            path_to_weights = os.path.join(current_directory, "vgg.pth")
        #self.vgg19 = LoadVGG19(path_to_weights,current_directory, TzlilSucces, TzlilTrain, num_classes)
        self.vgg19 = torchvggish.vggish()
       # self.vgg19 = models.vgg19(pretrained=True)
       # self.vgg19.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.TzlilTrain = TzlilTrain
        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).mean()
            std = torch.tensor([0.229, 0.224, 0.225]).mean()
            self.normalize = transforms.Normalize(mean=mean, std=std)
        else:
            self.normalize = nn.Identity()

        if post_activation:
            layer_names = {'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'}
        else:
            layer_names = {'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'}
        blocks, block_names, scale_factor, out_channels = extract_vgg_blocks(self.vgg19.features,
                                                                             layer_names)



        self.blocks = nn.ModuleList(blocks)
        self.block_names = block_names
        self.scale_factor = scale_factor
        self.out_channels = out_channels

    def forward(self, xs):
        #remove the first channel
        xs = xs.squeeze(0)
        xs = self.normalize(xs)
        xs.to(torch.float32)
        features = []
        for block in self.blocks:
            xs = block(xs)
            features.append(xs)

        return features

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False


# For AdaIn, not used in AdaConv.
class VGGDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [
            self._conv(512, 256),
            nn.ReLU(),
            self._upsample(),

            self._conv(256, 256),
            nn.ReLU(),
            self._conv(256, 256),
            nn.ReLU(),
            self._conv(256, 256),
            nn.ReLU(),
            self._conv(256, 128),
            nn.ReLU(),
            self._upsample(),

            self._conv(128, 128),
            nn.ReLU(),
            self._conv(128, 64),
            nn.ReLU(),
            self._upsample(),

            self._conv(64, 64),
            nn.ReLU(),
            self._conv(64, 1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, content):
        ys = self.layers(content)
        return ys

    @staticmethod
    def _conv(in_channels, out_channels, kernel_size=3, padding_mode='reflect'):
        padding = (kernel_size - 1) // 2
        return nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         padding=padding,
                         padding_mode=padding_mode)

    @staticmethod
    def _upsample(scale_factor=2, mode='nearest'):
        return nn.Upsample(scale_factor=scale_factor, mode=mode)


def extract_vgg_blocks(layers, layer_names):
    blocks, current_block, block_names = [], [], []
    scale_factor, out_channels = -1, -1
    depth_idx, relu_idx, conv_idx = 1, 1, 1
    for layer in layers:
        name = ''
        if isinstance(layer, nn.Conv2d):
            name = f'conv{depth_idx}_{conv_idx}'
            current_out_channels = layer.out_channels
            layer.padding_mode = 'reflect'
            conv_idx += 1
        elif isinstance(layer, nn.ReLU):
            name = f'relu{depth_idx}_{relu_idx}'
            layer = nn.ReLU(inplace=False)
            relu_idx += 1
        elif isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.MaxPool2d):
            name = f'pool{depth_idx}'
            depth_idx += 1
            conv_idx = 1
            relu_idx = 1
        else:
            warnings.warn(f' Unexpected layer type: {type(layer)}')

        current_block.append(layer)
        if name in layer_names:
            blocks.append(nn.Sequential(*current_block))
            block_names.append(name)
            scale_factor = 1 * 2 ** (depth_idx - 1)
            out_channels = current_out_channels
            current_block = []

    return blocks, block_names, scale_factor, out_channels






