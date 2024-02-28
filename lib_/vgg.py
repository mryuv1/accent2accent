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

def LoadVGG19(path_to_weights=None,current_directory=None,TzlilSuccess=False,TzlilTrain=False,num_classes=2):
    vgg19 = models.vgg19(pretrained=False)
    #Load Weights from the pre-trained model
    if path_to_weights:
        vgg19.load_state_dict(load(path_to_weights))
    else:
        vgg19.load_state_dict(models.vgg19(pretrained=True).state_dict())
        #IF FAIL  do
        # vgg19.load_state_dict(load(os.path.join(current_directory, "vgg19.pth")))
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
        self.vgg19 = LoadVGG19(path_to_weights,current_directory, TzlilSucces, TzlilTrain, num_classes)
        self.TzlilTrain = TzlilTrain
        if TzlilTrain:
            print("TZLIL TRAIN MODE IS ON")
        else:
            if normalize:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                self.normalize = transforms.Normalize(mean=mean, std=std)
            else:
                self.normalize = nn.Identity()

            if post_activation:
                layer_names = {'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'}
            else:
                layer_names = {'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'}
            blocks, block_names, scale_factor, out_channels = extract_vgg_blocks(self.vgg19.features,
                                                                                 layer_names)
            #make sure each bias is double type
            # for block in blocks:
            #     for layer in block:
            #         if isinstance(layer, nn.Conv2d):
            #             layer.bias = nn.Parameter(layer.bias.float())


            self.blocks = nn.ModuleList(blocks)
            self.block_names = block_names
            self.scale_factor = scale_factor
            self.out_channels = out_channels

    def forward(self, xs):
        if self.TzlilTrain:
            return self.vgg19(xs.float())
        else:
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












#IF old training method on than :
if 1==0:
    delete_dataset_cache("dataset_name")
    dataset = load_dataset("stable-speech/concatenated-accent-dataset")
    #create a directory output_directory
    os.makedirs("output_directory", exist_ok=True)
    # Iterate through the dataset to save audio files
    for i, example in enumerate(dataset['train']):
        audio_data = example['audio']['array']
        sampling_rate = example['audio']['sampling_rate']
        labels = str(example['labels'])
        file_name = f"audio_{i}_{labels}.wav"  # Customize the file name as per your requirements
        output_path = os.path.join("output_directory", file_name)  # Specify the directory to save the files

        # Save audio file as a WAV file
        sf.write(output_path, audio_data, samplerate=sampling_rate, subtype='PCM_24')


    # # Initialize a set to store unique labels\
    all_labels = dataset['train'].unique('labels')
    # print(all_labels)
    # # Initialize a dictionary to store counts of each label
    label_counts = {str(label): 0 for label in all_labels}
    print(all_labels)
    # Count occurrences of each label
    for example in dataset['train']:
        #Play the audio file
        print(example)
        #save audio file to wav file

        Audio(data=example['audio']['array'], rate=example['audio']['sampling_rate'])
       # librosa.display.waveshow(example['audio']['array'], sr=example['audio']['sampling_rate'])
        labels = str(example['labels'])
        label_counts[labels] += 1


    # Print the counts
    print("Label Counts:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    #label_to_index = {label: index for index, label in enumerate(unique_labels)}

    exit(1)


    # Define the VGG19 model
    class VGG19(nn.Module):
        def __init__(self,path_to_weights=None,current_directory=None):
            super().__init__()
            self.vgg19 = LoadVGG19(path_to_weights,current_directory)

        def forward(self, x):
            return self.vgg19(x)

    # Load VGG19 model with modified architecture
    def LoadVGG19(path_to_weights=None,current_directory=None):
        vgg19 = models.vgg19(pretrained=False)
        #Load Weights from the pre-trained model
        if path_to_weights:
            vgg19.load_state_dict(torch.load(path_to_weights))

        # Modify the first layer to accept 1 channel input
        vgg19.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        # Modify the fully connected layers
        num_features = vgg19.classifier[0].in_features
        vgg19.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 2),  # Change num_classes to the number of output classes
        )

        # Load weights if provided
        if path_to_weights:
            state_dict = torch.load(os.path.join(current_directory, "vgg19_new.pth"))
            # Remove 'vgg19.' from keys in state_dict to match the model structure
            state_dict = {k.replace('vgg19.', ''): v for k, v in state_dict.items()}
            vgg19.load_state_dict(state_dict)
            #Save the weights for backup
            torch.save(vgg19.state_dict(), os.path.join(current_directory, "vgg19_new_bckp.pth"))

       # vgg19.load_state_dict(torch.load(os.path.join(current_directory, "vgg19_new.pth")))

        return vgg19
    #    #modify each state_dict name from vgg19.features to features
    # features    weights = torch.load(os.path.join(current_directory, "vgg19_new.pth"))
    #     for name, module in weights.named_modules():
    #         if "vgg19" in name:
    #             new_name = name.replace("vgg19.", "")
    #             vgg19._modules[new_name] = module
    #             del vgg19._modules[name]

    if __name__ == "__main__":
        current_directory = os.path.dirname(os.path.abspath(__file__))
        #update
        # Construct relative paths from the current directory
        content_dir = os.path.join(current_directory, 'lib_', 'dataset', 'content', "indian")
        style_dir = os.path.join(current_directory, 'lib_', 'dataset', 'style', "american")

        # Initialize PyTorch DataLoader
        dataloader = DataModule(current_directory, content_dir, style_dir, 32*4).train_dataloader()
        dataloader = dataset = load_dataset("stable-speech/concatenated-accent-dataset")
        # Initialize the model
        model = VGG19(os.path.join(current_directory, '', 'vgg.pth'),current_directory)

        # Set device to CPU
        device = torch.device('cpu')
       # device = torch.device("cpu")
        model.to(device)
        #Check if cuda is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model.to(device)
        else:
            print("APGOSPGS")
        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Training loop
        num_epochs = 10
        j = 0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            for batch in dataloader:
                inputs = batch["content"].unsqueeze(1)
                targets = batch["style"].unsqueeze(1)
                one_hot_targets = []
                for i in range(inputs.shape[0]):
                    one_hot_targets.append([1, 0])
                for i in range(targets.shape[0]):
                    one_hot_targets.append([0, 1])
                one_hot_targets = torch.tensor(one_hot_targets)
                #Stack the inputs and targets to one vector, with inputs label 1 and targets label 0
                stacked = torch.cat((inputs, targets), 0)
                #one_hot_targets = torch.cat((torch.ones(inputs.shape[0]), torch.zeros(targets.shape[0])))
                #Convert one_hot_targets to float
                one_hot_targets = one_hot_targets.type(torch.float32)
                #Create one hot targets, when the first 8 are 1 and the last 8 are 0
                stacked, targets = stacked.to(device), targets.to(device)

                # Forward pass
                outputs = model(stacked).type(torch.float32)
                loss = criterion(outputs, one_hot_targets)
                print('Current Loss:', loss.item())
               # print("Current Output:", outputs.type(torch.float32), "Current Target:", one_hot_targets.type(torch.float32))
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                #Check Accuracy
                # Calculate accuracy

                j += 1
                if j % 3 == 0:
        #SAve the model weights as pth
            # Calculate accuracy
                    predicted_classes = torch.argmax(outputs, dim=1)
                    true_classes = torch.argmax(one_hot_targets, dim=1)
                    correct_predictions += (predicted_classes == true_classes).sum().item()
                    total_samples += 2*inputs.size(0)
                    torch.save(model.state_dict(), os.path.join(current_directory, '', 'vgg19_new.pth'))
                    print("Model weights saved to vgg19.pth")
                    print(f'Accuracy: {correct_predictions / total_samples:.4f}')
                    correct_predictions = 0
                    total_samples = 0

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    exit(1)
