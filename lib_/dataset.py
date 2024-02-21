import random
import warnings
from pathlib import Path

from PIL import Image
import torch
import pickle
from torch.utils.data import IterableDataset, Dataset
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop
from torchvision.utils import save_image
import argparse
#TO DELETE LATER
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
import numpy as np
import librosa
import pytorch_lightning as pl
from vgg import VGGEncoder
import os
from sklearn.metrics import accuracy_score
def files_in(dir):
    return list(sorted(Path(dir).glob('*')))


def load(file):
    # Load the pickle file
    with open(file, 'rb') as f:
        data = pickle.load(f)

    # Convert the loaded data to a torch tensor
    tensor_data = torch.tensor(data)

    return tensor_data

def save(img_tensor, file):
    if img_tensor.ndim == 4:
        assert len(img_tensor) == 1

    save_image(img_tensor, str(file))


def style_transforms(size=256):
    # Style images must be 256x256 for AdaConv
    return Compose([
        Resize(size=size),  # Resize to keep aspect ratio
        CenterCrop(size=(size, size)),  # Center crop to square
        ToTensor()])


def content_transforms(min_size=None):
    # min_size is optional as content images have no size restrictions
    transforms = []
    if min_size:
        transforms.append(Resize(size=min_size))
    transforms.append(ToTensor())
    return Compose(transforms)



class AccentHuggingBased(Dataset):
    def __init__(self,dataset,type="train",batch_size=1):
        self.batch_size = batch_size
        self.dataset = dataset[type].select_columns(['audio', 'labels']).map(lambda e: {'audio': e['audio']['array'], 'label': e['labels'], 'sr': e['audio']['sampling_rate']})
        #Filter all the labels that have less than 400 samples

        #save the dataset to a file for later use
       # self.dataset.save_to_disk('accent_hugging_based_dataset')
        self.label_to_number_mapping = None
        self.num_classes = 0
        self._filter_labels_with_less_than_samples()

    def _filter_labels_with_less_than_samples(self, min_samples=300):
        label_counts = {}
        for item in self.dataset:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        # Identify labels with less than 400 samples
        labels_to_remove = [label for label, count in label_counts.items() if count < min_samples]

        # Filter out samples associated with labels that have less than 400 samples
        self.dataset = self.dataset.filter(lambda e: e['label'] not in labels_to_remove)


    def _create_mapping(self):
        # Create a mapping from a label (str) to a number between 0 to amount of labels - 1,
        # this mapping will be always the same for the same labels
        #Check if the mapping file already exists
        if os.path.exists('label_to_number_mapping.pkl'):
            with open('label_to_number_mapping.pkl', 'rb') as f:
                self.label_to_number_mapping = pickle.load(f)
                self.num_classes = len(self.label_to_number_mapping)
                return
        unique_labels = self.dataset.unique('labels')
        #sort unique labels by alphabetical order
        unique_labels.sort()
        self.label_to_number_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        #save the mapping to a file for later use
        with open('label_to_number_mapping.pkl', 'wb') as f:
            pickle.dump(self.label_to_number_mapping, f)



    def get_label_number(self, label):
        if self.label_to_number_mapping is None:
            self._create_mapping()
        return self.label_to_number_mapping[label]

    def _CreateMapping(self):
        #Create a mapping from a label (str) to a number between 0 to amount of labels - 1, this mapping will be always the same for the same labels
        self.labels = self.dataset.unique('label')


    @staticmethod
    def _process_audio_array(audio_data, sr, segment_duration=4):
        # Calculate the total number of segments

        # Calculate the number of samples per segment
        samples_per_segment = sr * segment_duration

        # Compute the total number of segments
        num_segments = math.ceil(len(audio_data) / samples_per_segment)

        # Pad the audio to make it evenly divisible into segments
        pad_length = num_segments * samples_per_segment - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_length), mode='constant')
        # Initialize an empty list to store spectrograms and target amplitudes
        spectrograms = []
        target_amplitudes = []

        # Generate a Mel-spectrogram for each segment
        for i in range(num_segments):
            # Extract segment from the padded audio
            start_sample = samples_per_segment * i
            end_sample = start_sample + samples_per_segment
            audio_segment = audio_data[start_sample:end_sample]

            # Generate Mel-spectrogram for the segment
            n_fft = 2048
            hop_length = 512
            n_mels = 256
            S = librosa.feature.melspectrogram(
                y=audio_segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann'
            )
            # Convert to log scale (dB)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Compute target amplitude for the segment
            target_amplitude = np.max(np.abs(audio_segment))

            # Append spectrogram and target amplitude to the lists
            spectrograms.append(S_dB)
            target_amplitudes.append(target_amplitude)
        return spectrograms, target_amplitudes

    def _createBatch(self, sample_indices):
        batch_audio = []
        batch_labels = []
        batch_sr = []
        batch_max_amplitudes = []

        for idx in sample_indices:
            audio_data = self.dataset[idx]['audio']
            sr = self.dataset[idx]['sr']

            spectrograms, target_amplitudes = self._process_audio_array(audio_data, sr)

            # Append spectrograms to the batch
            for j in range(len(spectrograms)):
                batch_audio.append(torch.stack([torch.tensor(spectrograms[j]) for _ in range(3)]))
            for j in range(len(spectrograms)):
                batch_labels.append(self.get_label_number(self.dataset[idx]['label']))
            for j in range(len(spectrograms)):
                batch_sr.append(sr)
            for j in range(len(target_amplitudes)):
                batch_max_amplitudes.append(target_amplitudes[j])

        return torch.stack(batch_audio), torch.tensor(batch_labels), torch.tensor(batch_sr), torch.tensor(batch_max_amplitudes)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.dataset))

        sample_indices = range(start_idx, end_idx)
        batch_audio, batch_labels, batch_sr, max_amp = self._createBatch(sample_indices)

        return batch_audio, batch_labels, batch_sr, max_amp


class AccentHuggingBasedDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        #only 1000 items for now
        self.dataset = load_dataset("stable-speech/concatenated-accent-dataset")
        if os.path.exists('label_to_number_mapping.pkl'):
            with open('label_to_number_mapping.pkl', 'rb') as f:
                self.num_classes = len(pickle.load(f))
        else:
            self.num_classes = 2

        #self.dataset = load_dataset("stable-speech/concatenated-accent-dataset", )

    def modify_batch(self, batch):
        def modify_batch(self, batch):
            # Remove all tensors from the batch that are all zeros or have low amplitudes
            filtered_batch = [tensor for tensor in batch if torch.sum(tensor) != 0]

            return filtered_batch

        return batch

    def train_dataloader(self):
        # Use the modify_batch function to add noise to the batch
        return DataLoader(AccentHuggingBased(self.dataset,batch_size=self.batch_size, type="train"), batch_size=1,shuffle=True)#, num_workers=4)

    def val_dataloader(self):
        return DataLoader(AccentHuggingBased(self.dataset, type="test"), batch_size=1, shuffle=True)

    def test_dataloader(self):
        return DataLoader(AccentHuggingBased(self.dataset, type="test"), batch_size=1, shuffle=True)

    def get_dataloader(self):
        return self.train_dataloader()

    def transfer_batch_to_device(self, batch, device):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor()):
                batch[k] = v.to(device)
        return batch

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass


#TRAIN THE MODEL
# Define the Lightning module for training
class MOSHIKOTrainer(pl.LightningModule):
    def __init__(self, model, dataloader):
        super().__init__()
        self.model = model
        self.dataloader = dataloader

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, sr, max_amp = batch
        # Forward pass
        x = x.float()
        y = y.squeeze(0)
        logits = self.model(x.squeeze(0))

        # Compute the loss using CrossEntropyLoss
        loss = torch.nn.CrossEntropyLoss()(logits, y)

        # Compute accuracy
        _, predicted = torch.max(logits, 1)
        acc = accuracy_score(y.cpu().numpy(), predicted.cpu().numpy())

        # Log accuracy and loss for visualization
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Define your optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return self.dataloader

def main(args):
    # Define data loader and current directory
    dataloader = AccentHuggingBasedDataLoader(batch_size=args.batch_size)
    current_directory = os.getcwd()

    # Initialize model and Lightning modules
    model = VGGEncoder(TzlilTrain=True, current_directory=current_directory, path_to_weights=args.weights_path, num_classes=dataloader.num_classes)
    dataloader = dataloader.get_dataloader()
    trainer = pl.Trainer(max_epochs=args.epochs)
    moshiko_trainer = MOSHIKOTrainer(model, dataloader)

    # Start training
    trainer.fit(moshiko_trainer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--weights_path", type=str, default="vgg.pth", help="Path to the weights file")
    args = parser.parse_args()
    main(args)


# Test the dataloader
#dataloader = AccentHuggingBasedDataLoader(batch_size=130).get_dataloader()
dataloader = AccentHuggingBasedDataLoader(batch_size=24)
current_directory = os.getcwd()
# Initialize model and Lightning modules

model = VGGEncoder(TzlilTrain=True,current_directory=current_directory, path_to_weights="vgg.pth",num_classes=dataloader.num_classes)
dataloader = dataloader.get_dataloader()
trainer = pl.Trainer(max_epochs=10)
moshiko_trainer = MOSHIKOTrainer(model, dataloader)

# Start training
trainer.fit(moshiko_trainer)
exit(1)






























class AccentHuggingBased(Dataset):
    def __init__(self, content_files, style_files, content_transform=None, style_transform=None):
        self.content_files = content_files
        self.style_files = style_files

        id = lambda x: x
        self.content_transform = id if content_transform is None else content_transform
        self.style_transform = id if style_transform is None else style_transform

    def __len__(self):
        return len(self.content_files) * len(self.style_files)

    def __getitem__(self, idx):
        content_file, style_file = self.files_at_index(idx)

        content_img = load(content_file)
        style_img = load(style_file)

        content_img = self.content_transform(content_img)
        style_img = self.style_transform(style_img)

        return {
            'content': content_img,
            'style': style_img,
        }

    def files_at_index(self, idx):
        content_idx = idx % len(self.content_files)
        style_idx = idx // len(self.content_files)

        assert 0 <= content_idx < len(self.content_files)
        assert 0 <= style_idx < len(self.style_files)
        return self.content_files[content_idx], self.style_files[style_idx]

    @classmethod
    def from_dataset(cls, dataset_name, content_transform=None, style_transform=None):
        # Load the dataset
        dataset = load_dataset(dataset_name)

        # Extract content and style files from the dataset
        content_files = dataset['train']['content_files']
        style_files = dataset['train']['style_files']

        return cls(content_files, style_files, content_transform, style_transform)




class AccentDataset(Dataset):
    def __init__(self,content_files, style_files, content_transform=None, style_transform=None):
        self.content_files = content_files
        self.style_files = style_files

        id = lambda x: x
        self.content_transform = id if content_transform is None else content_transform
        self.style_transform = id if style_transform is None else style_transform

    def __getitem__(self, idx):
        content_file, style_file = self.files_at_index(idx)
        content_img = load(content_file)
        style_img = load(style_file)

        content_img = self.content_transform(content_img)
        style_img = self.style_transform(style_img)
        #padd the y axis to 500
 #       content_img = torch.nn.functional.pad(content_img, ( 0, 256 - content_img.shape[1]), 'constant', 0)
       # style_img = torch.nn.functional.pad(style_img, (0,  256 - style_img.shape[1]), 'constant', 0)
        return {
            'content': content_img,
            'style': style_img,
        }
    def __len__(self):
        return max(len(self.content_files) , len(self.style_files))
    def files_at_index(self, idx):
        content_idx = idx % len(self.content_files)
        style_idx = idx // len(self.content_files)

        assert 0 <= content_idx < len(self.content_files)
        assert 0 <= style_idx < len(self.style_files)
        return self.content_files[content_idx], self.style_files[style_idx]
class EndlessAccentDataset(IterableDataset):
    """
    Wrapper for AccentDataset which loops infinitely.
    Usefull when training based on iterations instead of epochs
    """
    def __init__(self, *args, **kwargs):
        self.dataset = AccentDataset(*args, **kwargs)
    def __len__(self):
        return len(self.dataset)
    def __iter__(self):
        while True:
            idx = random.randrange(len(self.dataset))

            try:
                yield self.dataset[idx]
            except Exception as e:
                files = self.dataset.files_at_index(idx)
                warnings.warn(f'\n{str(e)}\n\tFiles: [{str(files[0])}, {str(files[1])}]')


