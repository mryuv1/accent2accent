import random
import warnings
from pathlib import Path

from PIL import Image
import torch
import pickle
from torch.utils.data import IterableDataset, Dataset
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop
from torchvision.utils import save_image

#TO DELETE LATER
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
import numpy as np
import librosa
import pytorch_lightning as pl

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
    def __init__(self,dataset,type="train"):
        self.dataset = dataset[type].select_columns(['audio', 'labels']).map(lambda e: {'audio': e['audio']['array'], 'label': e['labels'], 'sr': e['audio']['sampling_rate']})
    @staticmethod
    def _process_audio_array(audio_data, sr, segment_duration=2):
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        a = self._process_audio_array(self.dataset[idx]['audio'], self.dataset[idx]['sr'])
        return a, self.dataset[idx]['label'], self.dataset[idx]['sr']


class AccentHuggingBasedDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.dataset = load_dataset("stable-speech/concatenated-accent-dataset")

    def modify_batch(self, batch):
        def modify_batch(self, batch):
            # Remove all tensors from the batch that are all zeros or have low amplitudes
            filtered_batch = [tensor for tensor in batch if torch.sum(tensor) != 0]

            return filtered_batch

        return batch

    def train_dataloader(self):
        # Use the modify_batch function to add noise to the batch
        return DataLoader(AccentHuggingBased(self.dataset, type="train"), batch_size=self.batch_size, shuffle=True)

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

#Test the dataloader
dataloader = AccentHuggingBasedDataLoader(batch_size=4).get_dataloader()
for batch in dataloader:
    print(batch)
    break
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


