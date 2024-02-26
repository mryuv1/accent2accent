
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
from functools import partial
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
import numpy as np
import librosa
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from vgg import VGGEncoder
import os
from sklearn.metrics import accuracy_score
import wandb
import multiprocessing
#set the environment variable to enable MPS
#os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["PYTORCH_MPS"] = "0"
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
    def __init__(self,dataset,type="train",batch_size=1,SlowRun=True,limit_samples=5000,enable_multiprocessing=True):
        self.batch_size = batch_size
        # if not enable_multiprocessing:
        #     if len(dataset)>1:
        #         self.dataset = []
        #         for part in dataset:
        #             processed_part = part[type].select_columns(['audio', 'labels']).map(
        #                 lambda e: {'audio': e['audio']['array'], 'label': e['labels'], 'sr': e['audio']['sampling_rate']})
        #             self.dataset.extend(processed_part)
        #     else:
        #         self.dataset = dataset[0][type].select_columns(['audio', 'labels']).map(lambda e: {'audio': e['audio']['array'], 'label': e['labels'], 'sr': e['audio']['sampling_rate']})
        # else:
        #     if len(dataset) > 1:
        #         self.dataset = []
        #         num_processes = min(multiprocessing.cpu_count(), len(dataset))
        #         print("Number of processes: ", num_processes)
        #         with multiprocessing.Pool(processes=num_processes) as pool:
        #   #          processed_parts = pool.map(self.process_part, [dataset,type,limit_samples])
        #             processed_parts = pool.map(partial(self.process_part, type=type, limit_samples=limit_samples),
        #                                        dataset)
        #             for part in processed_parts:
        #                 self.dataset.extend(part)
        #     else:
        #         self.dataset = dataset[0][type].select_columns(['audio', 'labels']).map(
        #             lambda e: {'audio': e['audio']['array'], 'label': e['labels'],
        #                        'sr': e['audio']['sampling_rate']})
        self.dataset = dataset[0][type]
        #Perform shuffle on the dataset
        self.label_to_number_mapping = None
        self.num_classes = 0
        self.slow_run = SlowRun
        self.good_labels = []
        self._create_mapping()
        self.check_if_valid_labels_exist()
        if not SlowRun:
            self._filter_labels_not_in_good_labels()
        self.dataset = self.dataset.shuffle()
        print("Number of classes: In Hugging face ", self.num_classes)
        print("AMount of good labels: ", len(self.good_labels))
        print("All Labels existed", self.dataset.unique('labels'))
        print("Mapping: ", self.label_to_number_mapping)



    def process_part(self, part,type,limit_samples=5000):
        return part[type][:limit_samples].select_columns(['audio', 'labels']).map(
            lambda e: {'audio': e['audio']['array'], 'label': e['labels'], 'sr': e['audio']['sampling_rate']})

    def check_if_valid_labels_exist(self):
        if not os.path.exists('valid_labels.pkl'):
            #Write all the labels of dataset to a file
            unique_labels = self.dataset.unique('label')
            #Check if the length of the unique labels is greater than 1, else raise an error
            if len(unique_labels) <= 1:
                raise ValueError("No labels found in the dataset")
            with open('valid_labels.pkl', 'wb') as f:
                pickle.dump(unique_labels, f)

    def _append_to_good_labels(self, label):
        if label not in self.good_labels:
            self.good_labels.append(label)
            with open('valid_labels.pkl', 'wb') as f:
                pickle.dump(self.good_labels, f)

    def _remove_from_good_labels(self, label):
        if label in self.good_labels:
            self.good_labels.remove(label)
            with open('valid_labels.pkl', 'wb') as f:
                pickle.dump(self.good_labels, f)
    def _filter_labels_not_in_good_labels(self):
        self.dataset = self.dataset.filter(lambda e: self._label_map(e['labels']) in self.good_labels)
        self._create_mapping()

    def _filter_labels_with_less_than_samples(self, min_samples=180):
        label_counts = {}
        for item in self.dataset:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        # Identify labels with less than 400 samples
        labels_to_remove = [label for label, count in label_counts.items() if count < min_samples]

        # Filter out samples associated with labels that have less than min_samples samples
        self.dataset = self.dataset.filter(lambda e: e['label'] not in labels_to_remove)
        #create mapping based on the filtered dataset
        self._create_mapping()


    def _create_mapping(self):
        if os.path.exists('valid_labels.pkl'):
            with open('valid_labels.pkl', 'rb') as f:
                self.good_labels = pickle.load(f)
                self.label_to_number_mapping = {label: idx for idx, label in enumerate(self.good_labels)}
        else:
            unique_labels = self.dataset.unique('label')
            print("Unique labels: ", unique_labels)
            # sort unique labels by alphabetical order
            unique_labels.sort()
            self.label_to_number_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            # MAYBE CORRECT
            self.good_labels = unique_labels
        self.num_classes = len(self.good_labels)

        # if not self.slow_run:
        #     if os.path.exists('valid_labels.pkl'):
        #         with open('valid_labels.pkl', 'rb') as f:
        #             self.good_labels = pickle.load(f)
        #     else:
        #         unique_labels = self.dataset.unique('labels')
        #     print("Unique labels: ", unique_labels)
        #     # sort unique labels by alphabetical order
        #     unique_labels.sort()
        #     self.label_to_number_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        #     # MAYBE CORRECT
        #     self.good_labels = unique_labels
        #     self.num_classes = len(unique_labels)
        # else:
        #     # Create a mapping from a label (str) to a number between 0 to amount of labels - 1,
        #     # this mapping will be always the same for the same labels
        #     #Check if file contain the valid labels is already exists if so load it
        #     if os.path.exists('valid_labels.pkl'):
        #         with open('valid_labels.pkl', 'rb') as f:
        #             self.good_labels = pickle.load(f)
        #             self.num_classes = len(self.good_labels)
        #             print("Number of classes: ", self.num_classes)
        #             self.label_to_number_mapping = {label: idx for idx, label in enumerate(self.good_labels)}
        #             print("Mapping: ", self.label_to_number_mapping)
        #     #Check if the mapping file already exists
        #     else:
        #         if os.path.exists('label_to_number_mapping.pkl'):
        #             with open('label_to_number_mapping.pkl', 'rb') as f:
        #                 self.label_to_number_mapping = pickle.load(f)
        #                 self.num_classes = max(len(self.label_to_number_mapping), len(self.good_labels))
        #                 return
        #         self.label_to_number_mapping = {label: idx for idx, label in enumerate(self.good_labels)}
        # #save the mapping to a file for later use
        # with open('label_to_number_mapping.pkl', 'wb') as f:
        #     pickle.dump(self.label_to_number_mapping, f)

    def _label_map(self,label):
        # Define the mapping from old labels to new labels
        label_mapping = {
            'United States English': 'American',
            'Canadian English': 'Canadian',
            'India and South Asia (India, Pakistan, Sri Lanka)': 'indian',
            'England English': 'British',
            'German English,Non native speaker': 'German',
            'Welsh English': 'Welsh',
            'Scottish English': 'Scottish',
            'Southern African (South Africa, Zimbabwe, Namibia)': 'SouthAfrican',
            'New Zealand English': 'NewZealand',
            'Australian English': 'Australian',
            'Irish English': 'Irish',
            'Northern Irish': 'NorthernIrish',
            'Filipino': 'Filipino',
            'Singaporean English': 'Singaporean',
            'Liverpool English,Lancashire English,England English': 'British',
            'Hong Kong English': 'Chinese'
        }
        if label not in label_mapping:
            return label
        return label_mapping[label]

    def get_label_number(self, label):
        label = self._label_map(label)
        if label not in self.good_labels:
            return -1
        if self.label_to_number_mapping is None:
            self._create_mapping()
        #Check if the label is "English", if so return the mapping of "American"

        if label == "English":
            label = "American"
        return self.label_to_number_mapping[label]



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
        audio_data = audio_data.astype(np.float32)
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

    def _createBatch2(self, sample_indices):
        batch_audio = []
        batch_labels = []
        batch_sr = []
        batch_max_amplitudes = []

        for idx in sample_indices:
            audio_data = self.dataset[idx]['audio']['array']
            sr = self.dataset[idx]['audio']['sampling_rate']
            label_number = self.get_label_number(self.dataset[idx]['labels'])
            if label_number == -1:
                # Skip this sample and continue with the next one
                continue
            spectrograms, target_amplitudes = self._process_audio_array(audio_data, sr)
            # Append spectrograms to the batch
            for j in range(len(spectrograms)):
                batch_audio.append(torch.stack([torch.tensor(spectrograms[j]) for _ in range(3)]))
                batch_labels.append(label_number)
                batch_sr.append(sr)
                batch_max_amplitudes.append(target_amplitudes[j])



        return torch.stack(batch_audio), torch.tensor(batch_labels), torch.tensor(batch_sr), torch.tensor(
            batch_max_amplitudes)

    def _createBatch(self, sample_indices):
        batch_audio = []
        batch_labels = []
        batch_sr = []
        batch_max_amplitudes = []

        zero_audio = []
        zero_labels = []
        zero_sr = []
        zero_max_amplitudes = []

        label_count = {}
        zero_count = 0
        all_count = 0

        for idx in sample_indices:
            audio_data = self.dataset[idx]['audio']['array']
            sr = self.dataset[idx]['audio']['sampling_rate']
            label_number = self.get_label_number(self.dataset[idx]['labels'])
            if label_number == -1:
                continue
            if label_number == 0:
                spectrograms, target_amplitudes = self._process_audio_array(audio_data, sr)
                for j in range(len(spectrograms)):
                    zero_audio.append(torch.stack([torch.tensor(spectrograms[j]) for _ in range(3)]))
                    zero_labels.append(label_number)
                    zero_sr.append(sr)
                    zero_max_amplitudes.append(target_amplitudes[j])
                    zero_count += 1
                    all_count += 1
                # Count the occurrences of each label
                label_count[label_number] = label_count.get(label_number, 0) + 1
            else:
                spectrograms, target_amplitudes = self._process_audio_array(audio_data, sr)
                for j in range(len(spectrograms)):
                    batch_audio.append(torch.stack([torch.tensor(spectrograms[j]) for _ in range(3)]))
                    batch_labels.append(label_number)
                    batch_sr.append(sr)
                    batch_max_amplitudes.append(target_amplitudes[j])
                    all_count += 1
                # Count the occurrences of each label
                label_count[label_number] = label_count.get(label_number, 0) + 1

        # Normalize zero labels if they exceed 20% of the batch
        if zero_count / all_count > 0.2:
            # Calculate the number of zero labels to remove
            num_to_remove = int(zero_count*0.65)
            # Randomly select instances to remove
            indices_to_remove = random.sample(range(zero_count), num_to_remove)
            # Remove instances from the zero label lists
            zero_audio = [zero_audio[i] for i in range(zero_count) if i not in indices_to_remove]
            zero_labels = [zero_labels[i] for i in range(zero_count) if i not in indices_to_remove]
            zero_sr = [zero_sr[i] for i in range(zero_count) if i not in indices_to_remove]
            zero_max_amplitudes = [zero_max_amplitudes[i] for i in range(zero_count) if i not in indices_to_remove]

        # Combine all lists
        batch_audio += zero_audio
        batch_labels += zero_labels
        batch_sr += zero_sr
        batch_max_amplitudes += zero_max_amplitudes


        labels_used = {}
        for label in batch_labels:
            if label not in labels_used:
                labels_used[label] = 1
            else:
                labels_used[label] += 1
        print("    Labeled Used ", labels_used)
        return torch.stack(batch_audio), torch.tensor(batch_labels), torch.tensor(batch_sr), torch.tensor(
            batch_max_amplitudes)
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.dataset))

        sample_indices = range(start_idx, end_idx)
        batch_audio, batch_labels, batch_sr, max_amp = self._createBatch(sample_indices)
  #      return batch_audio.to("cpu"), batch_labels.to("cpu"), batch_sr.to("cpu"), max_amp.to("cpu")
        return batch_audio, batch_labels, batch_sr, max_amp


class AccentHuggingBasedDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size=32,include_India=False,include_newdataset=False, SlowRun=False):
        self.SlowRun = SlowRun
        self.batch_size = batch_size
        #only 1000 items for now
        self.dataset = load_dataset("NathanRoll/commonvoice_train_gender_accent_16k")
        self.dataset = self.dataset.rename_column('accent', 'labels')
        if include_newdataset:
            self.dataset2 = load_dataset("stable-speech/concatenated-accent-dataset")
            #change the labels in train and test of 'accent' to 'labels'
            self.dataset = [self.dataset, self.dataset2]
        else:
            self.dataset = [self.dataset]
        if os.path.exists('valid_labels.pkl'):
            with open('valid_labels.pkl', 'rb') as f:
                self.num_classes = len(pickle.load(f))
        else:
            self.num_classes = 2


    def modify_batch(self, batch):
        def modify_batch(self, batch):
            # Remove all tensors from the batch that are all zeros or have low amplitudes
            filtered_batch = [tensor for tensor in batch if torch.sum(tensor) != 0]

            return filtered_batch

        return batch

    def train_dataloader(self):
        # Use the modify_batch function to add noise to the batch
        return DataLoader(AccentHuggingBased(self.dataset,batch_size=self.batch_size, type="train",SlowRun=self.SlowRun), batch_size=1,shuffle=True)#, num_workers=4)

    def val_dataloader(self):
        return DataLoader(AccentHuggingBased(self.dataset, type="test",SlowRun=self.SlowRun), batch_size=1, shuffle=True)

    def test_dataloader(self):
        return DataLoader(AccentHuggingBased(self.dataset, type="test",SlowRun=self.SlowRun), batch_size=1, shuffle=True)

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
class MOSHIKOTrainer(pl.LightningModule):
    def __init__(self, model, dataloader, save_dir):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.save_dir = save_dir


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, sr, max_amp = batch
        # Move data to GPU if available
        x = x.float()
        y = y.long().squeeze(0)

        logits = self.model(x.squeeze(0))
        #print logits and y data type

        # Compute the loss using CrossEntropyLoss
        loss = torch.nn.functional.cross_entropy(logits, y)

        # Compute accuracy
        _, predicted = torch.max(logits, 1)
        acc = accuracy_score(y.cpu().numpy(), predicted.cpu().numpy())

        # Log accuracy and loss for visualization
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        wandb.log({"acc": acc, "loss": loss})
        return loss

    def configure_optimizers(self):
        # Define your optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return self.dataloader

    def on_epoch_end(self):
        # Save weights at the end of each epoch
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'epoch_{self.current_epoch}_weights.pth'))


def main(args):
    wandb.init(project="accent2accent")

    # Define data loader and current directory
    dataloader = AccentHuggingBasedDataLoader(batch_size=args.batch_size, include_newdataset=args.big_dataset, SlowRun=args.SlowModel)
    current_directory = os.getcwd()

    # Initialize model and Lightning modules
    model = VGGEncoder(TzlilTrain=True, current_directory=current_directory, path_to_weights=args.weights_path,
                       num_classes=dataloader.num_classes)

    wandb.watch(model)
    dataloader = dataloader.get_dataloader()
    save_dir = args.save_dir  # Directory to save weights

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir, filename='{epoch}-{step}',save_top_k = 3, monitor="acc", mode="max", every_n_train_steps=30)
    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback])
    moshiko_trainer = MOSHIKOTrainer(model, dataloader, save_dir)

    # Start training
    trainer.fit(moshiko_trainer)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument("--batch_size", type=int, default=70, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--weights_path", type=str, default="vgg.pth", help="Path to the weights file")
    parser.add_argument("--save_dir", type=str, default="NewVGGWeights", help="Directory to save weights")
    parser.add_argument("--big_dataset", type=bool, default=False, help="If want to include the big dataset in the dataloader")
    parser.add_argument("--SlowModel", type=bool, default=False, help="If want to use the slow model")
    args = parser.parse_args()
    main(args)
    #TODO WHEN combine need to change the only the datasets, we need to remain the thing that save the model from weiz computer



