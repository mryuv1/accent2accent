
import random
import torch
import pickle
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
import numpy as np
import librosa
import pytorch_lightning as pl
import os
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from lib_.adaconv.adaconv_model import AdaConvModel
from lib_.adain.adain_model import AdaINModel
from lib_.loss import MomentMatchingStyleLoss, GramStyleLoss, CMDStyleLoss, MSEContentLoss
class AccentHuggingBased(Dataset):
    def __init__(self,data_type="train",batch_size=2,SlowRun=True,limit_samples=5000,enable_multiprocessing=True,TzlilTrain=False):
        self.batch_size = batch_size

        # Load the dataset
        dataset = load_dataset("NathanRoll/commonvoice_train_gender_accent_16k")["train"]
        # Rename column 'accent' to 'labels'
        dataset = dataset.rename_column('accent', 'labels')

        # Shuffle the dataset
        dataset = dataset.shuffle(seed=38)
        if data_type == "train":
            self.dataset = dataset.select(range(0,int(len(dataset)*0.92)))
        else:
            self.dataset = dataset.select(range(int(len(dataset)*0.92),len(dataset)))

        self.label_to_number_mapping = None
        self.num_classes = 0
        self.slow_run = SlowRun
        self.TzlilTrain = TzlilTrain
        self.good_labels = []
        self._create_mapping()
        self.check_if_valid_labels_exist()
        if not SlowRun and data_type == "train":
            self._filter_labels_not_in_good_labels()



    def process_part(self, part,type,limit_samples=5000):
        return part[type][:limit_samples].select_columns(['audio', 'labels']).map(
            lambda e: {'audio': e['audio']['array'], 'label': e['labels'], 'sr': e['audio']['sampling_rate']})

    def check_if_valid_labels_exist(self):
        if not os.path.exists('../lib_/valid_labels.pkl'):
            #Write all the labels of dataset to a file
            unique_labels = self.dataset.unique('label')
            #Check if the length of the unique labels is greater than 1, else raise an error
            if len(unique_labels) <= 1:
                raise ValueError("No labels found in the dataset")
            with open('../lib_/valid_labels.pkl', 'wb') as f:
                pickle.dump(unique_labels, f)

    def _append_to_good_labels(self, label):
        if label not in self.good_labels:
            self.good_labels.append(label)
            with open('../lib_/valid_labels.pkl', 'wb') as f:
                pickle.dump(self.good_labels, f)

    def _remove_from_good_labels(self, label):

        if label in self.good_labels:
            self.good_labels.remove(label)
            with open('../lib_/valid_labels.pkl', 'wb') as f:
                pickle.dump(self.good_labels, f)
    def _filter_labels_not_in_good_labels(self):
       # self.dataset = self.dataset.filter(lambda e: self._label_map(e['labels']) in self.good_labels)
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
        if os.path.exists('../lib_/valid_labels.pkl'):
            with open('../lib_/valid_labels.pkl', 'rb') as f:
                self.good_labels = pickle.load(f)
                self.label_to_number_mapping = {label: idx for idx, label in enumerate(self.good_labels)}
        else:
            unique_labels = self.dataset.unique('label')
            # sort unique labels by alphabetical order
            unique_labels.sort()
            self.label_to_number_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            # MAYBE CORRECT
            self.good_labels = unique_labels
        self.num_classes = len(self.good_labels)


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
        """
           Process the input audio data into spectrograms and target amplitudes.

           Args:
               audio_data (np.ndarray): Input audio data.
               sr (int): Sample rate of the audio data.
               segment_duration (float): Duration of each segment in seconds. Default is 4.

           Returns:
               tuple: A tuple containing a list of spectrograms and a list of target amplitudes.
           """
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
                    zero_audio.append(torch.stack([torch.tensor(spectrograms[j]) for _ in range(1)]))
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
                    batch_audio.append(torch.stack([torch.tensor(spectrograms[j]) for _ in range(1)]))
                    batch_labels.append(label_number)
                    batch_sr.append(sr)
                    batch_max_amplitudes.append(target_amplitudes[j])
                    all_count += 1
                # Count the occurrences of each label
                label_count[label_number] = label_count.get(label_number, 0) + 1
        max_size = min(len(batch_audio), len(zero_audio))
        content = []
        style = []
        for i in range(max_size):
            content.append(batch_audio[i])
            style.append(zero_audio[i])
        batch_labels += zero_labels
        while content == []:
            print("PROBLEM - CONTENT IS 0 , need to check maybe the labels are the same")
            #get a random index, and check if the label after get_label_number is 0, if the label is 0, then append the content len(style) times
            idx = random.randint(0, len(self.dataset)-1)
            audio_data = self.dataset[idx]['audio']['array']
            sr = self.dataset[idx]['audio']['sampling_rate']
            label_number = self.get_label_number(self.dataset[idx]['labels'])
            if label_number != 0:
                continue
            spectrograms, target_amplitudes = self._process_audio_array(audio_data, sr)
            #Now append to content len(style) times
            for j in range(len(spectrograms)):
                content.append(torch.stack([torch.tensor(spectrograms[j]) for _ in range(1)]))
                batch_labels.append(label_number)
                batch_sr.append(sr)
                batch_max_amplitudes.append(target_amplitudes[j])
                all_count += 1
                label_count[label_number] = label_count.get(label_number, 0) + 1
        while style == []:
            print("PROBLEM - STYLE IS 0 , need to check maybe the labels are the same")
            #get a random index, and check if the label after get_label_number is 0, if the label is 0, then append the content len(style) times
            idx = random.randint(0, len(self.dataset))
            audio_data = self.dataset[idx]['audio']['array']
            sr = self.dataset[idx]['audio']['sampling_rate']
            label_number = self.get_label_number(self.dataset[idx]['labels'])
            if label_number == 0 or label_number == -1:
                continue
            spectrograms, target_amplitudes = self._process_audio_array(audio_data, sr)
            #Now append to content len(style) times
            for j in range(len(spectrograms)):
                style.append(torch.stack([torch.tensor(spectrograms[j]) for _ in range(1)]))
                batch_labels.append(label_number)
                batch_sr.append(sr)
                batch_max_amplitudes.append(target_amplitudes[j])
                all_count += 1
                label_count[label_number] = label_count.get(label_number, 0) + 1
        labels_used = {}
        for label in batch_labels:
            if label not in labels_used:
                labels_used[label] = 1
            else:
                labels_used[label] += 1
        print("    Labeled Used ", labels_used)
        return {"content": torch.stack(content), "style": torch.stack(style)}, torch.tensor(batch_labels), torch.tensor(batch_sr), torch.tensor(
            batch_max_amplitudes)
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
        if self.TzlilTrain:
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, len(self.dataset))

            sample_indices = range(start_idx, end_idx)
            batch_audio, batch_labels, batch_sr, max_amp = self._createBatch(sample_indices)
            return batch_audio, batch_labels, batch_sr, max_amp
        else:
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, len(self.dataset))
            sample_indices = range(start_idx, end_idx)
            batch_audio, batch_labels, batch_sr, max_amp = self._createBatch2(sample_indices)
            return batch_audio
            return batch_audio, batch_labels, batch_sr, max_amp



class AccentHuggingBasedDataLoader(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=45, help="Batch size for training")
        parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
        parser.add_argument("--weights_path", type=str, default="vgg.pth", help="Path to the weights file")
        parser.add_argument("--save_dir", type=str, default="NewVGGWeights", help="Directory to save weights")
        parser.add_argument("--big_dataset", type=bool, default=False,
                            help="If want to include the big dataset in the dataloader")
        parser.add_argument("--SlowModel", type=bool, default=False, help="If want to use the slow model")
        return parser
    def __init__(self, batch_size=32,include_India=False,include_newdataset=False, SlowRun=False,**_):
        super().__init__()
        self.SlowRun = SlowRun
        self.batch_size = batch_size
        #only 1000 items for now




    def modify_batch(self, batch):
        def modify_batch(self, batch):
            # Remove all tensors from the batch that are all zeros or have low amplitudes
            filtered_batch = [tensor for tensor in batch if torch.sum(tensor) != 0]

            return filtered_batch

        return batch

    def train_dataloader(self):
        # Use the modify_batch function to add noise to the batch
        return DataLoader(AccentHuggingBased(batch_size=self.batch_size, data_type="train",SlowRun=self.SlowRun), batch_size=1,shuffle=True)#, num_workers=4)

    def val_dataloader(self):
        return DataLoader(AccentHuggingBased(data_type="test",batch_size=self.batch_size,SlowRun=self.SlowRun), batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(AccentHuggingBased(data_type="test",batch_size=self.batch_size,SlowRun=self.SlowRun), batch_size=1, shuffle=False)

    def get_dataloader(self):
        return self.train_dataloader()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch

    def prepare_data(self):
        pass

    def prepare_data_per_node(self):
        pass

    def setup(self, stage=None):
        pass
    def _log_hyperparams(self):
        pass

    def _spectogram_to_audio(self, spectrogram, sr, target_amplitude):
        """
        Convert a spectrogram back to audio.

        Args:
            spectrogram (np.ndarray): Input spectrogram.
            sr (int): Sample rate of the audio.
            target_amplitude (float): Target amplitude of the audio.

        Returns:
            np.ndarray: The reconstructed audio waveform.
        """
        # Convert the Mel-spectrogram back to linear scale
        S = librosa.db_to_power(spectrogram, ref=np.max)

        # Invert the Mel-spectrogram to obtain audio waveform
        audio_segment = librosa.feature.inverse.mel_to_audio(S, sr=sr, n_fft=2048, hop_length=512, win_length=2048,
                                                             window='hann')

        # Normalize the audio segment to match the target amplitude
        audio_segment *= target_amplitude / np.max(np.abs(audio_segment))

        return audio_segment

    def process_and_concat_audio(self, spectrograms_list, sr_list, max_amplitudes_list, segment_duration=4):
        """
        Process a list of spectrograms into audio and concatenate them.

        Args:
            spectrograms_list (list): List of input spectrograms.
            sr_list (list): List of sample rates corresponding to the input spectrograms.
            max_amplitudes_list (list): List of max amplitudes corresponding to the input spectrograms.
            segment_duration (float): Duration of each segment in seconds. Default is 4.

        Returns:
            tuple: A tuple containing the concatenated audio waveform and the sample rate.
        """
        concatenated_audio = np.array([])
        for spectrograms, sr, max_amplitudes in zip(spectrograms_list, sr_list, max_amplitudes_list):
            for spectrogram, max_amplitude in zip(spectrograms, max_amplitudes):
                audio_segment = self._spectrogram_to_audio(spectrogram, sr, max_amplitude)
                concatenated_audio = np.append(concatenated_audio, audio_segment)
        return concatenated_audio, sr  # Return the concatenated audio and sample rate

    def _audio_to_wav(self, audio, sr, path):
        """
        Save the input audio to a .wav file.

        Args:
            audio (np.ndarray): Input audio waveform.
            sr (int): Sample rate of the audio.
            path (str): Path to save the .wav file.
        """
        librosa.output.write_wav(path, audio, sr)

