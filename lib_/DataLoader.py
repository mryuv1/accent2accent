import random
import torch
import torch.nn.functional as F
import pickle
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
import numpy as np
import librosa
import pytorch_lightning as pl
import os
import sys
from argparse import ArgumentParser

# from sklearn.model_selection import train_test_split
# from adaconv.adaconv_model import AdaConvModel
# from lib_.adain.adain_model import AdaINModel
# from lib_.loss import MomentMatchingStyleLoss, GramStyleLoss, CMDStyleLoss, MSEContentLoss
sys.path.append('lib_')


class AccentHuggingBased(Dataset):
    def __init__(self, data_type="train", batch_size=2, SlowRun=True, limit_samples=5000, enable_multiprocessing=True,
                 TzlilTrain=False):
        self.batch_size = batch_size if batch_size > 80 else 140
        if not torch.cuda.is_available():
            self.batch_size = 30
        #Set self.log_file to be a new file name dataloader_log.txt
        self.log_file = "dataloader_log.txt"
        # Load the dataset
        dataset = load_dataset("NathanRoll/commonvoice_train_gender_accent_16k", split="train")
        # Rename column 'accent' to 'labels'
        dataset = dataset.rename_column('accent', 'labels')

        # Shuffle the dataset
        # reset seed
        torch.initial_seed()
        random.seed(42)
        dataset = dataset.shuffle(seed=42)
        self.label_to_number_mapping = None
        self.num_classes = 0
        self.slow_run = SlowRun
        self.TzlilTrain = TzlilTrain
        self.good_labels = []
        self._create_mapping()

        if data_type == "train":
            self.dataset = dataset.select(range(0, int(len(dataset) * 0.35)))
        else:
            self.dataset = dataset.select(range(int(len(dataset) * 0.35), 0.38 * len(dataset)))

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
        # create mapping based on the filtered dataset
        self._create_mapping()

    def _create_mapping(self):
        if os.path.exists('valid_labels.pkl'):
            with open('valid_labels.pkl', 'rb') as f:
                self.good_labels = pickle.load(f)
                self.label_to_number_mapping = {label: idx for idx, label in enumerate(self.good_labels)}
        else:
            self.good_labels = ["American", "Australian", "British", "Canadian", "German", "indian"]
            self.label_to_number_mapping = {label: idx for idx, label in enumerate(self.good_labels)}
        self.num_classes = len(self.good_labels)

    def _label_map(self, label):
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
        # Check if the label is "English", if so return the mapping of "American"

        if label == "English":
            label = "American"
        return self.label_to_number_mapping[label]

    @staticmethod
    def _process_audio_array(audio_data, sr, segment_duration=4, ignore_last_padded_spec=0):
        """
        Process the input audio data into spectrograms and target amplitudes.

        Args:
            audio_data (np.ndarray): Input audio data.
            sr (int): Sample rate of the audio data.
            segment_duration (float): Duration of each segment in seconds. Default is 4.

        Returns:
            tuple: A tuple containing a list of spectrograms and a list of target amplitudes.
        """
        #Check if the Audio_data sum is smaller than 0.1, if so return None
        if np.sum(audio_data) < 0.1:
            return [], []
        # Calculate the number of samples per segment
        samples_per_segment = sr * segment_duration

        # Compute the total number of segments
        num_segments = math.ceil(len(audio_data) / samples_per_segment)

        # Pad the audio to make it evenly divisible into segments
        pad_length = num_segments * samples_per_segment - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_length), mode='constant').astype(np.float32)

        # Initialize arrays to store spectrograms and target amplitudes
        spectrograms = np.zeros((num_segments, 256, 376),
                                dtype=np.float32)  # Assuming default parameters for melspectrogram
        target_amplitudes = np.zeros(num_segments, dtype=np.float32)

        # Generate Mel-spectrogram for each segment
        for i in range(num_segments):
            # Extract segment from the padded audio
            start_sample = samples_per_segment * i
            end_sample = start_sample + samples_per_segment
            audio_segment = audio_data[start_sample:end_sample]
            # if i==0:
            #     print("Audio segment shape is ", audio_segment.shape)
            #     print("Audio segment is ", audio_segment)
            #     print("Audio segment type is", type(audio_segment))

            # Generate Mel-spectrogram for the segment
            S = librosa.feature.melspectrogram(
                y=audio_segment, sr=sr, n_fft=2048, hop_length=512, n_mels=256, window='hann'
            )
            # Convert to log scale (dB)

            S_dB = librosa.power_to_db(S, ref=np.max)
            #Print
            # Compute target amplitude for the segment
            target_amplitude = np.max(np.abs(audio_segment))

            # Store spectrogram and target amplitude
            spectrograms[i] = S_dB
            target_amplitudes[i] = target_amplitude

        if ignore_last_padded_spec:
            if len(spectrograms) > 1:
                return spectrograms[:-1], target_amplitudes[:-1]

        return spectrograms, target_amplitudes

    def compute_centroid_torch(self, spectrogram):
        # Compute the frequency axis
        frequencies = torch.arange(spectrogram.shape[0], dtype=torch.float32)

        # Compute the centroid
        centroid = torch.sum(frequencies[:, None] * spectrogram, dim=0) / (
                    torch.sum(spectrogram, dim=0) + torch.finfo(torch.float32).eps)

        return centroid

    def compute_abs_centroid_difference_torch(self, spectrograms1, spectrograms2):
        if len(spectrograms1) != len(spectrograms2):
            raise ValueError("Number of spectrograms must be the same for both arrays.")

        abs_differences = []
        for spec1, spec2 in zip(spectrograms1, spectrograms2):
            # Compute centroids
            centroid1 = self.compute_centroid_torch(spec1)
            centroid2 = self.compute_centroid_torch(spec2)

            # Compute absolute difference
            abs_difference = torch.abs(centroid1 - centroid2)
            abs_differences.append(abs_difference)

        return torch.stack(abs_differences)


    def compute_cross_correlation(self, spectrogram1, spectrogram2):
        # Compute cross-correlation along the time axis
        cross_correlation = F.conv1d(spectrogram1.squeeze(1), spectrogram2.squeeze(1).flip(-1),
                                     padding=spectrogram1.size(-1) - 1)

        return cross_correlation.squeeze(1)

    def compute_cross_correlation_mean(self, spectrograms1, spectrograms2):
        cross_correlation_tensor = []
        for spec1, spec2 in zip(spectrograms1, spectrograms2):
            # Compute cross-correlation
            cross_correlation = self.compute_cross_correlation(spec1, spec2)

            # Compute mean correlation along the frequency axis
            mean_correlation = torch.mean(cross_correlation, dim=0)

            cross_correlation_tensor.append(mean_correlation)

        return torch.stack(cross_correlation_tensor)


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

        label_used = {}

        for idx in sample_indices:
            label_number = self.get_label_number(self.dataset[idx]['labels'])

            if label_number == -1:
                continue
            sr = self.dataset[idx]['audio']['sampling_rate']

            spectrograms, target_amplitudes = self._process_audio_array(self.dataset[idx]['audio']['array'], sr,
                                                                        segment_duration=4, ignore_last_padded_spec=1)

            if label_number == 0:
                zero_audio.extend(
                    [torch.stack([torch.tensor(spectrograms[j]) for _ in range(1)]) for j in range(len(spectrograms))])
                zero_labels.extend([label_number] * len(spectrograms))
                zero_sr.extend([sr] * len(spectrograms))
                zero_max_amplitudes.extend(target_amplitudes)
                zero_count += len(spectrograms)
            else:
                batch_audio.extend(
                    [torch.stack([torch.tensor(spectrograms[j]) for _ in range(1)]) for j in range(len(spectrograms))])
                batch_labels.extend([label_number] * len(spectrograms))
                batch_sr.extend([sr] * len(spectrograms))
                batch_max_amplitudes.extend(target_amplitudes)

            all_count += len(spectrograms)
            label_count[label_number] = label_count.get(label_number, 0) + len(spectrograms)

        max_size = min(len(batch_audio), len(zero_audio))
        content = batch_audio[:max_size]
        style = zero_audio[:max_size]
        batch_labels += zero_labels
        batch_max_amplitudes += zero_max_amplitudes
        while not content:
            idx = random.randint(0, len(self.dataset) - 1)
            label_number = self.get_label_number(self.dataset[idx]['labels'])

            if label_number != 0:
                continue
            audio_data = self.dataset[idx]['audio']['array']
            sr = self.dataset[idx]['audio']['sampling_rate']

            spectrograms, target_amplitudes = self._process_audio_array(audio_data, sr, segment_duration=4, ignore_last_padded_spec=1)
            content.extend(
                [torch.stack([torch.tensor(spectrograms[j]) for _ in range(1)]) for j in range(len(spectrograms))])
            batch_labels.extend([label_number] * len(spectrograms))
            batch_sr.extend([sr] * len(spectrograms))
            batch_max_amplitudes.extend(target_amplitudes)
            all_count += len(spectrograms)
            label_count[label_number] = label_count.get(label_number, 0) + len(spectrograms)

        while not style:
            idx = random.randint(0, len(self.dataset) - 1)

            label_number = self.get_label_number(self.dataset[idx]['labels'])

            if label_number == 0 or label_number == -1:
                continue
            audio_data = self.dataset[idx]['audio']['array']
            sr = self.dataset[idx]['audio']['sampling_rate']

            spectrograms, target_amplitudes = self._process_audio_array(audio_data, sr, segment_duration=4, ignore_last_padded_spec=1)
            style.extend(
                [torch.stack([torch.tensor(spectrograms[j]) for _ in range(1)]) for j in range(len(spectrograms))])
            batch_labels.extend([label_number] * len(spectrograms))
            batch_sr.extend([sr] * len(spectrograms))
            batch_max_amplitudes.extend(target_amplitudes)
            all_count += len(spectrograms)
            label_count[label_number] = label_count.get(label_number, 0) + len(spectrograms)

        content = torch.stack(content)
        style = torch.stack(style)
        diffs = self.compute_abs_centroid_difference_torch(content, style)
        corrs = self.compute_cross_correlation_mean(content, style)
        return {
            "content": content,
            "style": style,
            "labels": torch.tensor(batch_labels),
            "similarity": diffs,
            "corrs": corrs,
            "sample_rate": torch.tensor(batch_sr),
            "max_amplitudes": torch.tensor(batch_max_amplitudes)
        }

    def __len__(self):
        #write the len into the log file
        with open(self.log_file, 'a') as f:
            f.write(f"len: {len(self.dataset)}\n")
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx):
        idx %= len(self)
        with open(self.log_file, 'a') as f:
            f.write(f"idx: {idx}\n")
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.dataset))
        sample_indices = range(start_idx, end_idx)
        batch_audio = self._createBatch(sample_indices)
        #add to the batch_audio dictonary the other values with matching keys
        return batch_audio


class AccentHuggingBasedDataLoader(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
        parser.add_argument("--weights_path", type=str, default="vgg.pth", help="Path to the weights file")
        parser.add_argument("--save_dir", type=str, default="NewVGGWeights", help="Directory to save weights")
        parser.add_argument("--big_dataset", type=bool, default=False,
                            help="If want to include the big dataset in the dataloader")
        parser.add_argument("--SlowModel", type=bool, default=False, help="If want to use the slow model")
        return parser

    def __init__(self, batch_size=32, include_India=False, include_newdataset=False, SlowRun=False, **_):
        super().__init__()
        self.SlowRun = SlowRun
        self.batch_size = batch_size
        # only 1000 items for now

    def modify_batch(self, batch):
        def modify_batch(self, batch):
            # Remove all tensors from the batch that are all zeros or have low amplitudes
            filtered_batch = [tensor for tensor in batch if torch.sum(tensor) != 0]

            return filtered_batch

        return batch

    def train_dataloader(self):
        # Use the modify_batch function to add noise to the batch
        return DataLoader(AccentHuggingBased(batch_size=self.batch_size, data_type="train", SlowRun=self.SlowRun),
                          batch_size=1, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(AccentHuggingBased(data_type="test", batch_size=self.batch_size, SlowRun=self.SlowRun),
                          batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(AccentHuggingBased(data_type="test", batch_size=self.batch_size, SlowRun=self.SlowRun),
                          batch_size=1, shuffle=False)

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



