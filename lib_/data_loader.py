import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import skimage.io
from scipy.io.wavfile import write
import noisereduce as nr
import soundfile as sf
import pickle
import pytorch_lightning as pl
from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataset import files_in
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import AccentDataset, files_in, EndlessAccentDataset


class DataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # TODO CHANGE THE DEFAULT VALUES
        parser.add_argument('--content', type=str, default='./data/MSCOCO/train2017',
                            help='Directory with content images.')
        parser.add_argument('--style', type=str, default='./data/WikiArt/train',
                            help='Directory with style images.')
        parser.add_argument('--test-content', type=str, default='./test_images/content',
                            help='Directory with test content images (or path to single image). If not set, takes 5 random train content images.')
        parser.add_argument('--test-style', type=str, default='./test_images/style',
                            help='Directory with test style images (or path to single image). If not set, takes 5 random train style images.')
        parser.add_argument('--batch-size', type=int, default=8,
                            help='Training batch size.')

        return parser

    def __init__(self, curr_dir, content_wav_dir, style_wav_dir, batch_size=8, test_content=None, test_style=None, **_):
        self.content_wav_dir = content_wav_dir
        self.style_wav_dir = style_wav_dir
        self.curr_dir = curr_dir
        self.batch_size = batch_size
        self.content_spectrograms = []  # List of tuples (file_name, S)
        self.style_spectrograms = []  # List of tuples (file_name, S)
        self.tr_content_pickle_path = os.path.join(curr_dir, "", "dataset", "pickle_data", "train",
                                                   "content_spectrograms")
        self.te_content_pickle_path = os.path.join(curr_dir, "", "dataset", "pickle_data", "test",
                                                   "content_spectrograms")
        self.te_style_pickle_path = os.path.join(curr_dir, "", "dataset", "pickle_data", "test",
                                                 "style_spectrograms")
        self.tr_style_pickle_path = os.path.join(curr_dir, "", "dataset", "pickle_data", "train",
                                                 "style_spectrograms")
        # Check if the directories exists
        if self.create_dirs():
            self.wav_to_spectrogram()
            #    self.clean_spectogram_arrays()
            self.save_spectograms_to_pickle()

            # TODO - maybe will crash RAM, need maybe to do it in iterations ! WELL WORK ON BATCHES FROM MEMORY (PICKLE OR WHATEVER)
        self.train_content_files = files_in(self.tr_content_pickle_path)
        self.train_style_files = files_in(self.tr_style_pickle_path)
        self.test_content_files = files_in(self.te_content_pickle_path)
        self.test_style_files = files_in(self.te_style_pickle_path)
        #Generate random torch seed
        pl.seed_everything(43)
        self.train_dataset = AccentDataset(self.train_content_files, self.train_style_files)
        self.test_dataset = AccentDataset(self.test_content_files, self.test_style_files)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1,shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1,shuffle=True)
    def transfer_batch_to_device(self, batch, device):
        for k, v in batch.items():
            if isinstance(v, Tensor):
                batch[k] = v.to(device)
        return batch

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass


    def create_dirs(self):
        if not os.path.exists(
                os.path.join(self.curr_dir, "", "dataset", "pickle_data", "train", "content_spectrograms")):
            if not os.path.exists(os.path.join(self.curr_dir, "", "dataset", "pickle_data", "train")):
                os.makedirs(os.path.join(self.curr_dir, "", "dataset", "pickle_data", "train"))
            os.makedirs(os.path.join(self.curr_dir, "", "dataset", "pickle_data", "train", "content_spectrograms"))
            os.makedirs(os.path.join(self.curr_dir, "", "dataset", "pickle_data", "train", "style_spectrograms"))
            if not os.path.exists(
                    os.path.join(self.curr_dir, "", "dataset", "pickle_data", "test", "content_spectrograms")):
                if not os.path.exists(os.path.join(self.curr_dir, "", "dataset", "pickle_data", "test")):
                    os.makedirs(os.path.join(self.curr_dir, "", "dataset", "pickle_data", "test"))
            os.makedirs(os.path.join(self.curr_dir, "", "dataset", "pickle_data", "test", "content_spectrograms"))
            os.makedirs(os.path.join(self.curr_dir, "", "dataset", "pickle_data", "test", "style_spectrograms"))
            return True
        return False

    def wav_to_spectrogram(self):
        for directory in os.listdir(self.content_wav_dir):
            if directory.startswith('.'):
                continue
                # Iterate over the files in the directory
            for filename in os.listdir(os.path.join(self.content_wav_dir, directory)):
                file_path = os.path.join(self.content_wav_dir, directory, filename)
                # print(file_path)
                if filename.endswith('.wav'):
                    S, target_amplitude = self._process_wav_file(file_path)
                    self.content_spectrograms.append((file_path, S, target_amplitude))

        for directory in os.listdir(self.style_wav_dir):
            if directory.startswith('.'):
                continue
            for filename in os.listdir(os.path.join(self.style_wav_dir, directory)):
                file_path = os.path.join(self.style_wav_dir, str(directory), filename)
                if file_path.endswith('.wav'):
                    S, target_amplitude = self._process_wav_file(file_path)
                    self.style_spectrograms.append((file_path, S, target_amplitude))

    def _process_wav_file(self, wav_path):
        # Load the audio file
        y, sr = librosa.load(wav_path)
        target_amplitude = np.max(np.abs(y))
        # Generate a Mel-spectrogram
        n_fft = 2048
        hop_length = 512
        n_mels = 256
        # S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann')
        # Convert to log scale (dB)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return (S_dB, target_amplitude)

    def spectrogram_to_wav(self, spectrograms, output_dir, sr=22050):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename, S_dB, target_amplitude in spectrograms:
            # Convert dB back to power
            S = librosa.db_to_power(S_dB)
            # Inverse Mel spectrogram to audio
            y = librosa.feature.inverse.mel_to_audio(S, sr=sr)
            # Normalize the audio signal
            normalization_factor = target_amplitude / np.max(np.abs(y))
            y_normalized = y * normalization_factor
            # Reduce noise
            y_normalized = nr.reduce_noise(y_normalized, sr=sr)
            # Save the audio file
            new_filename = f"{os.path.splitext(filename)[0]}_converted.wav"
            output_path = os.path.join(output_dir, new_filename)
            sf.write(output_path, y_normalized, sr)

    def clean_spectogram_arrays(self):
        # Extract spectrograms from content_spectrograms
        self.content_spectrograms = [item[1] for item in self.content_spectrograms]

        # Extract spectrograms from style_spectrograms
        self.style_spectrograms = [item[1] for item in self.style_spectrograms]

    def save_spectograms_to_pickle(self, test_path=None, test_size=5.0 / 100):
        if test_path is None:
            print(self.content_spectrograms)
            train_content, test_content = train_test_split(self.content_spectrograms, test_size=test_size)
            train_style, test_style = train_test_split(self.style_spectrograms, test_size=test_size)
        else:
            train_content = self.content_spectrograms
            train_style = self.style_spectrograms
            test_content = []
            test_style = []
            # TODO ADD THE TEST FILES TO THE PICKLE

        # Save content_spectrograms_only to a pickle file - train
        if not os.path.exists(os.path.dirname(self.tr_content_pickle_path)):
            os.makedirs(os.path.dirname(self.tr_content_pickle_path))
        for file in train_content:
            file_name = str(os.path.basename(file[0]).split('.wav')[0])
            with open(os.path.join(self.tr_content_pickle_path,file_name), 'wb') as f:
                pickle.dump(file[1], f)
        if not os.path.exists(os.path.dirname(self.tr_style_pickle_path)):
            os.makedirs(os.path.dirname(self.tr_style_pickle_path))
        for file in train_style:
            file_name = str(os.path.basename(file[0]).split('.wav')[0])
            with open(os.path.join(self.tr_style_pickle_path, file_name), 'wb') as f:
                pickle.dump(file[1], f)
        if not os.path.exists(os.path.dirname(self.te_content_pickle_path)):
            os.makedirs(os.path.dirname(self.te_content_pickle_path))
        for file in test_content:
            file_name = str(os.path.basename(file[0]).split('.wav')[0])
            with open(os.path.join(self.te_content_pickle_path, file_name), 'wb') as f:
                pickle.dump(file[1], f)
        if not os.path.exists(os.path.dirname(self.te_style_pickle_path)):
            os.makedirs(os.path.dirname(self.te_style_pickle_path))
        for file in test_style:
            file_name = str(os.path.basename(file[0]).split('.wav')[0])
            with open(os.path.join(self.te_style_pickle_path, file_name), 'wb') as f:
                pickle.dump(file[1], f)


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct relative paths from the current directory
    content_dir = os.path.join(current_directory, '', 'dataset', 'content', "indian")
    style_dir = os.path.join(current_directory, '', 'dataset', 'style', "american")


    a = DataModule(current_directory, content_dir, style_dir,1)


