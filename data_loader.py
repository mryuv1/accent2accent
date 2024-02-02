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



class DataLoader:
    def __init__(self, content_wav_dir, style_wav_dir):
        self.content_wav_dir = content_wav_dir
        self.style_wav_dir = style_wav_dir
        self.content_spectrograms = []  # List of tuples (file_name, S)
        self.style_spectrograms = []  # List of tuples (file_name, S)

    def wav_to_spectrogram(self):
        # Process content WAV files
        for filename in os.listdir(self.content_wav_dir):
            if filename.endswith('.wav'):
                S, target_amplitude = self._process_wav_file(os.path.join(self.content_wav_dir, filename))
                self.content_spectrograms.append((filename, S, target_amplitude))

        # Process style WAV files
        for filename in os.listdir(self.style_wav_dir):
            if filename.endswith('.wav'):
                S = self._process_wav_file(os.path.join(self.style_wav_dir, filename))
                self.style_spectrograms.append((filename, S))

    def _process_wav_file(self, wav_path):
        # Load the audio file
        y, sr = librosa.load(wav_path)
        target_amplitude = np.max(np.abs(y))
        # Generate a Mel-spectrogram
        n_fft = 2048
        hop_length = 512
        n_mels = 256
        #S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann')
        # Convert to log scale (dB)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return (S_dB,target_amplitude)

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

if __name__ == "__main__":
    data_loader = DataLoader(r"C:\Users\dvirpe\Desktop\github\accent2accent\dataset\test_run_wav", r"C:\Users\dvirpe\Desktop\github\accent2accent\dataset\test_run_wav")
    data_loader.wav_to_spectrogram()
    data_loader.spectrogram_to_wav(data_loader.content_spectrograms, r"C:\Users\dvirpe\Desktop\github\accent2accent\dataset\test_run_img")
