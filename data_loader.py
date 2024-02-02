import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import skimage.io

class DataLoader:
    def __init__(self, wav_dir, img_dir):
        """
        Initializes the DataLoader with the directory of WAV files and the target directory for images.
        """
        self.wav_dir = wav_dir
        self.img_dir = img_dir
        # Ensure img_dir exists
        os.makedirs(self.img_dir, exist_ok=True)

    def wav_to_spectrogram(self):
        """
        Converts WAV files in the specified directory to spectrogram images and saves them to the target directory.
        """
        # Iterate over each file in the WAV directory
        for filename in os.listdir(self.wav_dir):
            if filename.endswith('.wav'):
                # Construct the full file path
                wav_path = os.path.join(self.wav_dir, filename)
                # Load the audio file
                y, sr = librosa.load(wav_path)
                
                # Generate a Mel-spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                # Convert to log scale (dB)
                S_dB = librosa.power_to_db(S, ref=np.max)
                
                # Plotting
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
                plt.axis('off')
                plt.tight_layout(pad=0)


                # Save the figure with adjusted dpi to match the figsize
                # The dpi value of 100 is standard, but you may adjust it as needed for your specific figure size
                
                # Save the figure
                img_filename = filename.replace('.wav', '.png')  # Change file extension to .png
                img_path = os.path.join(self.img_dir, img_filename)
                plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()

                self.spectrogram_to_wav(S_dB, sr)

    def image_to_mel_spectrogram(image_path):
        # Load the image file
        img = skimage.io.imread(image_path)
        # Convert the image to grayscale since the color doesn't contain any additional information
        img_gray = np.mean(img, axis=2)
        # Assume the image values are in dB, convert them back to power
        # This assumes that the max value in the image corresponds to 0 dB
        S_dB = img_gray / img_gray.max() * -80  # Adjust this scaling factor as needed
        S = librosa.db_to_power(S_dB)

        return S

    def mel_spectrogram_to_audio(M, sr=22050, hop_length=512, n_fft=2048):
        # Convert Mel spectrogram back to STFT
        S_inv = librosa.feature.inverse.mel_to_stft(M, sr=sr, n_fft=n_fft)
        # Use the Griffin-Lim algorithm to estimate the phase
        y_inv = librosa.griffinlim(S_inv, n_iter=32, hop_length=hop_length, win_length=n_fft)
        return y_inv
    

    def reconstruct_wav():
            # Replace 'path_to_image.png' with the actual path to your Mel spectrogram image
        image_path = 'path_to_image.png'

        # The sampling rate and other parameters must match those that were used to create the Mel spectrogram
        sr = 22050  # Sampling rate
        hop_length = 512
        n_fft = 2048

        # Convert the image to a Mel spectrogram
        mel_spectrogram = image_to_mel_spectrogram(image_path)

        # Convert the Mel spectrogram to audio
        reconstructed_audio = mel_spectrogram_to_audio(mel_spectrogram, sr=sr, hop_length=hop_length, n_fft=n_fft)

        # Save the reconstructed audio to a WAV file
        wavfile.write('reconstructed_audio.wav', sr, reconstructed_audio)

if __name__ == "__main__":
    data_loader = DataLoader(r"C:\Users\dvirpe\Desktop\github\accent2accent\dataset\test_run_wav", r"C:\Users\dvirpe\Desktop\github\accent2accent\dataset\test_run_img")
    data_loader.wav_to_spectrogram()