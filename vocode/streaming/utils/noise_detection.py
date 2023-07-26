from pydub import AudioSegment
import numpy as np
import time
import librosa
import io
import matplotlib.pyplot as plt


class NoiseDetector:
    def __init__(
        self,
    ) -> None:
        self.dbfs_values = []
        self.threshold = -4
        self.window_size = 300
        self.detected_noise = False
        self.interrupted = False
        self.last_print_time = time.time()
        self.created = False
        self.count = 40

    def receive_audio(self, chunk: bytes):
        audio_data = np.frombuffer(chunk, dtype=np.int16)
        audio_data = librosa.util.buf_to_float(audio_data)
        n_fft = min(len(audio_data), 2048)
        stft_matrix = librosa.stft(audio_data, n_fft=n_fft)
        magnitude_spectogram = np.abs(stft_matrix)
        if time.time() - self.last_print_time > 5:
            self.count += 1
            self.last_print_time = time.time()
            fig, ax = plt.subplots()
            img = librosa.display.specshow(
                librosa.amplitude_to_db(magnitude_spectogram, ref=np.max),
                ax=ax,
                y_axis="log",
                x_axis="time",
            )
            ax.set_title("Power spectrogram")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            plt.savefig(f"spectogram{self.count}.png")
        magnitude_spectogram = np.mean(
            magnitude_spectogram, axis=1
        )  # merge across time axis into a 1 dimensional array containing magnitude of each frequency bin

        threshold = np.mean(magnitude_spectogram) + np.std(magnitude_spectogram)
        num_bins_above_threshold = np.sum(magnitude_spectogram > threshold)
        # print("num bins above threshold", num_bins_above_threshold)

        # # audio_segment = AudioSegment(
        # #     data=chunk, sample_width=2, frame_rate=16000, channels=1
        # # )
        # dbfs_value = audio_segment.dBFS
        # self.dbfs_values.append(dbfs_value)

        # if len(self.dbfs_values) > self.window_size:
        #     self.dbfs_values.pop(0)
        # moving_avg_dbfs = np.mean(self.dbfs_values)

        # if time.time() - self.last_print_time > 1:
        #     self.last_print_time = time.time()
        #     print("curr moving avg", moving_avg_dbfs, "curr dbfs", dbfs_value)

        # if moving_avg_dbfs > self.threshold and not self.detected_noise:
        #     self.detected_noise = True

    def send_noise_interrupt(self):
        return self.detected_noise and not self.interrupted

    def sent_interrupt(self):
        self.interrupted = True
