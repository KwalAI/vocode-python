from pydub import AudioSegment
import numpy as np
import time
import librosa
import io


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

    def receive_audio(self, chunk: bytes):
        audio_data = np.frombuffer(chunk, dtype=np.int16)
        audio_data = librosa.util.buf_to_float(audio_data)
        n_fft = min(len(audio_data), 2048)
        stft_matrix = librosa.stft(audio_data, n_fft=n_fft)

        # y, sr = librosa.load(io.BytesIO(chunk), sr=None)
        # # audio_segment = AudioSegment(
        # #     data=chunk, sample_width=2, frame_rate=16000, channels=1
        # # )
        # stft_matrix = librosa.stft(y)

        magnitude_spectogram = np.abs(stft_matrix)
        magnitude_spectogram = np.mean(magnitude_spectogram, axis=1)
        average_magnitude = np.mean(magnitude_spectogram)
        num_bins_above_threshold = np.sum(magnitude_spectogram > average_magnitude)
        print("num bins above threshold", num_bins_above_threshold)

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
