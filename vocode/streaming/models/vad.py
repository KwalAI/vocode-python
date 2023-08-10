import torch
import numpy as np
import io
import torchaudio
import scipy as sc


class VADProcessor:
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VADProcessor, cls).__new__(cls)
            cls._instance._init_instance()
        return cls._instance

    def _init_instance(self):
        self.model = None
        self.get_speech_timestamps = None
        self.read_audio = None
        self.count = 0
        self.buffer = None

    def download_and_load_vad_model(self):
        if self.model is None:
            # Pre-download the model and utilities if not already downloaded
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
            )
            (
                self.get_speech_timestamps,
                _,
                self.read_audio,
                self.VADIterator,
                collect_chunks,
            ) = utils

    def Int2Float(sound):
        _sound = np.copy(sound)  #
        abs_max = np.abs(_sound).max()
        _sound = _sound.astype("float32")
        if abs_max > 0:
            _sound *= 1 / abs_max
        audio_float32 = torch.from_numpy(_sound.squeeze())
        return audio_float32

    def bytes_to_audio_tensor(self, audio_bytes: bytes) -> torch.Tensor:
        bytes_io = io.BytesIO()
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int32)
        sc.io.wavfile.write(bytes_io, 8000, raw_data)
        audio, _ = torchaudio.load(bytes_io)
        return audio.squeeze(0)

    def process_audio(self, audio_bytes):
        if self.count == 30:
            audio_bytes = self.buffer + audio_bytes
            self.count = 0
        else:
            if self.buffer == None or self.count == 0:
                self.buffer = audio_bytes
            else:
                self.buffer = self.buffer + audio_bytes
            self.count += 1
            return

        if self.model is None or self.get_speech_timestamps is None:
            raise ValueError(
                "Model and utilities have not been loaded. Call download_and_load_vad_model() first."
            )
        # audio_int16 = np.frombuffer(audio_bytes, np.int16)

        # audio_float32 = self.int2float(audio_int16)

        waveform = self.bytes_to_audio_tensor(audio_bytes)
        new_confidence = self.model(waveform, 8000).item()

        # speech_prod = self.model(waveform, 8000).item()
        # # print("prod", speech_prod)

        # vad_iterator = self.VADIterator(self.model, sampling_rate=8000)
        # speech_dict = vad_iterator(waveform, return_seconds=True)
        # if speech_dict:
        #     print(speech_dict)

        # newsound = np.frombuffer(bytearray(audio_bytes), np.int16)
        # audio_float32 = self.Int2Float(newsound)

        time_stamps = self.get_speech_timestamps(
            waveform,
            self.model,
            sampling_rate=8000,
        )
        print(time_stamps)


# # Create an instance of VADProcessor
# vad_processor = VADProcessor()

# # Pre-download the model and utilities
# vad_processor.download_and_load_vad_model()

# # Load your audio data and sample rate
# audio_bytes = b""  # Replace with your audio data in bytes
# sample_rate = 16000  # Adjust according to your audio data sample rate

# # Process audio using the class method
# timestamps = vad_processor.process_audio(audio_bytes, sample_rate)

# # Print speech timestamps
# for start, end in timestamps:
#     print(f"Speech detected from {start:.2f} to {end:.2f} seconds")
