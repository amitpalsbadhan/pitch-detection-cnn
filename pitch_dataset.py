import os
import json
import torch
from torch.utils.data import Dataset
import torchaudio

class PitchDataset(Dataset):
    """
    Dataset for NSynth pitch detection.

    Expects a JSON file where each entry is a sample dict containing at least:
      - 'note_str': filename without extension
      - 'pitch': integer target label
    Audio files should be stored in `audio_dir` named `<note_str>.wav`.
    """
    def __init__(
        self,
        annotations_file: str,
        audio_dir: str,
        transformation: torch.nn.Module,
        target_sample_rate: int,
        num_samples: int,
        device: str,
    ):
        # Load JSON into list of samples
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        # Each value is a dict of sample metadata
        self.annotations = list(data.values())

        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        sample = self.annotations[index]
        note_str = sample['note_str']
        pitch = sample['pitch'] - 21

        # Build path and load
        filename = note_str + '.wav'
        file_path = os.path.join(self.audio_dir, filename)
        signal, sr = torchaudio.load(file_path)
        signal = signal.to(self.device)

        # Preprocessing
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)

        return signal, torch.tensor(pitch, dtype=torch.long, device=self.device)

    def _resample_if_necessary(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        # convert multi-channel to mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        # trim signal if too long
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        # pad signal if too short
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            padding = self.num_samples - length_signal
            signal = torch.nn.functional.pad(signal, (0, padding))
        return signal

# Example usage
if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/amit/Documents/projects/pitch-detection/data/nsynth-train/examples.json"
    AUDIO_DIR = "/home/amit/Documents/projects/pitch-detection/data/nsynth-train/audio"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = SAMPLE_RATE * 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )

    pitch_dataset = PitchDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
    )

    print(f"There are {len(pitch_dataset)} samples in the dataset.")

    signal, target = pitch_dataset[1]
    print(target)
    print(signal)