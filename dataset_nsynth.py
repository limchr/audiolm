"""
@Project: audiolm
@File: dataset_nsynth.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2024/08/06
"""

import numpy as np
import torchaudio
import torch
import glob
import os
import json
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple


class AudioLoader:
    def __init__(
            self,
            sample_rate: int = 16000,
            hop_size: int = 256,
            n_frames: int = 64,
            trim_silence: bool = False,
            trim_silence_top_db: int = 10,
            trim_silence_frame_size: int = 1024,
            trim_silence_front_only: bool = True,
            crop_mode: str = 'front',
    ) -> None:
        self.sample_rate = sample_rate
        self.audio_length = n_frames * hop_size - 1
        self.trim_silence = trim_silence
        self.trim_silence_top_db = trim_silence_top_db
        self.trim_silence_frame_size = trim_silence_frame_size
        self.trim_silence_hop_size = hop_size
        self.trim_silence_front_only = trim_silence_front_only
        self.crop_mode = crop_mode

    def _load(self, fname: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(fname)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        return waveform.squeeze(0)  # (L,)

    def _silence_trimming(self, x: torch.Tensor) -> torch.Tensor:
        non_silent_indices = torchaudio.transforms.Vad(sample_rate=self.sample_rate)(x)
        if self.trim_silence_front_only:
            x = x[non_silent_indices[0]:]
        else:
            x = torch.cat([x[interval[0]:interval[1]] for interval in non_silent_indices])
        return x

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        if len(x) < self.audio_length:
            return x
        if self.crop_mode == 'front':
            return x[:self.audio_length]
        if self.crop_mode == 'random':
            start_idx = np.random.randint(0, len(x) - self.audio_length + 1)
            return x[start_idx:start_idx + self.audio_length]
        raise ValueError(f"Unknown crop mode: {self.crop_mode}")

    def _zero_padding(self, x: torch.Tensor) -> torch.Tensor:
        if len(x) < self.audio_length:
            return torch.nn.functional.pad(x, (0, self.audio_length - len(x)))
        return x

    def _fade_out(self, x: torch.Tensor) -> torch.Tensor:
        fade_length = int(0.3 * len(x))
        fade_curve = torch.logspace(1, 0, steps=fade_length)
        fade_curve = (fade_curve - fade_curve.min()) / (fade_curve.max() - fade_curve.min())
        x[-fade_length:] *= fade_curve
        return x

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        max_val = torch.max(torch.abs(x))
        return x / max_val if max_val > 0 else x

    def __call__(self, fname: str) -> torch.Tensor:
        x = self._load(fname)
        if self.trim_silence:
            x = self._silence_trimming(x)
        x = self._crop(x)
        x = self._zero_padding(x)
        x = self._fade_out(x)
        x = self._normalize(x)
        return x


class NSynthDataset(Dataset):
    def __init__(
            self,
            split_dir: str,  # ex.) "../NSynthDataset/nsynth-train"
            sample_rate: int = 16000,
            hop_size: int = 256,
            n_frames: int = 64,
            trim_silence: bool = False,
            trim_silence_top_db: int = 10,
            trim_silence_frame_size: int = 1024,
            trim_silence_front_only: bool = True,
            crop_mode: str = 'front',
            return_audio: bool = True,
            filter_conditions: Dict[str, List[Any]] = None,
            return_attribute_keys: List[str] = None,
            return_attribute_converters: Dict[str, Any] = None,
    ):
        super().__init__()
        self.return_audio = return_audio
        self.filter_conditions = filter_conditions
        self.return_attribute_keys = return_attribute_keys
        self.return_attribute_converters = return_attribute_converters

        self.audio_loader = AudioLoader(
            sample_rate=sample_rate,
            hop_size=hop_size,
            n_frames=n_frames,
            trim_silence=trim_silence,
            trim_silence_top_db=trim_silence_top_db,
            trim_silence_frame_size=trim_silence_frame_size,
            trim_silence_front_only=trim_silence_front_only,
            crop_mode=crop_mode,
        )

        # Load audio filenames
        self.audio_fnames = sorted(glob.glob(os.path.join(split_dir, 'audio', '*.wav')))

        # Load attributes
        with open(os.path.join(split_dir, 'examples.json'), 'r') as f:
            metadata = json.load(f)
        self.attributes = [metadata[key] for key in sorted(metadata.keys())]

        # Apply filters
        self._filter_out()

    def _compute_valid_indices(self) -> np.ndarray:
        valid_mask = np.ones(len(self.audio_fnames), dtype=bool)
        if self.filter_conditions:
            for key, cond in self.filter_conditions.items():
                valid_mask &= np.array([
                    self._get_attribute(idx, key, False) in cond for idx in range(len(valid_mask))
                ])
        return np.where(valid_mask)[0]

    def _filter_out(self):
        valid_indices = self._compute_valid_indices()
        self.audio_fnames = [self.audio_fnames[i] for i in valid_indices]
        self.attributes = [self.attributes[i] for i in valid_indices]

    def _get_audio_wave(self, idx: int) -> torch.Tensor:
        return self.audio_loader(self.audio_fnames[idx])

    def _get_attribute(self, idx: int, key: str, use_converter: bool = True) -> Any:
        if key == 'audio_fname':
            return self.audio_fnames[idx]
        value = self.attributes[idx][key]
        if use_converter and self.return_attribute_converters and key in self.return_attribute_converters:
            value = self.return_attribute_converters[key](value)
        return value

    def __len__(self) -> int:
        return len(self.audio_fnames)

    def __getitem__(self, idx: int) -> Tuple:
        result = ()
        if self.return_audio:
            result += (self._get_audio_wave(idx),)
        if self.return_attribute_keys:
            result += tuple(self._get_attribute(idx, key) for key in self.return_attribute_keys)
        return result

