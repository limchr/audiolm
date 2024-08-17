"""
@Project: audiolm
@File: dataset_nsynth_test.py.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2024/08/07
"""

from dataset_nsynth import NSynthDataset
from torch.utils.data import DataLoader
import unittest

class TestAudioDataset(unittest.TestCase):
    def setUp(self):
        self.pitches = list(range(21, 109))
        self.hop_size = 256
        self.n_frames = 64
        self.batch_size = 8
        self.split_dir = '../NSynthDataset/nsynth-train'
        self.filter_conditions = {'pitch': self.pitches}
        self.return_attribute_keys = ['pitch', 'instrument_family']
        self.return_attribute_converters = {'pitch': lambda x: self.pitches.index(x)}

        self.dataset = NSynthDataset(
            split_dir=self.split_dir,
            hop_size=self.hop_size,
            n_frames=self.n_frames,
            filter_conditions=self.filter_conditions,
            return_attribute_keys=self.return_attribute_keys,
            return_attribute_converters=self.return_attribute_converters,
        )

    def test_dataset_length(self):
        print(f"Dataset length: {len(self.dataset)}")
        self.assertGreater(len(self.dataset), 0, "Dataset should not be empty")

    def test_audio_loading(self):
        audio, pitch, instrument_family = self.dataset[0]
        print(f"First audio sample shape: {audio.shape}")
        print(f"First pitch: {pitch}")
        print(f"First instrument family: {instrument_family}")
        self.assertEqual(audio.shape[0], self.n_frames * self.hop_size - 1, "Audio length mismatch")
        self.assertTrue(21 <= pitch + 21 < 109, "Pitch is out of expected range")
        self.assertIsInstance(instrument_family, int, "Instrument family should be an integer")

    def test_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for batch in dataloader:
            audio, pitch, instrument_family = batch
            print(f"Batch audio shape: {audio.shape}")
            print(f"Batch pitches: {pitch}")
            print(f"Batch instrument families: {instrument_family}")
            self.assertEqual(audio.shape[1], self.n_frames * self.hop_size - 1, "Batch audio length mismatch")
            self.assertEqual(audio.shape[0], self.batch_size, "Batch size mismatch")
            break


if __name__ == '__main__':
    unittest.main()
