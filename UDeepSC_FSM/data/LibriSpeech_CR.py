import os
import glob
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple

_librosa_mod = None
def _get_librosa():
    global _librosa_mod
    if _librosa_mod is None:
        try:
            import librosa
            _librosa_mod = librosa
        except ImportError:
            _librosa_mod = False
    return _librosa_mod

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False


class LibriSpeech_CR(Dataset):
    """
    LibriSpeech dataset loader for Speech Reconstruction ('sper') and classification.
    
    Lazily loads audio samples and extracts log Mel-spectrograms of shape [time_steps, num_mels] = [128, 80]
    to feed directly into SPTEncoder_FSM and UDeepSC reconstruction pipeline.
    
    For reconstruction (if_class=False):
        __getitem__ returns (mel_spec, mel_spec)
    For classification (if_class=True):
        __getitem__ returns (mel_spec, speaker_label)
    """

    def __init__(
        self,
        root: Optional[str] = None,
        train: bool = True,
        split_ratio: float = 0.8,
        num_mels: int = 80,
        time_steps: int = 128,
        sr: int = 16000,
        hop_length: int = 512,
        n_fft: int = 1024,
        if_class: bool = False,
        max_samples: Optional[int] = None,
        preload_all: bool = False
    ):
        super().__init__()
        self.train = train
        self.num_mels = num_mels
        self.time_steps = time_steps
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.if_class = if_class
        self.preload_all = preload_all
        self._memory_cache = {}

        # Resolve root directory
        self.root = self._resolve_root(root)
        
        # Discover audio files and split by speaker
        file_list, speaker_list = self._collect_files(self.root, split_ratio, train)
        if max_samples and max_samples > 0:
            file_list = file_list[:max_samples]
            speaker_list = speaker_list[:max_samples]

        self.file_list = file_list
        self.speaker_list = speaker_list

        unique_speakers = sorted(list(set(speaker_list))) if speaker_list else []
        self.speaker_to_idx = {spk: i for i, spk in enumerate(unique_speakers)}
        self.labels = [self.speaker_to_idx.get(spk, 0) for spk in speaker_list]

        if len(self.file_list) == 0:
            warnings.warn(f"[LibriSpeech_CR] No audio files found in {self.root}. Using synthetic fallback.")
            self.synthetic_count = 100 if train else 20
        else:
            self.synthetic_count = 0

        # Optional preload if explicitly requested
        if self.preload_all and len(self.file_list) > 0:
            for idx in range(len(self.file_list)):
                self._memory_cache[idx] = self._process_audio(self.file_list[idx])

    def _resolve_root(self, root: Optional[str]) -> str:
        candidates = []
        if root:
            candidates.extend([
                root,
                os.path.join(root, "speech", "LibriSpeech", "dev-clean"),
                os.path.join(root, "speech", "LibriSpeech"),
                os.path.join(root, "LibriSpeech", "dev-clean"),
                os.path.join(root, "LibriSpeech"),
                os.path.join(root, "dev-clean")
            ])
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.extend([
            os.path.join(pkg_dir, "speech", "LibriSpeech", "dev-clean"),
            os.path.join(pkg_dir, "speech", "LibriSpeech"),
            os.path.join(pkg_dir, "..", "data", "speech", "LibriSpeech", "dev-clean"),
            os.path.join(pkg_dir, "..", "data", "speech", "LibriSpeech"),
            os.path.join(os.getcwd(), "UDeepSC_FSM", "data", "speech", "LibriSpeech", "dev-clean"),
            os.path.join(os.getcwd(), "data", "speech", "LibriSpeech", "dev-clean")
        ])

        for path in candidates:
            if os.path.exists(path):
                flacs = glob.glob(os.path.join(path, "**", "*.flac"), recursive=True)
                if len(flacs) > 0:
                    return path

        return candidates[0]

    def _collect_files(self, root_dir: str, split_ratio: float, train: bool) -> Tuple[List[str], List[str]]:
        flac_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.flac"), recursive=True))
        if not flac_files:
            return [], []

        # Group by speaker id (the parent directory name or filename prefix)
        speaker_map = {}
        for f in flac_files:
            parts = os.path.normpath(f).split(os.sep)
            spk_id = parts[-3] if len(parts) >= 3 and parts[-3].isdigit() else parts[-1].split("-")[0]
            speaker_map.setdefault(spk_id, []).append(f)

        sorted_speakers = sorted(list(speaker_map.keys()))
        n_train = max(1, int(len(sorted_speakers) * split_ratio))
        
        selected_speakers = sorted_speakers[:n_train] if train else sorted_speakers[n_train:]
        selected_files = []
        selected_labels = []

        for spk in selected_speakers:
            for f in speaker_map[spk]:
                selected_files.append(f)
                selected_labels.append(spk)

        return selected_files, selected_labels

    def _process_audio(self, audio_path: str) -> torch.FloatTensor:
        try:
            librosa = _get_librosa()
            if librosa:
                y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
                mel = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=self.num_mels,
                    hop_length=self.hop_length, n_fft=self.n_fft
                )
                log_mel = librosa.power_to_db(mel, ref=np.max)
            elif _HAS_SOUNDFILE:
                data, sr = sf.read(audio_path)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                n_fft = self.n_fft
                hop = self.hop_length
                window = np.hanning(n_fft)
                num_frames = 1 + (len(data) - n_fft) // hop
                if num_frames <= 0:
                    data = np.pad(data, (0, n_fft))
                    num_frames = 1
                frames = np.lib.stride_tricks.sliding_window_view(
                    data[:num_frames * hop + (n_fft - hop)],
                    window_shape=n_fft
                )[::hop]
                spec = np.abs(np.fft.rfft(frames * window, n=n_fft)).T
                log_mel = np.log(np.maximum(spec[:self.num_mels, :], 1e-6))
            else:
                return self._generate_synthetic_spec()

            # Normalize log-mel to [-1, 1]
            min_val = log_mel.min()
            max_val = log_mel.max()
            norm_mel = (log_mel - min_val) / (max_val - min_val + 1e-8) * 2.0 - 1.0

            # Crop or pad along time axis to time_steps
            if norm_mel.shape[1] >= self.time_steps:
                cropped = norm_mel[:, :self.time_steps]
            else:
                pad_width = self.time_steps - norm_mel.shape[1]
                cropped = np.pad(norm_mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1.0)

            # Transpose to [time_steps, num_mels] = [128, 80]
            spec_out = cropped.T.astype(np.float32)
            return torch.from_numpy(spec_out)

        except Exception as e:
            warnings.warn(f"[LibriSpeech_CR] Audio loading error on {audio_path}: {e}")
            return self._generate_synthetic_spec()

    def _generate_synthetic_spec(self, seed: int = 0) -> torch.FloatTensor:
        t = np.linspace(0, 4 * np.pi, self.time_steps)
        f = np.linspace(0, 10, self.num_mels)
        T, F = np.meshgrid(t, f)
        spec = (np.sin(T * 2 + F + seed * 0.1) * np.exp(-F / 5.0) + 0.3 * np.cos(T * 5 - F * 2))
        spec = np.clip(spec, -1.0, 1.0).astype(np.float32)
        # Transpose [80, 128] -> [128, 80]
        return torch.from_numpy(spec.T)

    def __len__(self) -> int:
        if self.synthetic_count > 0:
            return self.synthetic_count
        return len(self.file_list)

    def __getitem__(self, index: int):
        if self.synthetic_count > 0:
            spec = self._generate_synthetic_spec(seed=index)
            label = 0
        else:
            if index in self._memory_cache:
                spec = self._memory_cache[index]
            else:
                spec = self._process_audio(self.file_list[index])
                # Cache up to 1000 items in memory to accelerate epoch repetitions
                if len(self._memory_cache) < 1000:
                    self._memory_cache[index] = spec
            label = self.labels[index]

        if self.if_class:
            return spec, label
        else:
            return spec, spec
