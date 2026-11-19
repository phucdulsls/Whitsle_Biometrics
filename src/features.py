import os
import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import librosa
from pathlib import Path

# Fix OpenMP duplicate runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import config

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extracts fixed-dimensional audio embeddings for speaker/whistler biometrics.
    
    Supports:
    - 'mfcc': 20 MFCCs + delta + delta-delta with temporal mean and std pooling (120 dimensions).
    - 'whisper': Extensible interface utilizing OpenAI's Whisper audio encoder embeddings.
    """
    
    def __init__(self, backend: str = "mfcc"):
        self.backend = backend.lower()
        if self.backend not in ["mfcc", "whisper"]:
            raise ValueError(f"Unsupported backend '{backend}'. Choose 'mfcc' or 'whisper'.")
            
        self._whisper_model = None
        self._whisper_processor = None
        
        if self.backend == "whisper":
            self._init_whisper()

    def _init_whisper(self):
        """Initializes Whisper model and feature extractor."""
        try:
            import torch
            from transformers import WhisperProcessor, WhisperModel
            
            model_id = "openai/whisper-tiny"
            print(f"[FeatureExtractor] Loading Whisper model ({model_id})...")
            self._whisper_processor = WhisperProcessor.from_pretrained(model_id)
            self._whisper_model = WhisperModel.from_pretrained(model_id).encoder
            self._whisper_model.eval()
            print("[FeatureExtractor] Whisper encoder loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(
                f"Whisper backend initialization failed: {e}. "
                "Ensure torch and transformers are installed, or use backend='mfcc'."
            ) from e

    def extract(self, audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
        """Extracts a fixed-length 1D feature vector from the input audio signal."""
        if audio is None or len(audio) == 0:
            raise ValueError("Audio signal is empty or None.")

        # Ensure audio is float32
        audio = audio.astype(np.float32)

        if self.backend == "mfcc":
            return self._extract_mfcc(audio, sr)
        elif self.backend == "whisper":
            return self._extract_whisper(audio, sr)
        else:
            raise NotImplementedError(f"Backend '{self.backend}' is not implemented.")

    def _extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extracts 120-dimensional MFCC statistical feature vector.
        
        Structure:
        - 20 MFCCs
        - 20 Delta MFCCs (1st derivative)
        - 20 Delta-Delta MFCCs (2nd derivative / acceleration)
        - Temporal pooling: mean (60) + std (60) = 120 dimensions
        """
        # Ensure minimum length for STFT (at least n_fft)
        if len(audio) < config.N_FFT:
            pad_len = config.N_FFT - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant")

        # 1. Base MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=config.N_MFCC,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            fmin=config.FMIN,
            fmax=config.FMAX,
        )  # Shape: (20, T)

        # 2. Delta & Delta-Delta
        # librosa delta requires width <= T; handle short clips safely
        width = 9 if mfccs.shape[1] >= 9 else (3 if mfccs.shape[1] >= 3 else 1)
        if width > 1:
            delta1 = librosa.feature.delta(mfccs, order=1, width=width)
            delta2 = librosa.feature.delta(mfccs, order=2, width=width)
        else:
            delta1 = np.zeros_like(mfccs)
            delta2 = np.zeros_like(mfccs)

        # 3. Stack features: (60, T)
        combined = np.vstack([mfccs, delta1, delta2])

        # 4. Temporal Pooling: Mean + Standard Deviation
        mean_feat = np.mean(combined, axis=1)  # (60,)
        std_feat = np.std(combined, axis=1)    # (60,)

        # 5. Concatenate to 1D vector: (120,)
        feature_vector = np.concatenate([mean_feat, std_feat]).astype(np.float32)

        # Replace any potential NaN or Inf with 0.0
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_vector

    def _extract_whisper(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extracts deep temporal embedding using OpenAI's Whisper Encoder."""
        import torch

        # Whisper expects 16kHz audio
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = self._whisper_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features

        with torch.no_grad():
            encoder_outputs = self._whisper_model(input_features)
            # Hidden states: (1, T_frames, hidden_dim)
            hidden_states = encoder_outputs.last_hidden_state.squeeze(0)  # (T, hidden_dim)
            # Mean pooling over time frames
            embedding = torch.mean(hidden_states, dim=0).cpu().numpy().astype(np.float32)

        return np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
