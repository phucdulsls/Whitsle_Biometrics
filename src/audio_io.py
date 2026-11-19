import os
import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import soundfile as sf
import librosa

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from pathlib import Path
import config

logger = logging.getLogger(__name__)


def list_microphones():
    """Lists available audio input devices."""
    if sd is None:
        print("[AudioIO] sounddevice is not available.")
        return []
    
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            input_devices.append((i, dev.get("name"), dev.get("default_samplerate")))
    return input_devices


def record_audio(
    duration: float = config.DURATION,
    sample_rate: int = config.SAMPLE_RATE,
    channels: int = config.CHANNELS,
    device: int = None,
) -> np.ndarray:
    """Records audio from the default or specified input device.
    
    Raises:
        RuntimeError: If sounddevice is unavailable, mic is busy or no input device found.
    """
    if sd is None:
        raise RuntimeError("sounddevice is not installed or audio drivers are not functional.")
    
    try:
        # Check input device availability
        if device is None:
            default_input = sd.default.device[0]
            if default_input == -1:
                raise RuntimeError("No default audio input device detected. Please plug in a microphone.")
        
        num_samples = int(duration * sample_rate)
        print(f"[AudioIO] Recording for {duration:.1f}s at {sample_rate}Hz... Speak or whistle now!")
        
        recording = sd.rec(
            frames=num_samples,
            samplerate=sample_rate,
            channels=channels,
            dtype=config.DTYPE,
            device=device,
        )
        sd.wait()
        
        # Flatten to 1D
        audio = recording.squeeze()
        print("[AudioIO] Recording complete.")
        return audio.astype(np.float32)

    except Exception as e:
        raise RuntimeError(f"Failed to record audio from microphone: {e}") from e


def normalize_audio(audio: np.ndarray, target_peak: float = config.PEAK_TARGET) -> np.ndarray:
    """Normalizes audio to a target peak amplitude to eliminate volume and distance discrepancy."""
    if audio is None or len(audio) == 0:
        return np.array([], dtype=np.float32)
    
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = (audio / peak) * target_peak
    return audio


def apply_vad(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    top_db: float = config.VAD_TOP_DB,
    frame_length: int = config.VAD_FRAME_LENGTH,
    hop_length: int = config.VAD_HOP_LENGTH,
) -> np.ndarray:
    """Applies Voice Activity Detection (VAD) by trimming leading and trailing silence."""
    if audio is None or len(audio) == 0:
        return np.array([], dtype=np.float32)

    try:
        # Trim leading and trailing silence
        trimmed_audio, _ = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        
        # Fallback to original if trim left too little audio (< 0.2s)
        if len(trimmed_audio) < int(sr * 0.2):
            return audio.astype(np.float32)
            
        return trimmed_audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"VAD trimming encountered an error: {e}. Falling back to untrimmed audio.")
        return audio.astype(np.float32)


def save_audio(file_path: str, audio: np.ndarray, sr: int = config.SAMPLE_RATE):
    """Saves audio data to a WAV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def load_audio(file_path: str, target_sr: int = config.SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Loads an audio file and resamples to target_sr if necessary."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32), sr
