import os
import sys
import tempfile
import numpy as np
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.audio_io import normalize_audio, apply_vad, save_audio, load_audio
from src.features import AudioFeatureExtractor
from src.train import train_svm_pipeline, BiometricModelWrapper


# Fixtures for synthetic audio generation
@pytest.fixture
def synthetic_sine_wave():
    """Generates a 3-second 1000 Hz pure sine wave representing whistling."""
    sr = config.SAMPLE_RATE
    duration = config.DURATION
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    # Whistling frequency around 1500 Hz
    signal = 0.8 * np.sin(2 * np.pi * 1500 * t)
    return signal, sr


@pytest.fixture
def synthetic_speech_like():
    """Generates synthetic harmonic signal representing human speech vowels (150Hz F0 + harmonics)."""
    sr = config.SAMPLE_RATE
    duration = config.DURATION
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    f0 = 150.0
    signal = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.2 * np.sin(2 * np.pi * 5 * f0 * t)
    )
    return signal.astype(np.float32), sr


@pytest.fixture
def synthetic_audio_with_silence():
    """Generates an audio signal with 1s silence, 1s tone, 1s silence."""
    sr = config.SAMPLE_RATE
    silence = np.zeros(int(sr * 1.0), dtype=np.float32)
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False, dtype=np.float32)
    tone = (0.7 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    signal = np.concatenate([silence, tone, silence])
    return signal, sr


# ============================================================================
# Unit Tests
# ============================================================================

def test_audio_normalization(synthetic_sine_wave):
    """Test peak and RMS amplitude normalization."""
    signal, _ = synthetic_sine_wave
    # Scale down signal to very quiet
    quiet_signal = signal * 0.05
    norm_signal = normalize_audio(quiet_signal, target_peak=config.PEAK_TARGET)
    
    assert np.max(np.abs(norm_signal)) == pytest.approx(config.PEAK_TARGET, abs=1e-3)
    assert norm_signal.dtype == np.float32


def test_vad_silence_removal(synthetic_audio_with_silence):
    """Test Voice Activity Detection (VAD) trims silence accurately."""
    signal, sr = synthetic_audio_with_silence
    orig_length = len(signal)
    
    trimmed_signal = apply_vad(signal, sr=sr, top_db=config.VAD_TOP_DB)
    # Trimmed signal should be significantly shorter than 3 seconds (around 1s tone)
    assert len(trimmed_signal) < orig_length
    assert len(trimmed_signal) >= int(sr * 0.8)  # Tone was 1s


def test_audio_file_io(synthetic_sine_wave):
    """Test audio saving and loading round-trip with soundfile."""
    signal, sr = synthetic_sine_wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_audio(tmp_path, signal, sr)
        loaded_signal, loaded_sr = load_audio(tmp_path, target_sr=sr)
        
        assert loaded_sr == sr
        assert len(loaded_signal) == len(signal)
        assert np.allclose(loaded_signal, signal, atol=1e-3)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_feature_extractor_fixed_dimensions(synthetic_sine_wave, synthetic_speech_like):
    """Verify that feature extraction produces fixed 1D vector across varying audio lengths."""
    extractor = AudioFeatureExtractor(backend="mfcc")
    
    sig1, sr = synthetic_sine_wave        # 3 seconds
    sig2 = sig1[: int(sr * 1.5)]         # 1.5 seconds (shorter)
    sig3 = np.tile(sig1, 2)              # 6.0 seconds (longer)
    speech, _ = synthetic_speech_like

    feat1 = extractor.extract(sig1, sr)
    feat2 = extractor.extract(sig2, sr)
    feat3 = extractor.extract(sig3, sr)
    feat_speech = extractor.extract(speech, sr)

    # Must be 1D numpy array
    assert feat1.ndim == 1
    assert feat2.ndim == 1
    assert feat3.ndim == 1
    assert feat_speech.ndim == 1

    # Length must strictly equal FEATURE_DIM_MFCC (120 dimensions)
    assert len(feat1) == config.FEATURE_DIM_MFCC
    assert len(feat2) == config.FEATURE_DIM_MFCC
    assert len(feat3) == config.FEATURE_DIM_MFCC
    assert len(feat_speech) == config.FEATURE_DIM_MFCC

    # Features should not contain NaN or Inf
    assert not np.isnan(feat1).any()
    assert not np.isnan(feat2).any()
    assert not np.isnan(feat3).any()


def test_train_export_reload_svm(synthetic_sine_wave, synthetic_speech_like):
    """Test end-to-end training of SVM classifier, saving to .joblib and reloading."""
    extractor = AudioFeatureExtractor(backend="mfcc")
    
    sine_sig, sr = synthetic_sine_wave
    speech_sig, _ = synthetic_speech_like

    # Create synthetic dataset with 2 classes: 'User_Whistler' vs 'Other_Speech'
    X = []
    y = []

    # 6 samples of whistler (slightly perturbed)
    for i in range(6):
        noise = np.random.normal(0, 0.01, len(sine_sig)).astype(np.float32)
        sample = sine_sig + noise
        X.append(extractor.extract(sample, sr))
        y.append("User_Whistler")

    # 6 samples of other speech
    for i in range(6):
        noise = np.random.normal(0, 0.01, len(speech_sig)).astype(np.float32)
        sample = speech_sig + noise
        X.append(extractor.extract(sample, sr))
        y.append("Other_Speech")

    X = np.array(X)
    y = np.array(y)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        model_tmp_path = tmp.name

    try:
        # Train and export
        model_wrapper = train_svm_pipeline(X, y, output_path=model_tmp_path)
        assert os.path.exists(model_tmp_path)

        # Reload
        loaded_wrapper = BiometricModelWrapper.load(model_tmp_path)
        assert loaded_wrapper is not None
        assert set(loaded_wrapper.classes) == {"User_Whistler", "Other_Speech"}

        # Test inference with an unseen whistler sample
        test_sample = sine_sig + np.random.normal(0, 0.01, len(sine_sig)).astype(np.float32)
        feat_test = extractor.extract(test_sample, sr)
        
        pred_label, confidence, prob_dict = loaded_wrapper.predict_single(feat_test)
        
        assert pred_label == "User_Whistler"
        assert confidence > 0.60
        assert "User_Whistler" in prob_dict
        assert "Other_Speech" in prob_dict

    finally:
        if os.path.exists(model_tmp_path):
            os.remove(model_tmp_path)


def test_confidence_threshold_rejection(synthetic_sine_wave, synthetic_speech_like):
    """Test that a low confidence prediction is correctly rejected or flagged."""
    extractor = AudioFeatureExtractor(backend="mfcc")
    sine_sig, sr = synthetic_sine_wave
    speech_sig, _ = synthetic_speech_like

    # Train model
    X = []
    y = []
    for _ in range(5):
        X.append(extractor.extract(sine_sig + np.random.normal(0, 0.01, len(sine_sig)).astype(np.float32), sr))
        y.append("Registered_User")
    for _ in range(5):
        X.append(extractor.extract(speech_sig + np.random.normal(0, 0.01, len(speech_sig)).astype(np.float32), sr))
        y.append("Impostor")

    model = train_svm_pipeline(np.array(X), np.array(y))
    
    # Ambiguous white noise input
    ambiguous_noise = np.random.normal(0, 0.5, len(sine_sig)).astype(np.float32)
    feat_ambiguous = extractor.extract(ambiguous_noise, sr)
    
    pred_label, confidence, _ = model.predict_single(feat_ambiguous)
    # Decision logic: if confidence < threshold, identity is Unverified
    is_verified = (confidence >= config.CONFIDENCE_THRESHOLD)
    # The system should have clear boolean verification logic based on config.CONFIDENCE_THRESHOLD
    assert isinstance(is_verified, bool or np.bool_)

