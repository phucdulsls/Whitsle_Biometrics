import os
import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

# Fix OpenMP duplicate runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import config
from src.audio_io import record_audio, load_audio, normalize_audio, apply_vad
from src.features import AudioFeatureExtractor
from src.train import BiometricModelWrapper


def predict_audio_sample(
    audio: np.ndarray,
    sr: int,
    model_wrapper: BiometricModelWrapper,
    threshold: float = config.CONFIDENCE_THRESHOLD,
) -> dict:
    """Processes an audio sample and returns biometric identification results."""
    # 1. Amplitude Normalization
    norm_audio = normalize_audio(audio)

    # 2. Voice Activity Detection
    vad_audio = apply_vad(norm_audio, sr=sr)

    # 3. Feature Extraction
    extractor = AudioFeatureExtractor(backend=model_wrapper.backend)
    features = extractor.extract(vad_audio, sr=sr)

    # 4. Predict
    pred_label, confidence, prob_dict = model_wrapper.predict_single(features)
    is_verified = (confidence >= threshold)

    return {
        "predicted_label": pred_label,
        "confidence": confidence,
        "is_verified": is_verified,
        "probabilities": prob_dict,
        "threshold": threshold,
    }


def format_result_display(result: dict):
    """Prints a clear, formatted biometric verification result."""
    print()
    print("=" * 60)
    print("        BIOMETRIC IDENTIFICATION RESULT")
    print("=" * 60)
    
    pred = result["predicted_label"]
    conf = result["confidence"]
    verified = result["is_verified"]
    threshold = result["threshold"]

    if verified:
        print(f"  >>> STATUS: [ACCESS GRANTED - VERIFIED] <<<")
        print(f"  Identified As : {pred}")
        print(f"  Confidence    : {conf * 100:.2f}% (Threshold: {threshold * 100:.1f}%)")
    else:
        print(f"  >>> STATUS: [ACCESS DENIED - UNVERIFIED / IMPOSTOR] <<<")
        print(f"  Nearest Match : {pred}")
        print(f"  Confidence    : {conf * 100:.2f}% (Below required threshold: {threshold * 100:.1f}%)")

    print("-" * 60)
    print("  Class Probability Distribution:")
    # Sort classes by probability descending
    sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
    for cls_name, prob in sorted_probs:
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"    - {str(cls_name):<18} : {prob * 100:6.2f}% [{bar}]")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="Live Voice & Whistle Biometric Identification")
    parser.add_argument("--model", type=str, default=str(config.MODEL_PATH), help="Path to trained model artifact")
    parser.add_argument("--input", type=str, default=None, help="Path to input .wav file (if None, records from mic)")
    parser.add_argument("--duration", type=float, default=config.DURATION, help="Recording duration in seconds")
    parser.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--device", type=int, default=None, help="Microphone device index")
    args = parser.parse_args()

    # Load model artifact
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[Error] Model artifact not found at: {model_path}", file=sys.stderr)
        print("Please train a model first: python src/train.py", file=sys.stderr)
        sys.exit(1)

    print(f"[Predict] Loading biometric model from {model_path}...")
    model_wrapper = BiometricModelWrapper.load(str(model_path))
    print(f"[Predict] Model loaded successfully! Registered classes: {model_wrapper.classes}")

    # Acquire audio
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[Error] Audio file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[Predict] Loading audio from file: {input_path}")
        audio, sr = load_audio(str(input_path), target_sr=config.SAMPLE_RATE)
    else:
        print("[Predict] Ready to record from microphone.")
        input("Press [Enter] and speak or whistle for verification...")
        audio = record_audio(
            duration=args.duration,
            sample_rate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            device=args.device,
        )
        sr = config.SAMPLE_RATE

    # Perform prediction
    result = predict_audio_sample(
        audio=audio,
        sr=sr,
        model_wrapper=model_wrapper,
        threshold=args.threshold,
    )

    # Display results
    format_result_display(result)


if __name__ == "__main__":
    main()
