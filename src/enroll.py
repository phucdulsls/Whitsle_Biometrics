import os
import sys
import argparse
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

# Fix OpenMP duplicate runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import config
from src.audio_io import record_audio, normalize_audio, apply_vad, save_audio, list_microphones


def enroll_identity(
    user_name: str,
    num_samples: int = 5,
    duration: float = config.DURATION,
    output_dir: str = str(config.DATA_DIR),
    device: int = None,
):
    """Interactively records audio samples to enroll an identity into the dataset."""
    target_dir = Path(output_dir) / user_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  BIOMETRIC ENROLLMENT FOR: [{user_name}]")
    print(f"  Samples to record: {num_samples} | Duration per sample: {duration:.1f}s")
    print(f"  Destination directory: {target_dir}")
    print("=" * 60)
    print("Instructions:")
    print("  - Speak your passphrase clearly, or whistle your signature melody.")
    print("  - Maintain a consistent distance from the microphone (~15-30 cm).")
    print()

    # Determine starting sample index
    existing_files = list(target_dir.glob("sample_*.wav"))
    current_idx = len(existing_files) + 1

    recorded_count = 0
    while recorded_count < num_samples:
        sample_num = recorded_count + 1
        input(f"--> [{sample_num}/{num_samples}] Press [Enter] and get ready to speak/whistle...")
        
        print("    Recording starting in 1 second...")
        time.sleep(0.5)

        try:
            raw_audio = record_audio(
                duration=duration,
                sample_rate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                device=device,
            )

            # Check if signal is virtually silence
            peak = np.max(np.abs(raw_audio))
            if peak < 0.02:
                print("    [WARNING] Recorded signal was very quiet or silent! Would you like to retry?")
                retry = input("    Retry this sample? (y/n, default y): ").strip().lower()
                if retry != "n":
                    continue

            # Normalize amplitude
            norm_audio = normalize_audio(raw_audio)

            # Save file
            file_name = f"sample_{current_idx:02d}.wav"
            file_path = target_dir / file_name
            save_audio(str(file_path), norm_audio, sr=config.SAMPLE_RATE)
            print(f"    [Saved] Successfully recorded: {file_path.name}")

            current_idx += 1
            recorded_count += 1
            print()

        except Exception as e:
            print(f"    [Error] Recording failed: {e}")
            retry = input("    Retry? (y/n): ").strip().lower()
            if retry != "y":
                break

    print("=" * 60)
    print(f"Enrollment completed for [{user_name}]! Total samples in directory: {len(list(target_dir.glob('*.wav')))}")
    print("Next step: Train the model by running:")
    print("  python src/train.py")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Enroll voice or whistle biometric samples")
    parser.add_argument("--user", type=str, required=False, help="Identity label to enroll (e.g., 'Owner', 'Impostor')")
    parser.add_argument("--samples", type=int, default=5, help="Number of audio samples to record")
    parser.add_argument("--duration", type=float, default=config.DURATION, help="Recording duration in seconds")
    parser.add_argument("--output-dir", type=str, default=str(config.DATA_DIR), help="Directory to save raw WAVs")
    parser.add_argument("--device", type=int, default=None, help="Input audio device index")
    parser.add_argument("--list-mics", action="store_true", help="List available microphones and exit")
    args = parser.parse_args()

    if args.list_mics:
        print("[Audio Devices] Available input microphones:")
        mics = list_microphones()
        if not mics:
            print("  No input devices found.")
        for idx, name, rate in mics:
            print(f"  Index {idx}: {name} (Default SR: {rate})")
        return

    if not args.user:
        parser.error("--user is required (e.g. python src/enroll.py --user MyName --samples 5)")

    enroll_identity(
        user_name=args.user,
        num_samples=args.samples,
        duration=args.duration,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
