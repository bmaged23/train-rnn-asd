"""Run checkpoint-backed inference; prepared features are required, not raw video."""
import argparse
import json
from pathlib import Path

from src.inference import WindowPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    predictor = WindowPredictor(args.checkpoint, args.manifest, args.device)
    result = predictor.predict(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
