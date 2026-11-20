"""Export the validation-selected IQL actor and its R1 Lite runtime configuration."""

import argparse
import json
from pathlib import Path
import tempfile

import torch


def export(checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "policy"
        stage.mkdir()
        torch.save(saved["actor"], stage / "actor.pt")
        metadata = {**saved["metadata"], "checkpoint_step": saved["step"], "val_normalized_action_mae": saved["val_mae"]}
        (stage / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
