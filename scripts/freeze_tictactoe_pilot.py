"""Recreate the six frozen pilot checkpoints from the registered training recipe."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_value_isomorph.pilot_protocol import fingerprint, freeze_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "experiments/tictactoe_pilot/manifest.json").read_text())
    weights = freeze_weights(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(weights, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}  {fingerprint(weights)}")


if __name__ == "__main__":
    main()
