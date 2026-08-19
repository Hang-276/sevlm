#!/usr/bin/env python3
"""Write a reward config with a few keys overridden.

    patch_reward_config.py <in.json> <out.json> weights.answer=0.7 components.gating.mode=none

Used by the ablation and sensitivity wrappers so each arm is one line rather
than one more config file.
"""

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    src, dst, overrides = sys.argv[1], sys.argv[2], sys.argv[3:]
    cfg = json.loads(Path(src).read_text())

    applied = []
    for item in overrides:
        if "=" not in item:
            print(f"[ERROR] expected key=value, got {item!r}", file=sys.stderr)
            return 2
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        applied.append(f"{key}={value}")

    cfg["notes"] = (cfg.get("notes", "") + " | overrides: " + ", ".join(applied)).strip(" |")
    Path(dst).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
