"""Load local .env values without replacing exported environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def _candidate_paths(explicit: Optional[str]) -> List[Path]:
    if explicit:
        return [Path(explicit)]
    # Stop at the project root even when Git metadata has been removed.
    here = Path.cwd()
    candidates = []
    for directory in (here, *here.parents):
        candidates.append(directory / ".env")
        if (directory / ".git").exists() or (
            (directory / "setup.py").is_file() and (directory / "src/open_r1").is_dir()
        ):
            break
    return candidates


def load_dotenv(path: Optional[str] = None, override: bool = False) -> List[str]:
    """Load the first local .env file and return the names of imported keys."""
    loaded: List[str] = []
    for candidate in _candidate_paths(path):
        if not candidate.is_file():
            continue
        try:
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.lower().startswith("export "):
                    line = line[len("export "):].strip()
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not key:
                    continue
                if override or key not in os.environ:
                    os.environ[key] = value
                    loaded.append(key)
        except Exception:
            # A malformed .env must never crash the loop; just skip it.
            continue
        break  # first existing .env wins
    return loaded
