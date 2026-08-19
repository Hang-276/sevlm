"""Zero-dependency .env loader for the self-evolve loop.

Why this exists
---------------
A reusable, open-source-grade project must support a ``.env`` file so users can
supply credentials (e.g. the Reference VLM API key) without exporting them by
hand every session. This loader keeps that ergonomic while staying safe:

  - It loads ``KEY=VALUE`` lines into ``os.environ`` ONLY if the key is not
    already set (real environment / ``read -s`` always wins).
  - It NEVER prints values.
  - The ``.env`` file itself is gitignored (see repo ``.gitignore``) and is
    never written by this code — only read. A committed ``.env.example`` holds
    placeholders, never real secrets.

No third-party dependency (python-dotenv is not required).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def _candidate_paths(explicit: Optional[str]) -> List[Path]:
    if explicit:
        return [Path(explicit)]
    # Search cwd upward to the git repo root for a .env file.
    here = Path.cwd()
    candidates = [here / ".env"]
    for parent in here.parents:
        candidates.append(parent / ".env")
        if (parent / ".git").exists():
            break
    return candidates


def load_dotenv(path: Optional[str] = None, override: bool = False) -> List[str]:
    """Load a ``.env`` file into ``os.environ``.

    Parameters
    ----------
    path : str or None
        Explicit path to a ``.env`` file. When None, searches cwd upward to the
        repo root.
    override : bool
        When False (default), existing environment variables are preserved — a
        key exported via ``read -s`` or the shell always takes precedence over
        the file. When True, file values overwrite.

    Returns
    -------
    list[str]
        The names (NOT values) of the keys that were loaded. Safe to print/log.
    """
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
