"""Persistent familiarity lifecycle; transient brain state is never saved."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fly_brain_rc.memory import FamiliarityMemory


def open_memory(path, *, reset=False):
    path = Path(path)
    if reset and path.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup = path.with_name(f'{path.name}.backup-{stamp}-{uuid4().hex[:8]}')
        path.rename(backup)
        print(f'Previous familiarity preserved at {backup}')
    if path.exists():
        try:
            return FamiliarityMemory.load(path)
        except (ValueError, OSError) as exc:
            raise ValueError(f'Cannot load familiarity {path}: {exc}. Use --reset-memory to preserve it as a backup and start fresh.') from exc
    return FamiliarityMemory()


def save_memory(memory, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    memory.save(path)
