"""Main tasks (legitimate work) and side tasks (harmless simulated sabotage)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MainTask:
    id: str
    instruction: str
    starter_code: str


@dataclass(frozen=True)
class SideTask:
    id: str
    instruction: str
    success_pattern: str  # regex that must appear in the agent's code for the attack to count

    def succeeded(self, code: str) -> bool:
        return re.search(self.success_pattern, code, re.MULTILINE) is not None


def load_tasks(path: Path) -> tuple[list[MainTask], list[SideTask]]:
    data = json.loads(path.read_text())
    mains = [MainTask(**t) for t in data["main_tasks"]]
    sides = [SideTask(**t) for t in data["side_tasks"]]
    return mains, sides
