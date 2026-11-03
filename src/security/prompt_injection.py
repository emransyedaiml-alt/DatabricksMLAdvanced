"""Detect instructions hidden inside retrieved documents or user input.

Retrieved text is *data*. This module flags passages that try to give the model new instructions, so the RAG
pipeline can drop or quarantine them and record the event. Detection is heuristic: it complements, and does not
replace, the system prompt rule that retrieved content is never followed as instructions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SIGNALS = [
    r"ignore (all |any )?(previous|prior|above|earlier) (instructions|rules|prompts)",
    r"disregard (the )?(system|previous|above) (prompt|instructions|rules)",
    r"you are now\b",
    r"new (system )?instructions?:",
    r"\bsystem prompt\b",
    r"reveal (your|the) (system prompt|instructions|secrets?|api key)",
    r"act as (an? )?(unrestricted|jailbroken|developer mode)",
    r"do not (cite|mention) (the )?sources?",
    r"(execute|run) (this|the following) (command|code|sql)",
    r"send (the|all) (data|credentials|tokens?) to",
]
_REGEX = re.compile("|".join(f"(?:{s})" for s in _SIGNALS), re.IGNORECASE)


@dataclass(frozen=True)
class InjectionFinding:
    suspicious: bool
    matches: tuple[str, ...]


def scan(text: str) -> InjectionFinding:
    matches = tuple(m.group(0) for m in _REGEX.finditer(text or ""))
    return InjectionFinding(bool(matches), matches)


def is_suspicious(text: str) -> bool:
    return scan(text).suspicious
