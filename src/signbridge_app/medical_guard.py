from __future__ import annotations

import re


_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "satu": "1", "dua": "2", "tiga": "3", "empat": "4",
    "lima": "5", "enam": "6", "tujuh": "7", "lapan": "8",
    "sembilan": "9", "sepuluh": "10",
}
_NEGATIONS = {"no", "not", "never", "without", "tak", "tidak", "jangan"}
_UNITS = {"mg", "g", "mcg", "ml", "l", "tablet", "tablets", "pill", "pills"}
_MEDICATION_STOP = _NEGATIONS | _UNITS | {
    "take", "use", "ambil", "makan", "the", "a", "an", "after", "before",
    "with", "meal", "meals", "daily", "day", "once", "twice",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:-[a-z]+)?|\d+(?:\.\d+)?", text.casefold())


def _numbers(words: list[str]) -> tuple[str, ...]:
    return tuple(_NUMBER_WORDS.get(word, word) for word in words if word.isdigit() or word in _NUMBER_WORDS)


def _medications(words: list[str]) -> tuple[str, ...]:
    found: list[str] = []
    for index, word in enumerate(words[:-1]):
        if word not in {"take", "taking", "use", "using", "ambil"}:
            continue
        for candidate in words[index + 1:index + 5]:
            if candidate in _MEDICATION_STOP or candidate.isdigit() or candidate in _NUMBER_WORDS:
                continue
            found.append(candidate)
            break
    return tuple(found)


def compare_critical_terms(source: str, proposal: str) -> tuple[str, ...]:
    """Return safety-sensitive categories changed by a proposed rewrite."""
    source_words = _words(source)
    proposal_words = _words(proposal)
    changed: list[str] = []
    if set(source_words) & _NEGATIONS != set(proposal_words) & _NEGATIONS:
        changed.append("negation")
    if _numbers(source_words) != _numbers(proposal_words):
        changed.append("number")
    if _medications(source_words) != _medications(proposal_words):
        changed.append("medication")
    if tuple(word for word in source_words if word in _UNITS) != tuple(
        word for word in proposal_words if word in _UNITS
    ):
        changed.append("unit")
    return tuple(changed)
