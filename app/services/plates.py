"""Indian plate normalisation and an independent multi-frame vote.

Not a patent implementation. Syntax is a plausibility flag, not identity.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

STANDARD = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
BH_SERIES = re.compile(r"^\d{2}BH\d{4}[A-Z]{2}$")
GJ_GOVT = re.compile(r"^GJ18G{1,2}\d{4}$")


def normalize(raw: str | None) -> str:
    text = unicodedata.normalize("NFKC", raw or "")
    text = text.upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def syntax_ok(plate_norm: str) -> bool:
    if not plate_norm:
        return False
    return bool(STANDARD.match(plate_norm) or BH_SERIES.match(plate_norm) or GJ_GOVT.match(plate_norm))


def layout_hint(plate_norm: str) -> str:
    """Map letter/digit confusions using Indian standard 10-char slots.

    Does not overwrite plate_raw. Used only as an extra exact-match key.
    """
    if len(plate_norm) != 10:
        return plate_norm
    digit = str.maketrans("OQDGCILSB", "000061158")
    letter = str.maketrans("018", "OIB")
    out: list[str] = []
    for i, ch in enumerate(plate_norm):
        slot = "LLDDLLDDDD"[i]
        out.append(ch.translate(digit) if slot == "D" else ch.translate(letter))
    return "".join(out)


def vote(reads: list[str]) -> str:
    """Character-wise majority across normalised reads of one passage."""
    normed = [normalize(r) for r in reads]
    normed = [n for n in normed if n]
    if not normed:
        return ""
    width = max(len(s) for s in normed)
    chars: list[str] = []
    for i in range(width):
        column = [s[i] for s in normed if i < len(s)]
        chars.append(Counter(column).most_common(1)[0][0])
    return "".join(chars)
