"""Sentence splitting for the safety gate. Pure: no models, no data files.

Each sentence becomes one entailment hypothesis, so a bad split degrades the
gate itself. Fusing two claims lets an unsupported one hide behind a supported
one; severing a citation marker from its claim misattributes the source.

Deliberately a short explicit function rather than NLTK or pysbd: the failure
modes that matter here are a known, testable list (medical abbreviations,
decimals, bracketed citation markers), and a dependency that downloads a corpus
at first use is a poor trade for that.
"""

from __future__ import annotations

import re

# Abbreviations whose full stop is never a sentence boundary. Lower-cased for
# comparison; the trailing dot is implied.
ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # citation and reference conventions
        "al",
        "et al",
        "eg",
        "e.g",
        "ie",
        "i.e",
        "cf",
        "viz",
        "etc",
        "fig",
        "figs",
        "tab",
        "ref",
        "refs",
        "vol",
        "no",
        "pp",
        "ed",
        "eds",
        # titles
        "dr",
        "prof",
        "mr",
        "mrs",
        "ms",
        "st",
        # measurement and hedging
        "approx",
        "ca",
        "mm",
        "cm",
        "ml",
        "mg",
        "kg",
        "sec",
        "min",
        "hr",
        "wk",
        "mo",
        "yr",
        "vs",
        "v",
    }
)

_WHITESPACE = re.compile(r"\s+")

# A candidate boundary: terminator(s), then whitespace, then something that
# could begin a sentence.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Normalize markers that follow a full stop so they attach to the sentence they cite:
# e.g. "change is present. [1]" -> "change is present [1]."
_NORM_CITATIONS = re.compile(r"\.\s*(\[\d+(?:[,\s-]+\d+)*\](?:\s*\[\d+\])*)\.?")

# Trailing token before a full stop, used to test it against ABBREVIATIONS.
_LAST_TOKEN = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")

# A citation marker fragment alone, e.g. "[2]", "[2].", "[1] [2].", "[1, 2]"
_MARKER_ONLY = re.compile(r"^(?:\[[\d,\s-]+\][\s.]*)+$")

# Decimals need no special handling here: _BOUNDARY requires whitespace after
# the terminator, and a decimal point is always followed by a digit. Guarding
# them explicitly was worse than nothing -- it also matched a sentence that
# legitimately ends in a number ("the grade is 3."), suppressing a real boundary.


def _ends_with_abbreviation(fragment: str) -> bool:
    """Whether a fragment's final full stop belongs to an abbreviation."""
    stripped = fragment.rstrip()
    # Strip a trailing bracketed marker so "... et al. [1]" is still judged on
    # "al." rather than on the marker.
    stripped = re.sub(r"\s*\[\d+\]$", "", stripped).rstrip()

    match = _LAST_TOKEN.search(stripped)
    if match is None:
        return False
    token = match.group(1).rstrip(".").lower()
    if token in ABBREVIATIONS:
        return True
    # A single initial ("J." in "Kellgren J. L.") is never a boundary.
    return len(token) == 1


def split_sentences(text: str) -> list[str]:
    """Split note text into sentences.

    Returns an empty list for blank input. Text with no terminator at all is
    returned as a single sentence rather than discarded, because a model that
    omits its final full stop has still made a claim worth scoring.
    """
    if not text or not text.strip():
        return []

    normalised = _NORM_CITATIONS.sub(r" \1.", text)
    normalised = _WHITESPACE.sub(" ", normalised).strip()

    sentences: list[str] = []
    current = ""
    for piece in _BOUNDARY.split(normalised):
        if not piece:
            continue
        candidate = f"{current} {piece}".strip() if current else piece
        if _ends_with_abbreviation(candidate):
            # Not a real boundary: keep accumulating.
            current = candidate
        else:
            sentences.append(candidate)
            current = ""
    if current:
        sentences.append(current)

    cleaned = [s for s in (part.strip() for part in sentences) if s]

    # A marker fragment alone has no trailing whitespace for _BOUNDARY to consume,
    # so it arrives as its own fragment. Reattach it rather than scoring it as a clinical claim.
    merged: list[str] = []
    for sentence in cleaned:
        if _MARKER_ONLY.match(sentence):
            if merged:
                merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)
    return merged
