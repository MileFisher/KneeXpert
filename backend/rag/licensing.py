"""Corpus licence assertion — the gate that keeps non-redistributable text out.

Every article entering the corpus must carry a machine-readable Creative Commons
licence from the PMC Open Access "Commercial Use Allowed" bucket. This module
*raises* on anything else; it never skips, warns, or downgrades. Nothing is
chunked from an article whose licence cannot be asserted.

Why this exists: the original corpus plan (NICE NG226 + AAOS CPG + ACR
Appropriateness Criteria) does not survive a licence check from Vietnam. The NICE
UK Open Content Licence is UK-setting-only, and AAOS and ACR hold copyright on
their documents — free to download is not free to redistribute. See decision 3 in
``docs/rag-meditron-design.md``.

Two independent signals are checked and must agree:
  1. the JATS ``<license>`` element's ``ali:license_ref`` or ``xlink:href``, and
  2. the PMC OA web service's ``license`` attribute (its bucket label).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from rag.config import ALLOWED_LICENCE_REFS

JATS_NS = {
    "ali": "http://www.niso.org/schemas/ali/1.0/",
    "xlink": "http://www.w3.org/1999/xlink",
}

# PMC's own bucket labels for the commercial-use-allowed set. The OA service
# reports these on the <record license="..."> attribute.
ALLOWED_OA_BUCKETS: frozenset[str] = frozenset(
    {"cc0", "cc by", "cc by-sa", "cc by-nd"}
)


class LicenceNotRedistributable(RuntimeError):
    """Raised when an article's licence is absent, unreadable, or not on the allowlist.

    This is deliberately a hard failure. A corpus that silently drops an article
    is indistinguishable from one that silently includes a copyrighted one.
    """


@dataclass(frozen=True)
class LicenceAssertion:
    """A verified, redistributable licence for one article."""

    licence_ref: str
    """Canonical licence URL as asserted from the JATS XML."""

    oa_bucket: str | None
    """PMC OA service bucket label, when an OA record was supplied."""

    source: str
    """Which JATS construct carried the ref: "ali:license_ref" or "xlink:href"."""


def normalise_licence_ref(raw: str) -> str:
    """Reduce a licence URL to a canonical form for comparison.

    Lowercases, upgrades ``http`` to ``https``, and enforces a trailing slash, so
    the many equivalent spellings of a CC URL compare equal.
    """
    ref = raw.strip().lower()
    if ref.startswith("http://"):
        ref = "https://" + ref[len("http://") :]
    if not ref.endswith("/"):
        ref += "/"
    return ref


_ALLOWED_NORMALISED: frozenset[str] = frozenset(
    normalise_licence_ref(ref) for ref in ALLOWED_LICENCE_REFS
)


def extract_licence_refs(xml_bytes: bytes) -> list[str]:
    """Collect every licence URL declared in a JATS article's front matter.

    Looks only under ``front/article-meta/permissions`` — a licence URL appearing
    anywhere else in the document (a reference, a figure caption) is not a
    statement about this article and must not be trusted as one.

    Returns an empty list when no licence URL is present. Order is document order,
    ``ali:license_ref`` elements before ``xlink:href`` attributes.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        raise LicenceNotRedistributable(f"Article XML is not parseable: {e}") from e

    refs: list[str] = []
    for permissions in root.iterfind("./front/article-meta/permissions"):
        for licence in permissions.iterfind("./license"):
            for node in licence.iterfind("./ali:license_ref", JATS_NS):
                if node.text and node.text.strip():
                    refs.append(node.text.strip())
            href = licence.get(f"{{{JATS_NS['xlink']}}}href")
            if href and href.strip():
                refs.append(href.strip())
    return refs


def _licence_ref_source(xml_bytes: bytes, chosen: str) -> str:
    """Report which JATS construct supplied the accepted ref (for the manifest)."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return "unknown"
    target = normalise_licence_ref(chosen)
    for permissions in root.iterfind("./front/article-meta/permissions"):
        for licence in permissions.iterfind("./license"):
            for node in licence.iterfind("./ali:license_ref", JATS_NS):
                if node.text and normalise_licence_ref(node.text) == target:
                    return "ali:license_ref"
    return "xlink:href"


def normalise_oa_bucket(raw: str | None) -> str | None:
    """Canonicalise a PMC OA service bucket label, e.g. ``"CC BY"`` -> ``"cc by"``."""
    if raw is None:
        return None
    return " ".join(raw.strip().lower().split())


def assert_commercial_licence(
    pmcid: str,
    xml_bytes: bytes,
    oa_bucket: str | None = None,
) -> LicenceAssertion:
    """Verify an article may be redistributed, or raise.

    Args:
        pmcid: PMC identifier, used only for error messages.
        xml_bytes: the article's full JATS XML.
        oa_bucket: the PMC OA service's ``license`` attribute, when available.
            Supplying it adds a second, independent check; omitting it leaves the
            JATS assertion as the only signal.

    Returns:
        A :class:`LicenceAssertion` carrying the canonical licence URL.

    Raises:
        LicenceNotRedistributable: if the XML is unparseable, declares no licence
            URL, declares one outside the commercial-use allowlist, or disagrees
            with the OA service bucket.
    """
    refs = extract_licence_refs(xml_bytes)
    if not refs:
        raise LicenceNotRedistributable(
            f"{pmcid}: no licence URL in front/article-meta/permissions. "
            "PMC articles without a machine-readable licence are excluded — a "
            "human-readable licence-p paragraph is not sufficient."
        )

    normalised = [normalise_licence_ref(ref) for ref in refs]
    accepted = [
        original
        for original, norm in zip(refs, normalised, strict=True)
        if norm in _ALLOWED_NORMALISED
    ]
    if not accepted:
        raise LicenceNotRedistributable(
            f"{pmcid}: licence {refs!r} is not on the commercial-use allowlist. "
            "Only CC0 and CC BY / BY-SA are accepted; NC and ND variants are "
            "excluded because the corpus is committed to a public repository."
        )

    bucket = normalise_oa_bucket(oa_bucket)
    if bucket is not None and bucket not in ALLOWED_OA_BUCKETS:
        raise LicenceNotRedistributable(
            f"{pmcid}: PMC OA service reports licence bucket {oa_bucket!r}, which "
            f"is not commercial-use-allowed, even though the article XML declares "
            f"{accepted[0]!r}. Refusing on the stricter of the two signals."
        )

    chosen = accepted[0]
    return LicenceAssertion(
        licence_ref=normalise_licence_ref(chosen),
        oa_bucket=bucket,
        source=_licence_ref_source(xml_bytes, chosen),
    )
