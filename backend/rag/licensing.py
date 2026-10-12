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

# PMC bucket labels we accept, reported on the OA service's <record license="...">
# attribute. This MUST encode the same policy as config.ALLOWED_LICENCE_REFS, or
# the two signals that are supposed to cross-check each other would disagree and
# the stricter-wins rule would be meaningless.
#
# Note this is narrower than PMC's own "Commercial Use Allowed" bucket, which also
# contains CC BY-ND. ND is excluded deliberately: the corpus is redistributed as
# chunked excerpts with context headers prepended, which is a derivative work.
ALLOWED_OA_BUCKETS: frozenset[str] = frozenset({"cc0", "cc by", "cc by-sa"})


class LicenceNotRedistributable(RuntimeError):
    """Raised when an article's licence is absent, unreadable, or not on the allowlist.

    This is deliberately a hard failure. A corpus that silently drops an article
    is indistinguishable from one that silently includes a copyrighted one.
    """


@dataclass(frozen=True)
class LicenceDeclaration:
    """One licence URL as found in an article, with the construct that carried it."""

    raw: str
    source: str
    """"ali:license_ref" or "xlink:href"."""

    @property
    def normalised(self) -> str:
        return normalise_licence_ref(self.raw)


@dataclass(frozen=True)
class LicenceAssertion:
    """A verified, redistributable licence for one article."""

    licence_ref: str
    """Canonical licence URL as asserted from the JATS XML."""

    oa_bucket: str | None
    """PMC OA service bucket label, when an OA record was supplied."""

    source: str
    """Which JATS construct carried the ref: "ali:license_ref" or "xlink:href"."""


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _is_creative_commons(url: str) -> bool:
    """Whether a URL is a Creative Commons licence deed.

    Used to pick the licence out of ``<license-p>`` prose, which also links to
    unrelated things. Says nothing about whether the licence is *acceptable* —
    the allowlist decides that, so an NC deed matches here and is refused later.
    """
    return "creativecommons.org/licenses/" in url.lower() or (
        "creativecommons.org/publicdomain/" in url.lower()
    )


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


def extract_licence_declarations(xml_bytes: bytes) -> list[LicenceDeclaration]:
    """Collect every licence URL declared in a JATS article's front matter.

    Looks only under ``front/article-meta/permissions`` — a licence URL appearing
    anywhere else in the document (a reference, a figure caption) is not a
    statement about this article and must not be trusted as one.

    Each declaration records which construct carried it, so the caller does not
    have to re-parse the document to find out. Returns an empty list when no
    licence URL is present. Order is document order, ``ali:license_ref`` elements
    before ``xlink:href`` attributes.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        raise LicenceNotRedistributable(f"Article XML is not parseable: {e}") from e

    declarations: list[LicenceDeclaration] = []
    for permissions in root.iterfind("./front/article-meta/permissions"):
        for licence in permissions.iterfind("./license"):
            for node in licence.iterfind("./ali:license_ref", JATS_NS):
                if node.text and node.text.strip():
                    declarations.append(
                        LicenceDeclaration(node.text.strip(), "ali:license_ref")
                    )
            href = licence.get(f"{{{JATS_NS['xlink']}}}href")
            if href and href.strip():
                declarations.append(LicenceDeclaration(href.strip(), "xlink:href"))

            # MDPI, Springer Nature and others put the licence URL in an
            # <ext-link> inside <license-p> rather than on <license> itself.
            # That is still machine-readable and still inside the permissions
            # block, so it speaks for this article; missing it refused genuinely
            # CC BY papers. Only CC URLs are collected, so the surrounding
            # boilerplate about third-party material contributes nothing.
            for link in licence.iter():
                if _localname(link.tag) != "ext-link":
                    continue
                link_href = link.get(f"{{{JATS_NS['xlink']}}}href")
                if link_href and _is_creative_commons(link_href):
                    declarations.append(
                        LicenceDeclaration(link_href.strip(), "license-p/ext-link")
                    )
    return declarations


def extract_licence_refs(xml_bytes: bytes) -> list[str]:
    """Licence URLs from front matter, without their source labels."""
    return [d.raw for d in extract_licence_declarations(xml_bytes)]


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
    declarations = extract_licence_declarations(xml_bytes)
    if not declarations:
        raise LicenceNotRedistributable(
            f"{pmcid}: no licence URL in front/article-meta/permissions. "
            "PMC articles without a machine-readable licence are excluded — a "
            "human-readable licence-p paragraph is not sufficient."
        )

    accepted = [d for d in declarations if d.normalised in _ALLOWED_NORMALISED]
    rejected = [d for d in declarations if d.normalised not in _ALLOWED_NORMALISED]

    if not accepted:
        found = [d.raw for d in declarations]
        raise LicenceNotRedistributable(
            f"{pmcid}: licence {found!r} is not on the redistribution allowlist. "
            "Only CC0 and CC BY / BY-SA are accepted. NC is excluded because the "
            "corpus is committed to a public repository, and ND because chunked "
            "excerpts with prepended context headers are a derivative work."
        )

    # An article declaring both an allowed and a disallowed licence is ambiguous
    # about which governs its text. Stricter-wins means refusing, not picking the
    # convenient one.
    if rejected:
        raise LicenceNotRedistributable(
            f"{pmcid}: conflicting licence declarations in permissions — "
            f"{[d.raw for d in accepted]!r} is allowed but "
            f"{[d.raw for d in rejected]!r} is not. Refusing rather than choosing "
            "which one governs."
        )

    bucket = normalise_oa_bucket(oa_bucket)
    if bucket is not None and bucket not in ALLOWED_OA_BUCKETS:
        raise LicenceNotRedistributable(
            f"{pmcid}: PMC OA service reports licence bucket {oa_bucket!r}, which "
            f"is not on the redistribution allowlist, even though the article XML "
            f"declares {accepted[0].raw!r}. Refusing on the stricter of the two "
            "signals."
        )

    chosen = accepted[0]
    return LicenceAssertion(
        licence_ref=chosen.normalised, oa_bucket=bucket, source=chosen.source
    )
