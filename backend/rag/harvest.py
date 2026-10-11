"""Harvest PMC Open Access articles into the committed corpus.

    python -m rag.harvest             # fetch every id in corpus/seed_pmcids.txt
    python -m rag.harvest --dry-run   # check licences without writing anything

Reads a hand-curated, committed seed list rather than performing a live search:
search ranking drifts, so a re-run months later would silently produce a
different corpus and invalidate any published numbers.

Every article passes :func:`rag.licensing.assert_commercial_licence` before being
written. A refusal is reported and the article is skipped *loudly* — the run
reports a non-zero exit when anything was refused, so a licensing problem cannot
pass unnoticed into a committed corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from rag.config import (
    ARTICLES_DIR,
    MANIFEST_PATH,
    NCBI_API_KEY,
    NCBI_EMAIL,
    NCBI_EUTILS_BASE,
    NCBI_OA_SERVICE,
    NCBI_RATE_LIMIT_PER_S,
    NCBI_TOOL,
    SEED_PMCIDS_PATH,
)
from rag.corpus_types import ManifestEntry
from rag.licensing import (
    JATS_NS,
    LicenceNotRedistributable,
    assert_commercial_licence,
)

PMC_ARTICLE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"


class HarvestError(RuntimeError):
    """A fetch failed for a reason unrelated to licensing."""


@dataclass
class _RateLimiter:
    """Minimal spacing between NCBI requests.

    NCBI permits 3 requests/second unauthenticated and 10 with an API key; the
    limit is derived from whether a key is configured, so no key simply means a
    slower run rather than a failure.
    """

    per_second: float
    _last: float = 0.0

    def wait(self) -> None:
        interval = 1.0 / self.per_second
        elapsed = time.monotonic() - self._last
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last = time.monotonic()


def read_seed_pmcids(path=SEED_PMCIDS_PATH) -> list[str]:
    """Read the committed seed list, ignoring blanks and ``#`` comments."""
    if not path.is_file():
        raise HarvestError(
            f"Seed list missing: {path}. It is committed deliberately so that "
            "re-runs reproduce the same corpus."
        )
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            ids.append(token if token.upper().startswith("PMC") else f"PMC{token}")
    return ids


def _params(extra: dict[str, str]) -> dict[str, str]:
    """Add NCBI's required identification to a request.

    Without ``tool`` (and realistically ``email``), E-utilities answers with a
    302 to its abuse page instead of data — a failure that looks like an ordinary
    redirect unless the body is inspected.
    """
    params = dict(extra)
    params["tool"] = NCBI_TOOL
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def _check_not_throttled(response: httpx.Response, pmcid: str) -> None:
    """Raise if NCBI answered with its abuse/misuse page rather than data."""
    if "misuse.ncbi.nlm.nih.gov" in str(response.url) or b"abuse.shtml" in (
        response.content[:2048]
    ):
        raise HarvestError(
            f"{pmcid}: NCBI redirected to its abuse page. Set NCBI_EMAIL (and "
            "ideally NCBI_API_KEY) in backend/.env, then wait a few minutes "
            "before retrying — an existing block expires on its own."
        )


def fetch_oa_bucket(
    client: httpx.Client, limiter: _RateLimiter, pmcid: str
) -> str | None:
    """Return the PMC OA service's licence bucket label, or None if unavailable.

    Treated as advisory: a missing bucket leaves the JATS licence as the single
    signal, while a bucket that *contradicts* the JATS licence causes a refusal
    downstream.
    """
    limiter.wait()
    try:
        response = client.get(NCBI_OA_SERVICE, params=_params({"id": pmcid}))
        response.raise_for_status()
        _check_not_throttled(response, pmcid)
    except (httpx.HTTPError, HarvestError):
        return None
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return None
    for record in root.iter("record"):
        licence = record.get("license")
        if licence:
            return licence
    return None


def fetch_jats(client: httpx.Client, limiter: _RateLimiter, pmcid: str) -> bytes:
    """Fetch one article's full JATS XML via E-utilities."""
    limiter.wait()
    numeric = pmcid[3:] if pmcid.upper().startswith("PMC") else pmcid
    try:
        response = client.get(
            f"{NCBI_EUTILS_BASE}/efetch.fcgi",
            params=_params({"db": "pmc", "id": numeric, "rettype": "xml"}),
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise HarvestError(f"{pmcid}: fetch failed: {e}") from e
    _check_not_throttled(response, pmcid)
    if b"<article" not in response.content:
        raise HarvestError(
            f"{pmcid}: response contains no <article> element. The id may not be "
            "in the Open Access subset."
        )
    return response.content


def _first_text(root: ET.Element, paths: tuple[str, ...]) -> str | None:
    for path in paths:
        for node in root.iterfind(path):
            if node.text and node.text.strip():
                return " ".join(node.text.split())
    return None


def extract_metadata(xml_bytes: bytes) -> dict[str, str | None]:
    """Pull title, journal, year and DOI from JATS front matter."""
    root = ET.fromstring(xml_bytes)
    title = _first_text(
        root, ("./front/article-meta/title-group/article-title",)
    )
    journal = _first_text(
        root,
        (
            "./front/journal-meta/journal-title-group/journal-title",
            "./front/journal-meta/journal-title",
        ),
    )
    year = _first_text(
        root,
        (
            "./front/article-meta/pub-date[@pub-type='epub']/year",
            "./front/article-meta/pub-date/year",
        ),
    )
    doi = None
    for node in root.iterfind("./front/article-meta/article-id"):
        if node.get("pub-id-type") == "doi" and node.text:
            doi = node.text.strip()
            break
    return {"title": title, "journal": journal, "year": year, "doi": doi}


def harvest_one(
    client: httpx.Client, limiter: _RateLimiter, pmcid: str
) -> tuple[ManifestEntry, bytes]:
    """Fetch, licence-check and describe one article.

    Raises:
        LicenceNotRedistributable: the article may not be redistributed.
        HarvestError: the fetch itself failed.
    """
    bucket = fetch_oa_bucket(client, limiter, pmcid)
    xml_bytes = fetch_jats(client, limiter, pmcid)

    assertion = assert_commercial_licence(pmcid, xml_bytes, oa_bucket=bucket)
    meta = extract_metadata(xml_bytes)

    entry = ManifestEntry(
        pmcid=pmcid,
        title=meta["title"] or pmcid,
        licence_ref=assertion.licence_ref,
        licence_source=assertion.source,
        oa_bucket=assertion.oa_bucket,
        sha256=hashlib.sha256(xml_bytes).hexdigest(),
        source_url=PMC_ARTICLE_URL.format(pmcid=pmcid),
        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        journal=meta["journal"],
        year=meta["year"],
        doi=meta["doi"],
    )
    return entry, xml_bytes


def write_manifest(entries: list[ManifestEntry]) -> None:
    """Write the manifest sorted by id so the committed file diffs cleanly."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [e.to_dict() for e in sorted(entries, key=lambda e: e.pmcid)]
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def harvest(pmcids: list[str], dry_run: bool = False) -> tuple[list[ManifestEntry], list[str]]:
    """Harvest every id, returning accepted entries and refusal messages."""
    limiter = _RateLimiter(per_second=NCBI_RATE_LIMIT_PER_S)
    entries: list[ManifestEntry] = []
    refusals: list[str] = []

    if not dry_run:
        ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "KneeXpert-RAG/0.1 (research; contact via repository)"}
    with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
        for pmcid in pmcids:
            try:
                entry, xml_bytes = harvest_one(client, limiter, pmcid)
            except LicenceNotRedistributable as e:
                refusals.append(str(e))
                print(f"  REFUSED  {pmcid}: {e}")
                continue
            except HarvestError as e:
                refusals.append(str(e))
                print(f"  FAILED   {pmcid}: {e}")
                continue

            if not dry_run:
                (ARTICLES_DIR / f"{pmcid}.xml").write_bytes(xml_bytes)
            entries.append(entry)
            print(f"  ok       {pmcid}  {entry.oa_bucket or entry.licence_ref}")
            print(f"           {entry.title[:88]}")

    return entries, refusals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify licences without writing articles or the manifest",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="harvest these PMC ids instead of the committed seed list",
    )
    args = parser.parse_args(argv)

    pmcids = args.ids if args.ids else read_seed_pmcids()
    key_state = "with API key" if NCBI_API_KEY else "no API key"
    print(
        f"Harvesting {len(pmcids)} article(s) at "
        f"{NCBI_RATE_LIMIT_PER_S:g} req/s ({key_state})"
    )

    entries, refusals = harvest(pmcids, dry_run=args.dry_run)

    if entries and not args.dry_run:
        write_manifest(entries)
        print(f"\nManifest: {MANIFEST_PATH}  ({len(entries)} article(s))")
        print(f"Articles: {ARTICLES_DIR}")

    buckets: dict[str, int] = {}
    for entry in entries:
        label = entry.oa_bucket or entry.licence_ref
        buckets[label] = buckets.get(label, 0) + 1
    if buckets:
        print("\nLicence breakdown:")
        for label, count in sorted(buckets.items()):
            print(f"  {count:3d}  {label}")

    if refusals:
        print(f"\n{len(refusals)} article(s) refused or failed:")
        for message in refusals:
            print(f"  - {message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
