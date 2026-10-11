"""Tests for the corpus licence gate.

This is the test that protects the project legally: it pins down that
non-redistributable articles are *refused*, not quietly skipped. If these pass,
no NC/ND or unlicensed article can enter the committed corpus.
"""

from __future__ import annotations

import pytest

from rag.licensing import (
    LicenceNotRedistributable,
    assert_commercial_licence,
    extract_licence_refs,
    normalise_licence_ref,
    normalise_oa_bucket,
)


def _article(permissions_xml: str) -> bytes:
    """Wrap a <permissions> block in a minimal but structurally real JATS article."""
    return f"""<?xml version="1.0"?>
<article xmlns:ali="http://www.niso.org/schemas/ali/1.0/"
         xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <title-group><article-title>Knee OA grading</article-title></title-group>
      {permissions_xml}
    </article-meta>
  </front>
  <body><sec><title>Intro</title><p>Text.</p></sec></body>
</article>""".encode()


LICENSE_REF_CC_BY_4 = _article(
    """<permissions>
        <license license-type="open-access">
          <ali:license_ref>https://creativecommons.org/licenses/by/4.0/</ali:license_ref>
          <license-p>Open Access article.</license-p>
        </license>
      </permissions>"""
)

XLINK_HREF_CC_BY_4 = _article(
    """<permissions>
        <license license-type="open-access"
                 xlink:href="http://creativecommons.org/licenses/by/4.0/">
          <license-p>Open Access article.</license-p>
        </license>
      </permissions>"""
)

CC_ZERO = _article(
    """<permissions>
        <license>
          <ali:license_ref>http://creativecommons.org/publicdomain/zero/1.0/</ali:license_ref>
        </license>
      </permissions>"""
)

CC_BY_NC = _article(
    """<permissions>
        <license>
          <ali:license_ref>https://creativecommons.org/licenses/by-nc/4.0/</ali:license_ref>
        </license>
      </permissions>"""
)

CC_BY_NC_ND = _article(
    """<permissions>
        <license>
          <ali:license_ref>https://creativecommons.org/licenses/by-nc-nd/4.0/</ali:license_ref>
        </license>
      </permissions>"""
)

NO_LICENCE_URL = _article(
    """<permissions>
        <copyright-statement>(c) 2021 The Authors</copyright-statement>
        <license>
          <license-p>Readers may use this article as long as it is properly cited.</license-p>
        </license>
      </permissions>"""
)

NO_PERMISSIONS_AT_ALL = _article("")


# ── Acceptance ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("xml", "expected_ref", "expected_source"),
    [
        (
            LICENSE_REF_CC_BY_4,
            "https://creativecommons.org/licenses/by/4.0/",
            "ali:license_ref",
        ),
        (
            XLINK_HREF_CC_BY_4,
            "https://creativecommons.org/licenses/by/4.0/",
            "xlink:href",
        ),
        (
            CC_ZERO,
            "https://creativecommons.org/publicdomain/zero/1.0/",
            "ali:license_ref",
        ),
    ],
)
def test_accepts_commercial_use_licences(
    xml: bytes, expected_ref: str, expected_source: str
) -> None:
    assertion = assert_commercial_licence("PMC1", xml)
    assert assertion.licence_ref == expected_ref
    assert assertion.source == expected_source


def test_http_and_https_spellings_are_equivalent() -> None:
    """An article must not be refused over a URL scheme difference."""
    assert assert_commercial_licence("PMC1", XLINK_HREF_CC_BY_4).licence_ref.startswith(
        "https://"
    )


# ── Refusal — the cases that matter ───────────────────────────────────────────


def test_refuses_non_commercial() -> None:
    with pytest.raises(LicenceNotRedistributable, match="not on the commercial-use"):
        assert_commercial_licence("PMC2", CC_BY_NC)


def test_refuses_no_derivatives_non_commercial() -> None:
    with pytest.raises(LicenceNotRedistributable):
        assert_commercial_licence("PMC3", CC_BY_NC_ND)


def test_refuses_human_readable_licence_with_no_url() -> None:
    """A <license-p> paragraph is not a machine-readable licence."""
    with pytest.raises(LicenceNotRedistributable, match="no licence URL"):
        assert_commercial_licence("PMC4", NO_LICENCE_URL)


def test_refuses_article_with_no_permissions_block() -> None:
    with pytest.raises(LicenceNotRedistributable, match="no licence URL"):
        assert_commercial_licence("PMC5", NO_PERMISSIONS_AT_ALL)


def test_refuses_unparseable_xml() -> None:
    with pytest.raises(LicenceNotRedistributable, match="not parseable"):
        assert_commercial_licence("PMC6", b"<article><front>truncated")


def test_refuses_when_oa_bucket_contradicts_the_xml() -> None:
    """The stricter of the two signals wins.

    A CC BY licence_ref paired with an OA bucket of "CC BY-NC" means one of the
    two is wrong. Refusing is the only safe resolution.
    """
    with pytest.raises(LicenceNotRedistributable, match="not commercial-use-allowed"):
        assert_commercial_licence("PMC7", LICENSE_REF_CC_BY_4, oa_bucket="CC BY-NC")


def test_accepts_when_oa_bucket_agrees() -> None:
    assertion = assert_commercial_licence(
        "PMC8", LICENSE_REF_CC_BY_4, oa_bucket="CC BY"
    )
    assert assertion.oa_bucket == "cc by"


# ── Scoping: a licence URL elsewhere in the document is not a claim ───────────


def test_licence_url_outside_permissions_is_ignored() -> None:
    """A CC URL in a reference or caption says nothing about *this* article."""
    sneaky = _article(
        """<permissions>
            <license>
              <ali:license_ref>https://creativecommons.org/licenses/by-nc/4.0/</ali:license_ref>
            </license>
          </permissions>"""
    ).replace(
        b"<p>Text.</p>",
        b"<p>Adapted from a CC BY work: https://creativecommons.org/licenses/by/4.0/</p>",
    )
    with pytest.raises(LicenceNotRedistributable):
        assert_commercial_licence("PMC9", sneaky)


def test_extract_only_reads_front_matter_permissions() -> None:
    refs = extract_licence_refs(LICENSE_REF_CC_BY_4)
    assert refs == ["https://creativecommons.org/licenses/by/4.0/"]


# ── Normalisation helpers ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://creativecommons.org/licenses/by/4.0/", "https://creativecommons.org/licenses/by/4.0/"),
        ("HTTPS://CreativeCommons.org/licenses/BY/4.0/", "https://creativecommons.org/licenses/by/4.0/"),
        ("  https://creativecommons.org/licenses/by/4.0  ", "https://creativecommons.org/licenses/by/4.0/"),
    ],
)
def test_normalise_licence_ref(raw: str, expected: str) -> None:
    assert normalise_licence_ref(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("CC BY", "cc by"), ("  CC   BY-SA ", "cc by-sa"), (None, None)],
)
def test_normalise_oa_bucket(raw: str | None, expected: str | None) -> None:
    assert normalise_oa_bucket(raw) == expected
