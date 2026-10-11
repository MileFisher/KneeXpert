"""RAG + MEDITRON clinical-note configuration — paths, env vars, hyperparameters.

This is the ONLY module in the backend that reads the environment. Every other
module in ``rag/`` imports named constants from here, so there is exactly one
place to look when a deployment behaves differently from a dev machine.

See ``docs/rag-meditron-design.md`` for why each value is what it is.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAG_ROOT = Path(__file__).resolve().parent
BACKBONE_ROOT = RAG_ROOT.parent

CORPUS_DIR = RAG_ROOT / "corpus"
ARTICLES_DIR = CORPUS_DIR / "articles"
INTERNAL_DIR = CORPUS_DIR / "internal"
SEED_PMCIDS_PATH = CORPUS_DIR / "seed_pmcids.txt"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
CHUNKS_PATH = CORPUS_DIR / "chunks.jsonl"

# Qdrant embedded store. Gitignored — rebuilt by `python -m rag.ingest`.
INDEX_DIR = RAG_ROOT / "index"

EVAL_DIR = RAG_ROOT / "eval"
EVAL_PAYLOADS_PATH = EVAL_DIR / "payloads.json"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# Loaded before any os.getenv below, so backend/.env wins over a bare shell.
load_dotenv(BACKBONE_ROOT / ".env")

# ── Generation (remote Ollama host) ───────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "meditron3-8b")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "120"))

# Ollama defaults num_ctx to 2048 regardless of the model's 128K capability, and
# silently truncates FROM THE LEFT — dropping the system prompt's safety rules
# first. Always send this explicitly; never rely on the server default.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "500"))

# A dead GPU box must not hang /health, which the frontend calls on page load.
# The connect phase is budgeted separately and tightly: on Windows an unreachable
# host can burn the whole timeout on an IPv6 attempt before falling back to IPv4,
# so a single overall timeout understates the real wall-clock cost.
LLM_PROBE_CONNECT_TIMEOUT_S = 0.5
LLM_PROBE_TIMEOUT_S = 1.5
LLM_PROBE_CACHE_TTL_S = 30.0

# ── Retrieval ─────────────────────────────────────────────────────────────────

# BGE-small is asymmetric: queries need a prefix, passages do not. The prefix is
# applied inside Embedder.embed_query so it cannot be forgotten at a call site.
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_BATCH_SIZE = 64

COLLECTION = "kneexpert_corpus"
TOP_K = 6
TOP_K_RETRY = 10
MAX_CHUNKS_PER_ARTICLE = 2  # stop one verbose paper owning the context window

# ── Chunking ──────────────────────────────────────────────────────────────────

CHUNK_TARGET_CHARS = 900  # ~220 tokens; BGE-small peaks well below its 512 max
CHUNK_MAX_CHARS = 1300
CHUNK_MIN_CHARS = 200
CHUNK_MAX_NON_ALPHA_RATIO = 0.30  # table debris, statistics dumps
CHUNK_MAX_DIGIT_RATIO = 0.50

# ── Safety gate ───────────────────────────────────────────────────────────────

# DeBERTa-v3-base MNLI is 184M params (~740MB resident) and runs at roughly
# 100-400 ms/pair on integrated graphics. Override to a `-xsmall` MNLI variant on
# a weak dev box; §4.3 must report which variant produced its numbers.
NLI_MODEL = os.getenv(
    "RAG_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)
NLI_BATCH_SIZE = 16
NLI_MAX_LENGTH = 384
NLI_PREMISE_TOP_N = 3  # per sentence, pre-filtered by embedding cosine

GROUNDEDNESS_TAU = float(os.getenv("RAG_TAU", "0.70"))
MAX_RETRIES = 2  # 3 attempts total

# Attempt -> temperature. Attempt 3 also re-retrieves at TOP_K_RETRY.
RETRY_TEMPERATURES = (0.2, 0.1, 0.0)

# Fixed so §4.3 is reproducible for a grader.
GEN_SEED = int(os.getenv("RAG_SEED", "42"))

# ── Corpus licensing ──────────────────────────────────────────────────────────

# PMC Open Access "Commercial Use Allowed" bucket only. harvest.py RAISES on any
# article whose licence is not machine-assertable against this allowlist — it
# never skips silently. NICE (UK-setting-only licence), AAOS and ACR (copyright)
# are deliberately absent; see decision 3 in the design doc.
ALLOWED_LICENCE_REFS: tuple[str, ...] = (
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/licenses/by/2.0/",
    "https://creativecommons.org/licenses/by/2.0/",
    "http://creativecommons.org/licenses/by/3.0/",
    "https://creativecommons.org/licenses/by/3.0/",
    "http://creativecommons.org/licenses/by/4.0/",
    "https://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/licenses/by-sa/4.0/",
    "https://creativecommons.org/licenses/by-sa/4.0/",
)

# NCBI asks for <=3 req/s without an API key, <=10 with one.
NCBI_API_KEY = os.getenv("NCBI_API_KEY") or None
NCBI_RATE_LIMIT_PER_S = 10.0 if NCBI_API_KEY else 3.0
NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_OA_SERVICE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# E-utilities requires an identifying tool name, and strongly expects a contact
# address. Requests without them are redirected to NCBI's abuse page (an HTTP 302
# to misuse.ncbi.nlm.nih.gov) rather than refused outright, which looks like a
# successful redirect unless you read the body.
#
# NCBI_EMAIL is deliberately NOT defaulted. It is sent to a third party, so it
# must be supplied explicitly by whoever runs the harvest.
NCBI_TOOL = os.getenv("NCBI_TOOL", "KneeXpert-RAG")
NCBI_EMAIL = os.getenv("NCBI_EMAIL") or None


# ── Readiness probes (mirrors mri.pipeline.mri_models_available) ──────────────


def rag_corpus_chunk_count() -> int:
    """Number of chunks in the committed corpus, or 0 if it has not been built."""
    if not CHUNKS_PATH.is_file():
        return 0
    with CHUNKS_PATH.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def rag_index_ready() -> bool:
    """True when an on-disk Qdrant index exists with at least one point."""
    if not INDEX_DIR.is_dir():
        return False
    return any(INDEX_DIR.iterdir())
