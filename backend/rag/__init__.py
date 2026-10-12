"""RAG + MEDITRON clinical-note layer.

Retrieval and the NLI safety gate run in this process; generation runs on a
remote Ollama host reached via ``OLLAMA_BASE_URL``. See
``docs/rag-meditron-design.md``.
"""
