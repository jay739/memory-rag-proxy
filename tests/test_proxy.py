import json
import math
import os
import sys
import tempfile
from pathlib import Path

# The module reads MEMORY_DIR at import time, so point it at a scratch folder
# before importing.
_MEMORY_DIR = tempfile.mkdtemp(prefix="memory-rag-test-")
os.environ["MEMORY_DIR"] = _MEMORY_DIR
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory_rag_proxy as mrp  # noqa: E402


def test_parse_aliases_orders_candidates():
    raw = "fast=http://a:11434|llama3:8b,http://b:11434|qwen:7b; other=http://c|m"
    aliases = mrp._parse_aliases(raw)
    assert aliases["fast"] == [
        ("http://a:11434", "llama3:8b"),
        ("http://b:11434", "qwen:7b"),
    ]
    assert aliases["other"] == [("http://c", "m")]


def test_parse_aliases_skips_malformed_entries():
    assert mrp._parse_aliases("") == {}
    assert mrp._parse_aliases("no-equals-sign") == {}
    assert mrp._parse_aliases("empty=no-pipe-here") == {}


def test_split_into_chunks_on_headers():
    text = "preamble\n\n## One\nfirst\n\n### Two\nsecond\n"
    chunks = mrp.split_into_chunks(text, "note.md")
    assert [c["text"] for c in chunks] == [
        "preamble",
        "## One\nfirst",
        "### Two\nsecond",
    ]
    assert all(c["source"] == "note.md" for c in chunks)


def test_split_into_chunks_empty_text():
    assert mrp.split_into_chunks("   \n\n", "note.md") == []


def test_oversized_chunk_is_split_under_limit():
    para = "x" * (mrp.MAX_CHUNK_CHARS - 10)
    text = "\n\n".join([para, para, para])
    chunks = mrp._split_oversized(text, "big.md")
    assert len(chunks) == 3
    assert all(len(c["text"]) <= mrp.MAX_CHUNK_CHARS for c in chunks)


def test_single_huge_paragraph_is_hard_cut():
    text = "y" * (mrp.MAX_CHUNK_CHARS * 2 + 5)
    chunks = mrp._split_oversized(text, "huge.md")
    assert "".join(c["text"] for c in chunks) == text
    assert all(len(c["text"]) <= mrp.MAX_CHUNK_CHARS for c in chunks)


def test_small_paragraphs_are_repacked():
    paras = ["p" * 100] * 80  # well over the limit in total
    chunks = mrp._split_oversized("\n\n".join(paras), "many.md")
    assert 1 < len(chunks) < 80


def test_cosine_similarity():
    assert math.isclose(mrp.cosine_similarity([1, 0], [1, 0]), 1.0)
    assert math.isclose(mrp.cosine_similarity([1, 0], [0, 1]), 0.0)
    assert mrp.cosine_similarity([0, 0], [1, 1]) == 0.0


def test_openai_chat_translated_to_ollama_shape():
    resp = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 7},
    }
    out = json.loads(mrp._openai_chat_to_ollama(resp, "mlx-model"))
    assert out["model"] == "mlx-model"
    assert out["message"]["content"] == "hi"
    assert out["done"] is True
    assert out["done_reason"] == "length"
    assert (out["prompt_eval_count"], out["eval_count"]) == (3, 7)


def test_openai_chat_with_no_choices():
    out = json.loads(mrp._openai_chat_to_ollama({}, "m"))
    assert out["message"]["content"] == ""
    assert out["done_reason"] == "stop"


def test_build_context_message():
    assert mrp.build_context_message([]) == ""
    msg = mrp.build_context_message([{"source": "a.md", "text": "alpha"}])
    assert "[a.md]" in msg and "alpha" in msg


def test_memory_index_builds_caches_and_searches(monkeypatch):
    Path(_MEMORY_DIR, "a.md").write_text("## Cats\ncats purr\n")
    Path(_MEMORY_DIR, "b.md").write_text("## Dogs\ndogs bark\n")

    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        return [[1.0, 0.0] if "cat" in t else [0.0, 1.0] for t in texts]

    monkeypatch.setattr(mrp, "embed_texts", fake_embed)

    index = mrp.MemoryIndex()
    assert len(index.chunks) == 2
    hits = index.search([1.0, 0.0], top_k=3, threshold=0.5)
    assert [h["source"] for h in hits] == ["a.md"]

    # Unchanged files come from the cache, so nothing is re-embedded.
    calls.clear()
    mrp.MemoryIndex()
    assert calls == []
