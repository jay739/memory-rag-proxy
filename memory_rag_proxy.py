"""
Memory-RAG router for Ollama, with RTX-box-first / Mac-Mini-fallback routing.

Embedding and memory retrieval always happen against EMBED_UPSTREAM (the
local, always-on Ollama instance) since the memory folder is git-synced
identically across boxes anyway (see project_claude_memory_backup.md) --
there's nothing to gain from routing that part. Only the actual chat
generation call is routed: try PRIMARY_UPSTREAM (RTX box, on-demand) first
with a short cached health check, fall back to FALLBACK_UPSTREAM (Mac Mini,
always-on) if it's unreachable. Same design as the portfolio's
src/lib/host-failover.ts / feedback_ollama_failover_design.md: short connect
timeout so a powered-off RTX box doesn't hang every request, cached health
result so we don't pay that timeout on every single message, and a
mid-request retry against the fallback in case RTX drops after passing its
health check.

Config is read from environment variables so the same script can run
unmodified anywhere, though in practice this router is meant to run once,
on the always-on Mac Mini:
  MEMORY_DIR          - path to the memory .md folder (required)
  EMBED_UPSTREAM       - Ollama used for embedding/retrieval (default http://127.0.0.1:11434)
  PRIMARY_UPSTREAM      - tried first for chat generation (RTX box)
  FALLBACK_UPSTREAM     - used if PRIMARY is unreachable (Mac Mini, usually same as EMBED_UPSTREAM)
  HEALTH_TIMEOUT_S      - connect timeout for the primary health check (default 2)
  HEALTH_CACHE_TTL_S    - how long a health result is trusted before re-checking (default 45)
  PROXY_PORT           - port this proxy listens on (default 11435)
  EMBED_MODEL          - embedding model name (default nomic-embed-text)
  TOP_K                - number of chunks to inject (default 3)
  SIM_THRESHOLD        - minimum cosine similarity to inject a chunk (default 0.5)
"""

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MEMORY_DIR = Path(os.environ["MEMORY_DIR"])
EMBED_UPSTREAM = os.environ.get("EMBED_UPSTREAM", "http://127.0.0.1:11434")
PRIMARY_UPSTREAM = os.environ.get("PRIMARY_UPSTREAM", "http://127.0.0.1:11434")
FALLBACK_UPSTREAM = os.environ.get("FALLBACK_UPSTREAM", "http://127.0.0.1:11434")
HEALTH_TIMEOUT_S = float(os.environ.get("HEALTH_TIMEOUT_S", "2"))
HEALTH_CACHE_TTL_S = float(os.environ.get("HEALTH_CACHE_TTL_S", "45"))
PROXY_PORT = int(os.environ.get("PROXY_PORT", "11435"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
TOP_K = int(os.environ.get("TOP_K", "3"))
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.5"))

# MLX server(s) speak the OpenAI-compatible API (/v1/models, /v1/chat/completions),
# not Ollama's native format -- kept as a separate set of upstreams, each
# translated at the boundary so they merge into the same unified model list
# and Ollama-format responses as everything else. Comma-separated list of
# base URLs (each already includes /v1), e.g. "http://mac-mini.local:8080/v1".
MLX_UPSTREAMS = [u for u in os.environ.get("MLX_UPSTREAMS", "").split(",") if u]


# Aliases: one name that AnythingLLM/Open WebUI sees as a single model, backed
# by an ORDERED list of (host, real_model_name) candidates -- tried in order,
# first reachable one wins, same health-check/cache machinery as the plain
# RTX-first/Mac-Mini-fallback pair. The two underlying real model names are
# hidden from the merged /api/tags list so only the alias shows up (that's
# the whole point -- one name, guaranteed an answer from whichever backend
# is actually up). Format: "alias_name=host1|real_model1,host2|real_model2"
# (pipe, not colon, since both URLs and Ollama's own "model:tag" syntax
# already use colons); multiple aliases separated by ";".
def _parse_aliases(raw: str) -> dict[str, list[tuple[str, str]]]:
    aliases: dict[str, list[tuple[str, str]]] = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        name, candidates_raw = entry.split("=", 1)
        candidates = []
        for cand in candidates_raw.split(","):
            if "|" not in cand:
                continue
            host, model = cand.split("|", 1)
            candidates.append((host.strip(), model.strip()))
        if candidates:
            aliases[name.strip()] = candidates
    return aliases


MODEL_ALIASES = _parse_aliases(os.environ.get("MODEL_ALIASES", ""))
ALIASED_REAL_NAMES = {
    model for candidates in MODEL_ALIASES.values() for _, model in candidates
}

CACHE_FILE = MEMORY_DIR / ".rag_cache.json"

HEADER_RE = re.compile(r"^#{1,3}\s+.+$", re.MULTILINE)

# nomic-embed-text's context is ~8192 tokens; stay well under that in characters
# (roughly 4 chars/token for English prose) to leave margin for markdown/code.
MAX_CHUNK_CHARS = 6000


def _split_oversized(text: str, source: str) -> list[dict]:
    """Split a too-long chunk on paragraph boundaries, falling back to a hard
    character cut if a single paragraph is itself too long."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [{"source": source, "text": text}] if text.strip() else []

    parts = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= MAX_CHUNK_CHARS:
            parts.append(para)
        else:
            for i in range(0, len(para), MAX_CHUNK_CHARS):
                parts.append(para[i : i + MAX_CHUNK_CHARS])

    # Re-pack small paragraphs together up to the limit, so we don't end up
    # with a pile of tiny fragments losing surrounding context.
    packed: list[str] = []
    buf = ""
    for p in parts:
        if buf and len(buf) + 2 + len(p) > MAX_CHUNK_CHARS:
            packed.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        packed.append(buf)

    return [{"source": source, "text": p} for p in packed]


def split_into_chunks(text: str, source: str) -> list[dict]:
    """Split markdown text into chunks at ##/### headers, then further split
    any resulting chunk that's too long for the embedding model's context."""
    positions = [m.start() for m in HEADER_RE.finditer(text)]
    if not positions:
        stripped = text.strip()
        return _split_oversized(stripped, source) if stripped else []

    raw_chunks = []
    if positions[0] > 0:
        preamble = text[: positions[0]].strip()
        if preamble:
            raw_chunks.append(preamble)

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        chunk_text = text[start:end].strip()
        if chunk_text:
            raw_chunks.append(chunk_text)

    result = []
    for c in raw_chunks:
        result.extend(_split_oversized(c, source))
    return result


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    req = urllib.request.Request(
        f"{EMBED_UPSTREAM}/api/embed",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["embeddings"]


_health_cache: dict = {"host": None, "checked_at": 0.0}


def _fetch_tags(base_url: str) -> list[dict] | None:
    """Returns the raw model list from a box's /api/tags, or None if unreachable."""
    req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT_S) as r:
            if r.status != 200:
                return None
            return json.loads(r.read()).get("models", [])
    except Exception:
        return None


def _fetch_openai_models(base_url: str) -> list[dict] | None:
    """Returns an MLX/OpenAI-compatible server's model list, translated into
    Ollama's /api/tags entry shape (only the fields a frontend picker
    actually needs -- name/model/modified_at/size)."""
    req = urllib.request.Request(f"{base_url}/models", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT_S) as r:
            if r.status != 200:
                return None
            data = json.loads(r.read()).get("data", [])
    except Exception:
        return None

    return [
        {
            "name": m["id"],
            "model": m["id"],
            "modified_at": "",
            "size": 0,
            "details": {
                "family": "mlx",
                "parameter_size": "",
                "quantization_level": "",
            },
        }
        for m in data
        if m.get("id")
    ]


def _is_mlx_upstream(base_url: str) -> bool:
    return base_url in MLX_UPSTREAMS


def _openai_chat_to_ollama(openai_resp: dict, model_name: str) -> bytes:
    """Translates an MLX/OpenAI-format chat completion into Ollama's
    /api/chat response shape, since that's what every frontend here expects."""
    choice = (openai_resp.get("choices") or [{}])[0]
    message = choice.get("message", {"role": "assistant", "content": ""})
    usage = openai_resp.get("usage", {})
    return json.dumps(
        {
            "model": model_name,
            "created_at": "",
            "message": message,
            "done": True,
            "done_reason": choice.get("finish_reason", "stop"),
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }
    ).encode()


def _call_mlx_chat(base_url: str, payload: dict) -> bytes:
    """Sends an Ollama-shaped chat payload to an MLX/OpenAI-compatible
    upstream and returns the response translated back into Ollama's shape."""
    openai_payload = {"model": payload["model"], "messages": payload["messages"]}
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(openai_payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        openai_resp = json.loads(r.read())
    return _openai_chat_to_ollama(openai_resp, payload["model"])


def _is_reachable(base_url: str) -> bool:
    return _fetch_tags(base_url) is not None


def resolve_generation_host() -> str:
    """RTX-first with a cached health check, Mac Mini fallback -- mirrors
    the portfolio's host-failover.ts design (see feedback_ollama_failover_design.md).
    Used when the requested model exists on both boxes (or the model map is
    stale/unavailable), so "prefer RTX when it's up" remains the default."""
    now = time.monotonic()
    if now - _health_cache["checked_at"] < HEALTH_CACHE_TTL_S:
        return _health_cache["host"] or FALLBACK_UPSTREAM

    host = PRIMARY_UPSTREAM if _is_reachable(PRIMARY_UPSTREAM) else None
    _health_cache["host"] = host
    _health_cache["checked_at"] = now
    resolved = host or FALLBACK_UPSTREAM
    print(f"[memory-rag] health check: routing generation to {resolved}")
    return resolved


def invalidate_health_cache():
    _health_cache["checked_at"] = 0.0


_model_map_cache: dict = {"map": {}, "merged_tags": None, "built_at": 0.0}
MODEL_MAP_TTL_S = 30.0


def get_model_map() -> dict[str, list[str]]:
    """Maps each model name to the list of upstream base URLs that actually
    have it, refetched periodically. Also caches the merged /api/tags body
    (deduplicated by model name, Ollama boxes checked before MLX upstreams so
    a name collision favors the Ollama copy) so GET /api/tags requests don't
    refetch on every single call.

    Real model names that are folded into an alias (see MODEL_ALIASES) are
    hidden from BOTH the map and the merged /api/tags list -- only the alias
    name itself is exposed, so a frontend shows exactly one entry, not one
    per underlying backend."""
    now = time.monotonic()
    if now - _model_map_cache["built_at"] < MODEL_MAP_TTL_S and _model_map_cache["map"]:
        return _model_map_cache["map"]

    model_map: dict[str, list[str]] = {}
    merged_models: list[dict] = []
    seen_names: set[str] = set()

    for host in (PRIMARY_UPSTREAM, FALLBACK_UPSTREAM, *MLX_UPSTREAMS):
        models = (
            _fetch_openai_models(host) if _is_mlx_upstream(host) else _fetch_tags(host)
        )
        if models is None:
            continue
        for m in models:
            name = m.get("name")
            if not name or name in ALIASED_REAL_NAMES:
                continue
            model_map.setdefault(name, []).append(host)
            if name not in seen_names:
                merged_models.append(m)
                seen_names.add(name)

    for alias_name, candidates in MODEL_ALIASES.items():
        model_map[alias_name] = [host for host, _ in candidates]
        if alias_name not in seen_names:
            merged_models.append(
                {
                    "name": alias_name,
                    "model": alias_name,
                    "modified_at": "",
                    "size": 0,
                    "details": {
                        "family": "alias",
                        "parameter_size": "",
                        "quantization_level": "",
                    },
                }
            )
            seen_names.add(alias_name)

    if model_map:
        _model_map_cache["map"] = model_map
        _model_map_cache["merged_tags"] = json.dumps({"models": merged_models}).encode()
        _model_map_cache["built_at"] = now
        print(
            f"[memory-rag] model map refreshed: {len(model_map)} distinct model(s) across reachable boxes"
        )

    return _model_map_cache["map"]


_alias_health_cache: dict[str, dict] = {}


def resolve_alias(alias_name: str) -> tuple[str, str] | None:
    """Tries each candidate for an alias in order (e.g. RTX first, MLX
    fallback), health-checking + caching per alias the same way
    resolve_generation_host does for the plain RTX/Mac-Mini pair. Returns
    (host, real_model_name) for the first reachable candidate, or the last
    candidate unconditionally if none answer (so the caller still gets a
    real attempt + real error rather than silently doing nothing)."""
    candidates = MODEL_ALIASES.get(alias_name)
    if not candidates:
        return None

    cache = _alias_health_cache.setdefault(
        alias_name, {"host": None, "checked_at": 0.0}
    )
    now = time.monotonic()
    if now - cache["checked_at"] < HEALTH_CACHE_TTL_S and cache["host"] is not None:
        host = cache["host"]
        return next((h, m) for h, m in candidates if h == host)

    for host, model in candidates:
        if _is_reachable(host):
            cache["host"] = host
            cache["checked_at"] = now
            print(f"[memory-rag] alias {alias_name!r}: routing to {host} ({model})")
            return host, model

    # Nothing answered -- don't cache a dead result, just hand back the
    # first candidate so the caller's error message points at something real.
    cache["checked_at"] = 0.0
    print(
        f"[memory-rag] alias {alias_name!r}: no candidate reachable, trying first anyway"
    )
    return candidates[0]


def invalidate_alias_cache(alias_name: str):
    if alias_name in _alias_health_cache:
        _alias_health_cache[alias_name]["checked_at"] = 0.0


def resolve_host_for_model(model_name: str | None) -> str:
    """Routes based on which box(es) actually have the requested model.
    - On an MLX upstream (and nowhere else): use it directly, no RTX/Mac-Mini
      failover logic applies since it's a third, independent source.
    - Only on RTX: use RTX (or fall back to Mac Mini if RTX just went down).
    - Only on Mac Mini: use Mac Mini directly.
    - On both RTX and Mac Mini, or model unknown/unspecified: fall back to
      the health-checked RTX-first/Mac-Mini-fallback default
      (resolve_generation_host)."""
    if not model_name:
        return resolve_generation_host()

    if model_name in MODEL_ALIASES:
        resolved = resolve_alias(model_name)
        return resolved[0] if resolved else resolve_generation_host()

    hosts = get_model_map().get(model_name)
    if not hosts:
        # Model map may be stale or this box combo was momentarily
        # unreachable when it was built -- don't block on that, just use
        # the default failover.
        return resolve_generation_host()

    mlx_hosts = [h for h in hosts if _is_mlx_upstream(h)]
    if mlx_hosts and PRIMARY_UPSTREAM not in hosts and FALLBACK_UPSTREAM not in hosts:
        return mlx_hosts[0]

    if PRIMARY_UPSTREAM in hosts and FALLBACK_UPSTREAM in hosts:
        return resolve_generation_host()
    if PRIMARY_UPSTREAM in hosts:
        # RTX-only model: still worth a quick reachability check since RTX
        # is on-demand and may have gone down since the model map was built.
        return (
            PRIMARY_UPSTREAM if _is_reachable(PRIMARY_UPSTREAM) else FALLBACK_UPSTREAM
        )
    return FALLBACK_UPSTREAM


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryIndex:
    def __init__(self):
        self.chunks: list[dict] = []  # {source, text, embedding}
        self._load_or_build()

    def _file_signature(self, path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def _load_or_build(self):
        cached_by_file: dict[str, dict] = {}
        if CACHE_FILE.exists():
            try:
                cached_by_file = json.loads(CACHE_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                cached_by_file = {}

        md_files = sorted(
            p for p in MEMORY_DIR.glob("*.md") if p.name != CACHE_FILE.name
        )
        new_cache: dict[str, dict] = {}
        to_embed_texts: list[str] = []
        to_embed_meta: list[dict] = []

        for path in md_files:
            sig = self._file_signature(path)
            cached = cached_by_file.get(path.name)
            if cached and cached.get("sig") == sig:
                new_cache[path.name] = cached
                for c in cached["chunks"]:
                    self.chunks.append(
                        {
                            "source": path.name,
                            "text": c["text"],
                            "embedding": c["embedding"],
                        }
                    )
                continue

            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = split_into_chunks(text, path.name)
            new_cache[path.name] = {"sig": sig, "chunks": []}
            for c in chunks:
                to_embed_texts.append(c["text"])
                to_embed_meta.append({"file": path.name})

        if to_embed_texts:
            print(f"[memory-rag] embedding {len(to_embed_texts)} new/changed chunks...")
            embeddings = embed_texts(to_embed_texts)
            for meta, emb, text in zip(to_embed_meta, embeddings, to_embed_texts):
                fname = meta["file"]
                new_cache[fname]["chunks"].append({"text": text, "embedding": emb})
                self.chunks.append({"source": fname, "text": text, "embedding": emb})

        CACHE_FILE.write_text(json.dumps(new_cache))
        print(
            f"[memory-rag] index ready: {len(self.chunks)} chunks from {len(md_files)} files"
        )

    def search(
        self, query_embedding: list[float], top_k: int, threshold: float
    ) -> list[dict]:
        scored = [
            (cosine_similarity(query_embedding, c["embedding"]), c) for c in self.chunks
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in scored[:top_k] if score >= threshold]


def build_context_message(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    parts = ["Relevant background from the user's personal memory notes:\n"]
    for c in chunks:
        parts.append(f"[{c['source']}]\n{c['text']}\n")
    return "\n".join(parts)


import http.server
import socketserver
import threading

_index_lock = threading.Lock()
_index: MemoryIndex | None = None


def get_index() -> MemoryIndex:
    global _index
    with _index_lock:
        if _index is None:
            _index = MemoryIndex()
        return _index


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def _forward(self, path: str, body: bytes, inject: bool):
        is_generation = path in ("/api/chat", "/v1/chat/completions", "/api/generate")

        payload = None
        if body and (inject or is_generation):
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = None

        if inject and payload is not None:
            messages = payload.get("messages")
            if messages:
                last_user = next(
                    (
                        m["content"]
                        for m in reversed(messages)
                        if m.get("role") == "user"
                    ),
                    None,
                )
                if last_user:
                    try:
                        q_emb = embed_texts([last_user])[0]
                        chunks = get_index().search(q_emb, TOP_K, SIM_THRESHOLD)
                        ctx = build_context_message(chunks)
                        if ctx:
                            messages.insert(0, {"role": "system", "content": ctx})
                            payload["messages"] = messages
                            body = json.dumps(payload).encode()
                            print(
                                f"[memory-rag] injected {len(chunks)} chunk(s) for query: {last_user[:60]!r}"
                            )
                    except Exception as e:
                        print(
                            f"[memory-rag] retrieval failed, forwarding unmodified: {e}"
                        )

        # Route by which box actually has the requested model; anything
        # non-generation (e.g. a stray POST that isn't chat/generate) goes
        # straight to the fallback (Mac Mini, always-on) so it never blocks
        # on the RTX health check.
        requested_model = payload.get("model") if payload else None
        is_alias = requested_model in MODEL_ALIASES

        real_model = requested_model
        if is_generation and is_alias:
            resolved = resolve_alias(requested_model)
            host = resolved[0] if resolved else FALLBACK_UPSTREAM
            real_model = resolved[1] if resolved else requested_model
        elif is_generation:
            host = resolve_host_for_model(requested_model)
        else:
            host = FALLBACK_UPSTREAM

        # If this is an alias, the upstream (Ollama or MLX) has never heard
        # of the alias name -- rewrite the payload's model field to the real
        # underlying name before it goes anywhere, but keep the ALIAS name
        # for what we report back to the frontend so responses look
        # consistent regardless of which backend actually answered.
        if (
            is_generation
            and is_alias
            and payload is not None
            and real_model != requested_model
        ):
            payload = dict(payload, model=real_model)
            body = json.dumps(payload).encode()

        def _attempt(target_host: str):
            if is_generation and _is_mlx_upstream(target_host) and payload is not None:
                # MLX server only implements the OpenAI dialect and this
                # router only translates non-streaming responses -- force
                # stream:false for the upstream call regardless of what the
                # frontend asked for; the frontend still gets one complete
                # Ollama-shaped JSON response back, just not incrementally.
                mlx_payload = dict(payload)
                translated = _call_mlx_chat(target_host, mlx_payload)
                if is_alias:
                    # Report the alias name back, not MLX's real model id,
                    # so the frontend sees a consistent model name no matter
                    # which backend actually served this response.
                    translated_obj = json.loads(translated)
                    translated_obj["model"] = requested_model
                    translated = json.dumps(translated_obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(translated)
                return

            req = urllib.request.Request(
                f"{target_host}{path}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as upstream:
                resp_body = upstream.read()
                if is_alias:
                    try:
                        resp_obj = json.loads(resp_body)
                        resp_obj["model"] = requested_model
                        resp_body = json.dumps(resp_obj).encode()
                    except json.JSONDecodeError:
                        pass
                self.send_response(upstream.status)
                for k, v in upstream.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp_body)

        try:
            _attempt(host)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            if is_generation and is_alias:
                # Try the alias's remaining candidates in order, rewriting
                # the payload's model field for each one -- NOT the generic
                # Mac-Mini fallback, since that would send the wrong (still
                # rewritten-for-the-failed-candidate) model name to a host
                # that was never part of this alias's own candidate list.
                invalidate_alias_cache(requested_model)
                for next_host, next_model in MODEL_ALIASES[requested_model]:
                    if next_host == host:
                        continue
                    print(
                        f"[memory-rag] alias {requested_model!r}: {host} failed mid-request ({e}), trying {next_host}"
                    )
                    if payload is not None:
                        payload = dict(payload, model=next_model)
                        body = json.dumps(payload).encode()
                    try:
                        _attempt(next_host)
                        return
                    except Exception as e2:
                        print(f"[memory-rag] {next_host} also failed: {e2}")
                        continue
            elif (
                is_generation
                and host != FALLBACK_UPSTREAM
                and not _is_mlx_upstream(host)
            ):
                # RTX passed its cached health check but died mid-request (or
                # this is the first request after RTX went down since the
                # last check) -- invalidate the cache and retry once against
                # Mac Mini. Only meaningful for plain RTX/Mac-Mini requests:
                # an MLX-only model has no equivalent to retry on the
                # fallback Ollama host.
                print(
                    f"[memory-rag] {host} failed mid-request ({e}), retrying on {FALLBACK_UPSTREAM}"
                )
                invalidate_health_cache()
                try:
                    _attempt(FALLBACK_UPSTREAM)
                    return
                except Exception as e2:
                    print(f"[memory-rag] fallback also failed: {e2}")
            self.send_response(502)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        if self.path == "/api/show":
            self._handle_show(body)
            return

        inject = self.path in ("/api/chat", "/v1/chat/completions")
        self._forward(self.path, body, inject)

    def _handle_show(self, body: bytes):
        """AnythingLLM (and other frontends) call /api/show to check a
        model's capabilities (context length, tool-calling support, etc.)
        before enabling features like Agent Mode. For an alias, the real
        upstream has never heard of the alias name -- resolve it to a real
        candidate first. MLX has no /api/show equivalent, so if the alias's
        currently-active candidate is MLX, report against its first Ollama
        candidate instead (best-effort capability info, since Ollama-based
        /api/show is what capability-detection code actually expects)."""
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}

        requested_model = payload.get("model")
        target_model = requested_model
        target_host = None

        if requested_model in MODEL_ALIASES:
            candidates = MODEL_ALIASES[requested_model]
            ollama_candidates = [
                (h, m) for h, m in candidates if not _is_mlx_upstream(h)
            ]
            resolved = resolve_alias(requested_model)
            if resolved and not _is_mlx_upstream(resolved[0]):
                target_host, target_model = resolved
            elif ollama_candidates:
                target_host, target_model = ollama_candidates[0]
            # else: alias is entirely MLX-backed, target_host stays None,
            # handled by the no-Ollama-candidate check below.
        elif requested_model:
            # Not an alias -- route the same way a real chat request for
            # this model name would (RTX-only, Mac-Mini-only, on both, or
            # unknown), so /api/show hits whichever box actually has it
            # instead of blindly assuming Mac Mini.
            candidate_host = resolve_host_for_model(requested_model)
            if not _is_mlx_upstream(candidate_host):
                target_host = candidate_host

        if target_host is None:
            # Either an MLX-only alias/model (no /api/show equivalent to ask)
            # or a name that doesn't resolve to any Ollama host at all --
            # nothing meaningful to report either way.
            self.send_response(404)
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": f"model '{requested_model}' not found"}).encode()
            )
            return

        req = urllib.request.Request(
            f"{target_host}/api/show",
            data=json.dumps({**payload, "model": target_model}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as upstream:
                self.send_response(upstream.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(upstream.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        if self.path == "/api/tags":
            get_model_map()  # ensures merged_tags is populated/fresh
            merged = _model_map_cache["merged_tags"]
            if merged is not None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(merged)
                return
            # Neither box was reachable when we tried to build the map --
            # fall through to a direct passthrough attempt below.

        req = urllib.request.Request(f"{FALLBACK_UPSTREAM}{self.path}", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as upstream:
                self.send_response(upstream.status)
                for k, v in upstream.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(upstream.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())

    def log_message(self, fmt, *args):
        pass  # keep stdout to our own [memory-rag] lines


if __name__ == "__main__":
    print(f"[memory-rag] memory dir: {MEMORY_DIR}")
    print(f"[memory-rag] embed upstream: {EMBED_UPSTREAM}")
    print(f"[memory-rag] generation primary (RTX): {PRIMARY_UPSTREAM}")
    print(f"[memory-rag] generation fallback (Mac Mini): {FALLBACK_UPSTREAM}")
    print(f"[memory-rag] MLX upstream(s): {MLX_UPSTREAMS or '(none configured)'}")
    get_index()  # build index at startup
    print(f"[memory-rag] listening on 0.0.0.0:{PROXY_PORT}")
    with socketserver.ThreadingTCPServer(
        ("0.0.0.0", PROXY_PORT), ProxyHandler
    ) as httpd:
        httpd.serve_forever()
