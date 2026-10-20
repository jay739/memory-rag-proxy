# memory-rag-proxy

<p align="center">
  <img src="docs/logo.svg" alt="memory-rag-proxy logo" width="128" />
</p>

A single-file HTTP proxy that sits in front of multiple local LLM inference
backends (Ollama and MLX-compatible servers) and does two things:

1. **Retrieval-augmented generation** over a folder of Markdown notes, using
   local embeddings, injected as a system message before the request is
   forwarded upstream.
2. **Health-checked routing and failover** across heterogeneous, personally
   owned hardware: an on-demand GPU box, an always-on Mac, and any number of
   OpenAI-API-compatible MLX servers, presented to client apps (Open WebUI,
   AnythingLLM, anything speaking Ollama's API) as a single consistent
   endpoint.

## Why

Running local models across more than one machine surfaces a systems problem
that has nothing to do with the model itself: which box is actually up right
now, and how do you avoid a slow, hung request when it isn't.

This proxy was built to run a "general-purpose" model alias that's backed by
two independent servers on different hardware (a discrete-GPU box that's
powered on only when needed, and an always-on Mac serving the same model
family via MLX). Callers ask for one model name; the proxy resolves it to
whichever backend is actually reachable, with:

- **A short, cached health check** on the primary host, so a powered-off GPU
  box doesn't add its full connect-timeout to every single request — the
  proxy checks once and trusts the result for a configurable TTL.
- **Mid-request retry against the fallback** if the primary passes its cached
  health check but then fails during the actual call (e.g. it powers off
  between the check and the request).
- **Per-alias ordered candidate lists**, so a model name can fail over across
  more than two backends, and across different API dialects (native Ollama
  vs. OpenAI-compatible MLX) transparently to the caller.
- **A merged `/api/tags` model list** across all reachable backends, with
  alias real-names hidden so a frontend model picker shows one clean entry
  instead of one per physical machine.

## Design notes

- Config is entirely environment-variable driven (see `.env.example`), so the
  same script runs unmodified on any host.
- The retrieval index is chunked on Markdown headers, with oversized chunks
  further split on paragraph boundaries to stay under the embedding model's
  context limit, and is disk-cached by file mtime/size so unchanged notes
  aren't re-embedded on every restart.
- MLX (OpenAI-dialect) responses are translated back into Ollama's response
  shape at the boundary, so every client-facing response looks the same
  regardless of which backend actually served it.
- No framework — stdlib `http.server` + `urllib`, intentionally, since the
  whole proxy is a few hundred lines and a framework would add more surface
  than it would save.

## Usage

```bash
cp .env.example .env   # edit MEMORY_DIR and your actual host addresses
export $(grep -v '^#' .env | xargs)
python3 memory_rag_proxy.py
```

Point any Ollama-API-compatible client at `http://<this-host>:11435`.

## Requirements

- Python 3.10+ (stdlib only, no dependencies)
- One or more Ollama and/or OpenAI-compatible (MLX) inference servers
  reachable over HTTP
- A folder of Markdown files to use as the retrieval corpus

## License

MIT
