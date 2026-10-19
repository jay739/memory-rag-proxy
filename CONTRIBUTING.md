# Contributing to memory-rag-proxy

Thanks for considering a contribution. Issues, discussions and PRs are all
welcome.

## Getting set up

Python 3.10 or newer, standard library only. Copy `.env.example` to `.env`, set `MEMORY_DIR` to a folder of Markdown notes, then run `python memory_rag_proxy.py`. Point any Ollama client at port 11435.

## Running the checks

Install the dev tools with `pip install pytest ruff`, then run `ruff check --select=E9,F63,F7,F82 .` and `pytest -q tests`. The tests need no running Ollama: embedding calls are stubbed. CI runs both on every PR.

## Ground rules

- Open an issue or discussion before large changes, so effort isn't wasted.
- Keep PRs focused: one change per PR.
- New behavior needs a test where the repo has a test suite.
- CI must pass before merge.

## Licensing

By contributing you agree your contributions are licensed under the repo's
MIT license.
