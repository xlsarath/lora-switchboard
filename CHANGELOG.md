# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-16

First public release.

### Added

- SQLite registry of adapters, immutable versions (`name@vN`), aliases and an
  append-only event log (`register`, `load`, `load_fail`, `smoke_pass`,
  `smoke_fail`, `promote`, `rollback`, `unload`, `unload_fail`, `reload`).
- Rollout flow: `deploy` loads a version into vLLM, runs a YAML smoke suite
  (`contains`, `not_contains`, `regex`, `json_schema`, `max_latency_ms`) and
  atomically promotes an alias; failed versions are unloaded and recorded.
- `rollback` to the previous good version (or `--to vN`), reloading it if it
  had been unloaded; timings reported for every step.
- OpenAI-compatible routing proxy (`/v1/chat/completions`, `/v1/completions`,
  `/v1/models`, other `/v1/*` POSTs) with SSE pass-through; requests keep their
  target across flips and old versions unload only after a drain.
- Automatic `gc` keeping `LSB_KEEP_LOADED` versions per adapter, and a
  reconcile loop that re-loads alias targets after a vLLM restart.
- Admin REST API (optional bearer token), `lora-switchboard` CLI,
  Prometheus metrics at `/metrics`, `/healthz` and `/readyz` probes.
- `lora-switchboard fake-vllm`: a GPU-free stand-in for vLLM used by the tests,
  the demo and docker compose.
- Dockerfile (non-root), docker-compose quickstart, CI and release workflows.

[Unreleased]: https://github.com/xlsarath/lora-switchboard/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/xlsarath/lora-switchboard/releases/tag/v0.1.0
