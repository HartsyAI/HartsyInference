# HartsyInference — Gemini Agent Instructions

HartsyInference is a pure C#/.NET AI inference engine (net8.0 + net10.0) covering LLM text generation,
image generation, speech-to-text, text-to-speech, voice conversion, music, vision, object detection, video
generation, 3D mesh, and interactive world models.

**The instructions for every coding agent are the same, and they live in one place. Read
[`docs/Agents/AGENTS.md`](docs/Agents/AGENTS.md) first** — it carries the shared design rules, the core
engine patterns, and a routing table pointing at the specialized agent file for your task
(add a model, build a feature, audit, kernels, research, cleanup).

Then read [`docs/CODE_STYLE.md`](docs/CODE_STYLE.md), which is mandatory and non-negotiable, and
[`docs/README.md`](docs/README.md) for what each docs folder holds.

For current state: per-model status is indexed in
[`docs/Checklists/MODEL_STATUS.md`](docs/Checklists/MODEL_STATUS.md), open cross-cutting work is in
[`ROADMAP.md`](docs/Checklists/ROADMAP.md), and
[`TROUBLESHOOTING.md`](docs/Checklists/TROUBLESHOOTING.md) is the bring-up debugging reference — read it
before debugging a model that is wrong, crashes, or is slow.

[`CLAUDE.md`](CLAUDE.md) contains the same orientation in Claude Code's format; the two files are
interchangeable.
