# Build a feature

HartsyInference.Engine owns model lifecycle and load/generate dispatch. CLI, HTTP API and SwarmUI are consumers; extend the service contract rather than duplicating orchestration.

- Preserve public compatibility with published consumers. Prefer compatible overloads/options; coordinate intentional breaks with packaging and extension updates.
- Keep HTTP/SSE in HartsyInference.API. Native and compatibility endpoints should share Engine handlers. Preserve bounded default and long-running queue behavior.
- Trace shape, dtype, device and ownership across boundaries. Thread cancellation through work and cleanup; distinguish stream completion, cancellation and faults.
- Use source-generated JSON metadata at serialization boundaries. Backend-specific math belongs behind IBackend.
- Keep CLI compiling as a consumer example; verify relevant API/engine behavior and partial-failure cleanup. Ordinary request failures must not terminate the host.

Video features must follow the planning and exact-checkpoint identity rules in [shared architecture](AGENTS.md).
