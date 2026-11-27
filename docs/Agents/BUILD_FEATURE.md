# Build a feature

Use this file for any change to the engine, CLI, API or extension that is not a new model.

`HartsyInference.Engine` owns model lifecycle and load and generate dispatch. The CLI, HTTP API and SwarmUI are
consumers. Extend the service contract rather than copying orchestration into a consumer.

- Keep public compatibility with published consumers. Prefer compatible overloads and options. Coordinate any
  intentional break with packaging and extension updates.
- Keep HTTP and SSE in `HartsyInference.API`. Native and compatibility endpoints should share the Engine handlers, and
  the bounded default and long-running queue behaviour must stay intact.
- Trace shape, dtype, device and ownership across each boundary. Thread cancellation through the work and its cleanup.
  Keep three outcomes distinct: stream completion, cancellation and fault.
- Use source-generated JSON metadata at serialization boundaries. Put backend-specific math behind `IBackend`.
- Keep the CLI compiling, because it is the worked example for consumers. Verify the relevant API and engine behaviour,
  including cleanup after partial failure. An ordinary request failure must never terminate the host process.

Video features must follow the planning and exact-checkpoint identity rules in
[engine patterns](ENGINE_PATTERNS.md#video-planning-contract).
