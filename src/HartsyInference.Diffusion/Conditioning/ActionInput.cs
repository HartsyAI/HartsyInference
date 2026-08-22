namespace HartsyInference.Conditioning;

/// <summary>One frame's worth of raw action state from the application. <see cref="Payload"/> is a model-defined byte layout (key states, mouse deltas, camera pose, …) interpreted by the model's <see cref="IActionEncoder"/>; the engine never parses it. See <c>docs/Research/INTERACTIVE_INFERENCE.md</c> § 2.</summary>
/// <param name="Payload">Model-defined bytes; the encoder documents its expected layout.</param>
/// <param name="FrameIndex">Monotonic frame counter (RGB frame rate), for positional conditioning.</param>
/// <param name="TimestampNanos">Wall-clock capture time in nanoseconds (diagnostics / replay).</param>
public readonly record struct ActionInput(ReadOnlyMemory<byte> Payload, long FrameIndex, long TimestampNanos);
