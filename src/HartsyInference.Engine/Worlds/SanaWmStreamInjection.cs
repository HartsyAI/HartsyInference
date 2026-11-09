using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Worlds;

/// <summary>Recorded upstream tensors a SANA-WM stream runs on instead of its own, so a parity run compares the model and not
/// the random generator or the VAE encoder.</summary>
internal sealed record SanaWmStreamInjection
{
    /// <summary>The seed frame's normalized latent, host F32 <c>[1, 128, 1, 22, 40]</c>; the stream's phase takes ownership.
    /// Null encodes the seed image.</summary>
    public Tensor? FirstLatent { get; init; }

    /// <summary>The noise of a chunk by index, host F32 <c>[1, 128, T, 22, 40]</c>, borrowed for the call; a null result, or a
    /// null delegate, draws the seeded noise.</summary>
    public Func<int, Tensor?>? ChunkNoise { get; init; }
}
