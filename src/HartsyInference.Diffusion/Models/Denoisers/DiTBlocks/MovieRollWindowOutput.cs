using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The result of one MovieRoll inference-regime block over a window. <see cref="Image"/> is always set. <see cref="Text"/> is set
/// for a denoising window and null for an encode pass. <see cref="Keys"/> and <see cref="Values"/> are the window's own post-norm,
/// pre-RoPE K and V, token-major <c>[1, Sw, heads, D]</c>, set only for an encode pass; they are what the caller stores in the KV cache.
/// The caller owns every non-null tensor and disposes them.</summary>
public sealed record MovieRollWindowOutput(Tensor Image, Tensor? Text, Tensor? Keys, Tensor? Values) : IDisposable
{
    /// <summary>Disposes every non-null tensor. Idempotent: <see cref="Tensor.Dispose"/> is.</summary>
    public void Dispose()
    {
        Image.Dispose();
        Text?.Dispose();
        Keys?.Dispose();
        Values?.Dispose();
    }
}
