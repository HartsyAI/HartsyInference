using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Pixal3D;

/// <summary>What a Pixal3D flow cross-attends to: the DINOv3 class and register tokens (<c>Global</c>, <c>[1, 1 + registers, 1024]</c>) and one
/// view-aligned feature row per token of the flow (<c>Proj</c>, <c>[1, tokens, projChannels]</c>). Owns both tensors.</summary>
public sealed class Pixal3DCondition : IDisposable
{
    private int _disposed;

    /// <param name="global">Takes ownership.</param>
    /// <param name="proj">Takes ownership.</param>
    public Pixal3DCondition(Tensor global, Tensor proj)
    {
        Global = global ?? throw new ArgumentNullException(nameof(global));
        Proj = proj ?? throw new ArgumentNullException(nameof(proj));
        if (global.Shape.Rank != 3 || proj.Shape.Rank != 3 || global.Shape[0] != 1 || proj.Shape[0] != 1)
            throw new ArgumentException($"Conditioning must be [1, rows, channels] pairs; got {global.Shape} and {proj.Shape}.");
    }

    public Tensor Global { get; }

    public Tensor Proj { get; }

    /// <summary>The unconditional counterpart: zeros of both shapes, as upstream builds its negative. The flow still adds <c>proj_linear</c>'s bias for it.</summary>
    public Pixal3DCondition ZerosLike()
    {
        Tensor global = new(Global.Shape, DType.F32);
        Tensor proj = new(Proj.Shape, DType.F32);
        global.AsSpan<float>().Clear();
        proj.AsSpan<float>().Clear();
        return new Pixal3DCondition(global, proj);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Global.Dispose();
        Proj.Dispose();
    }
}
