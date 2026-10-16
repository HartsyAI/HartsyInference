using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>One prompt's timestep-independent MOSS-Image conditioning at one resolution: the text tokens after the
/// refiner blocks and the joint sequence's RoPE tables. Built once per generation by
/// <see cref="MossImageTransformer.PrepareConditioning"/> and pinned on the backend as resident weights, so every step
/// (and a captured step graph) reads them without an upload; disposing unpins and frees them.</summary>
public sealed class MossImageConditioning : IDisposable
{
    private readonly IBackend _backend;
    private int _disposed;

    internal MossImageConditioning(IBackend backend, Tensor text, Tensor cos, Tensor sin, int height, int width)
    {
        _backend = backend;
        Text = text;
        Cos = cos;
        Sin = sin;
        Height = height;
        Width = width;
        backend.PreloadWeights([text, cos, sin]);
    }

    /// <summary>Refined text tokens <c>[1, L, hidden]</c> (F32).</summary>
    public Tensor Text { get; }

    /// <summary>RoPE cosines for <c>[image ; text]</c>, <c>[1, H·W + L, headDim]</c>.</summary>
    public Tensor Cos { get; }

    /// <summary>RoPE sines for <c>[image ; text]</c>, <c>[1, H·W + L, headDim]</c>.</summary>
    public Tensor Sin { get; }

    /// <summary>Packed latent rows this conditioning was built for.</summary>
    public int Height { get; }

    /// <summary>Packed latent columns this conditioning was built for.</summary>
    public int Width { get; }

    /// <summary>Text tokens.</summary>
    public int TextLength => (int)Text.Shape[1];

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _backend.FreeWeights([Text, Cos, Sin]);
        Text.Dispose();
        Cos.Dispose();
        Sin.Dispose();
    }
}
