using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Main branch of a SANA-WM softmax attention layer (upstream <c>_forward_softmax_attn</c>): the GDN q/k/v projection,
/// full-channel RMSNorm, interleaved RoPE on every head dim, then softmax over all tokens with scale <c>headDim^-1/2</c>.
/// Returns the branch before the camera sum, output gate and projection. Runs one batch item at a time.</summary>
public sealed class SanaWmSoftmaxAttention
{
    private const float RmsEpsilon = 1e-5f;

    private readonly IBackend _backend;
    private readonly Tensor _qkvWeight;
    private readonly Tensor _qNorm;
    private readonly Tensor _kNorm;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _channels;

    /// <param name="backend">Backend the layer runs on.</param>
    /// <param name="qkvWeight">F32 or BF16 fused projection <c>[3C, C]</c>, no bias, borrowed; rows <c>[0, C)</c> are q, then k, then v.</param>
    /// <param name="qNorm">F32 full-channel RMSNorm weight for q <c>[C]</c>.</param>
    /// <param name="kNorm">F32 full-channel RMSNorm weight for k <c>[C]</c>.</param>
    /// <param name="heads">Attention heads (20 for SANA-WM).</param>
    /// <param name="headDim">Head dimension (112 for SANA-WM).</param>
    public SanaWmSoftmaxAttention(IBackend backend, Tensor qkvWeight, Tensor qNorm, Tensor kNorm, int heads, int headDim)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(qkvWeight);
        ArgumentNullException.ThrowIfNull(qNorm);
        ArgumentNullException.ThrowIfNull(kNorm);
        if (heads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(heads), "Heads and head dim must be positive.");

        _backend = backend;
        _heads = heads;
        _headDim = headDim;
        _channels = checked(heads * headDim);
        SanaWmTensors.RequireMatrix(qkvWeight, nameof(qkvWeight), 3L * _channels, _channels);
        SanaWmTensors.Require(qNorm, nameof(qNorm), DType.F32, [_channels]);
        SanaWmTensors.Require(kNorm, nameof(kNorm), DType.F32, [_channels]);
        _qkvWeight = qkvWeight;
        _qNorm = qNorm;
        _kNorm = kNorm;
    }

    /// <summary>The tensors the layer reads, for preloading on a device backend.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => [_qkvWeight, _qNorm, _kNorm];

    /// <summary>Softmax branch output <c>[B, N, C]</c> in F32. The caller disposes it.</summary>
    /// <param name="x">F32 layer input <c>[B, N, C]</c> with <c>N = frames * latentHeight * latentWidth</c>, frame-major.</param>
    /// <param name="ropeCos">Cos table <c>[N, headDim]</c> in interleaved-pair layout, F32.</param>
    /// <param name="ropeSin">Sin table in the same layout, F32.</param>
    public Tensor Forward(Tensor x, int frames, int latentHeight, int latentWidth, Tensor ropeCos, Tensor ropeSin)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (frames <= 0 || latentHeight <= 0 || latentWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(frames), "Frames and latent grid must be positive.");
        int tokens = checked(frames * latentHeight * latentWidth);
        if (x.DType != DType.F32 || x.Shape.Rank != 3 || x.Shape[1] != tokens || x.Shape[2] != _channels)
            throw new ArgumentException($"x must be F32 [B, {tokens}, {_channels}]; got {x.DType} {x.Shape}.", nameof(x));

        int batch = (int)x.Shape[0];
        long itemElements = (long)tokens * _channels;
        using SanaWmScratch scratch = new();
        Tensor output = scratch.New(x.Shape);
        for (int b = 0; b < batch; b++)
        {
            using SanaWmScratch item = new();
            Tensor a = item.Track(SanaWmTensors.Slice(_backend, x, b * itemElements, new TensorShape(tokens, _channels)));
            Tensor attended = item.Track(ForwardItem(a, tokens, ropeCos, ropeSin));
            SanaWmTensors.Write(_backend, output, attended, b * itemElements);
        }

        return scratch.Detach(output);
    }

    /// <summary>One batch item: <paramref name="a"/> is F32 token-major with <c>tokens·C</c> elements. Returns <c>[N, C]</c>.</summary>
    internal Tensor ForwardItem(Tensor a, int tokens, Tensor ropeCos, Tensor ropeSin)
    {
        if (a.DType != DType.F32 || a.ElementCount != (long)tokens * _channels)
            throw new ArgumentException($"The item input must be F32 with {tokens}x{_channels} elements; got {a.DType} {a.Shape}.", nameof(a));
        SanaWmTensors.Require(ropeCos, nameof(ropeCos), DType.F32, [tokens, _headDim]);
        SanaWmTensors.Require(ropeSin, nameof(ropeSin), DType.F32, [tokens, _headDim]);

        using SanaWmScratch scratch = new();
        Tensor qh = Heads(scratch, a, 0, _qNorm, ropeCos, ropeSin, tokens);
        Tensor kh = Heads(scratch, a, _channels, _kNorm, ropeCos, ropeSin, tokens);
        Tensor vh = Heads(scratch, a, 2 * _channels, null, ropeCos, ropeSin, tokens);
        Tensor attended = scratch.New(1, _heads, tokens, _headDim);
        _backend.ScaledDotProductAttention(attended, qh, kh, vh, null, (float)(1.0 / Math.Sqrt(_headDim)));
        scratch.Release(qh);
        scratch.Release(kh);
        scratch.Release(vh);
        Tensor output = scratch.New(tokens, _channels);
        _backend.Permute0213(output, attended, _heads, tokens, _headDim);
        return scratch.Detach(output);
    }

    // q, k (normed and rotated) or v rows of the fused projection, head-major [1, H, N, D].
    private Tensor Heads(SanaWmScratch scratch, Tensor a, int rowOffset, Tensor? norm, Tensor cos, Tensor sin, int tokens)
    {
        Tensor projected = scratch.New(tokens, _channels);
        _backend.LinearWeightRows(projected, a, _qkvWeight, null, rowOffset, _channels);
        if (norm is not null)
        {
            Tensor normed = scratch.New(tokens, _channels);
            _backend.RmsNorm(normed, projected, norm, RmsEpsilon);
            scratch.Release(projected);
            projected = normed;
            _backend.Rotary(RotaryCall.Rows(projected, cos, sin, tokens, _heads, _headDim, RotaryPairing.Interleaved,
                RopeTableLayout.InterleavedPair));
        }

        Tensor heads = scratch.New(1, _heads, tokens, _headDim);
        _backend.Permute0213(heads, projected, tokens, _heads, _headDim);
        scratch.Release(projected);
        return heads;
    }
}
