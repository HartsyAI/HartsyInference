using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Main branch of a SANA-WM softmax attention layer (upstream <c>_forward_softmax_attn</c>): the GDN q/k/v projection,
/// full-channel RMSNorm, interleaved RoPE on every head dim, then softmax over all tokens with scale <c>headDim^-1/2</c>.
/// Returns the branch before the camera sum, output gate and projection.</summary>
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
    /// <param name="qkvWeight">F32 fused projection <c>[3C, C]</c>, no bias; rows <c>[0, C)</c> are q, then k, then v.</param>
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
        SanaWmTensors.Require(qkvWeight, nameof(qkvWeight), DType.F32, [3L * _channels, _channels]);
        SanaWmTensors.Require(qNorm, nameof(qNorm), DType.F32, [_channels]);
        SanaWmTensors.Require(kNorm, nameof(kNorm), DType.F32, [_channels]);
        _qkvWeight = qkvWeight;
        _qNorm = qNorm;
        _kNorm = kNorm;
    }

    /// <summary>Softmax branch output <c>[B, N, C]</c> in F32. The caller disposes it.</summary>
    /// <param name="x">F32 layer input <c>[B, N, C]</c> with <c>N = frames * latentHeight * latentWidth</c>, frame-major.</param>
    /// <param name="ropeCos">Cos table <c>[N, headDim]</c> in interleaved-pair layout, F32.</param>
    /// <param name="ropeSin">Sin table in the same layout, F32.</param>
    public Tensor Forward(Tensor x, int frames, int latentHeight, int latentWidth, Tensor ropeCos, Tensor ropeSin)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(ropeCos);
        ArgumentNullException.ThrowIfNull(ropeSin);
        if (frames <= 0 || latentHeight <= 0 || latentWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(frames), "Frames and latent grid must be positive.");
        int tokens = checked(frames * latentHeight * latentWidth);
        if (x.DType != DType.F32 || x.Shape.Rank != 3 || x.Shape[1] != tokens || x.Shape[2] != _channels)
            throw new ArgumentException($"x must be F32 [B, {tokens}, {_channels}]; got {x.DType} {x.Shape}.", nameof(x));
        SanaWmTensors.Require(ropeCos, nameof(ropeCos), DType.F32, [tokens, _headDim]);
        SanaWmTensors.Require(ropeSin, nameof(ropeSin), DType.F32, [tokens, _headDim]);

        long batch = x.Shape[0];
        TensorShape rows = new([batch * tokens, _channels]);
        TensorShape headMajor = new([batch, _heads, tokens, _headDim]);
        TensorShape tokenMajor = new([batch, tokens, _heads, _headDim]);
        List<Tensor> scratch = [];
        Tensor output = new(x.Shape, DType.F32, _backend.Device);
        bool succeeded = false;
        try
        {
            Tensor xRows = x.Reshape(rows);
            Tensor q = Projected(scratch, xRows, 0, rows);
            Tensor k = Projected(scratch, xRows, _channels, rows);
            Tensor v = Projected(scratch, xRows, 2 * _channels, rows);
            Tensor qRot = Rotated(scratch, Normed(scratch, q, _qNorm), ropeCos, ropeSin, tokens, batch);
            Tensor kRot = Rotated(scratch, Normed(scratch, k, _kNorm), ropeCos, ropeSin, tokens, batch);

            Tensor qh = NewTensor(scratch, headMajor);
            Tensor kh = NewTensor(scratch, headMajor);
            Tensor vh = NewTensor(scratch, headMajor);
            _backend.Permute0213(qh, qRot.Reshape(tokenMajor), tokens, _heads, _headDim);
            _backend.Permute0213(kh, kRot.Reshape(tokenMajor), tokens, _heads, _headDim);
            _backend.Permute0213(vh, v.Reshape(tokenMajor), tokens, _heads, _headDim);

            Tensor attended = NewTensor(scratch, headMajor);
            float scale = (float)(1.0 / Math.Sqrt(_headDim));
            _backend.ScaledDotProductAttention(attended, qh, kh, vh, null, scale);
            _backend.Permute0213(output.Reshape(tokenMajor), attended, _heads, tokens, _headDim);
            succeeded = true;
            return output;
        }
        finally
        {
            if (!succeeded)
                output.Dispose();
            SanaWmTensors.DisposeAll(scratch);
        }
    }

    // q, k or v rows of the fused projection for every token of every batch item.
    private Tensor Projected(List<Tensor> scratch, Tensor xRows, int rowOffset, TensorShape rows)
    {
        Tensor projected = NewTensor(scratch, new TensorShape([rows[0], _channels]));
        _backend.LinearWeightRows(projected, xRows, _qkvWeight, null, rowOffset, _channels);
        return projected;
    }

    private Tensor Normed(List<Tensor> scratch, Tensor input, Tensor weight)
    {
        Tensor normed = NewTensor(scratch, input.Shape);
        _backend.RmsNorm(normed, input, weight, RmsEpsilon);
        return normed;
    }

    // The RoPE table covers one batch item's tokens, so the rotation runs per item on views of the copy.
    private Tensor Rotated(List<Tensor> scratch, Tensor source, Tensor cos, Tensor sin, int tokens, long batch)
    {
        Tensor rotated = NewTensor(scratch, source.Shape);
        _backend.CopyTo(rotated, source);
        for (long b = 0; b < batch; b++)
        {
            Tensor item = rotated.SliceRows(b * tokens, tokens);
            scratch.Add(item);
            _backend.Rotary(RotaryCall.Rows(item, cos, sin, tokens, _heads, _headDim, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair));
        }

        return rotated;
    }

    private Tensor NewTensor(List<Tensor> scratch, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32, _backend.Device);
        scratch.Add(tensor);
        return tensor;
    }
}
