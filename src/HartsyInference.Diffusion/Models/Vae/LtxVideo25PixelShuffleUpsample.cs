using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Linear channel expansion followed by a channels-last 3D pixel shuffle: <c>proj</c> widens each token to
/// <c>∏stride · in / reduction</c> channels, which then unpack into a <c>stride</c>-sized (t, h, w) sub-block. A
/// temporal stride of 2 duplicates the leading frame (the causal shuffle has nothing before t=0 to interpolate from),
/// so that frame is dropped whenever this call owns the latent's temporal origin.</summary>
internal sealed class LtxVideo25PixelShuffleUpsample
{
    private readonly int _inChannels;
    private readonly (int T, int H, int W) _stride;
    private readonly int _projChannels;

    private Tensor? _projWeight, _projBias;

    public LtxVideo25PixelShuffleUpsample(int inChannels, (int T, int H, int W) stride, int reduction)
    {
        _inChannels = inChannels;
        _stride = stride;
        _projChannels = stride.T * stride.H * stride.W * inChannels / reduction;
        OutChannels = _projChannels / (stride.T * stride.H * stride.W);
    }

    public int OutChannels { get; }

    public void LoadWeights(LtxVideo25WeightScope scope, string prefix)
    {
        _projWeight = scope.Raw($"{prefix}.proj.weight");
        _projBias = scope.OptionalF32($"{prefix}.proj.bias");
        if (_projWeight.Shape[0] != _projChannels || _projWeight.Shape[1] != _inChannels)
            throw new InvalidOperationException($"'{prefix}.proj.weight' is {_projWeight.Shape}, expected [{_projChannels}, {_inChannels}].");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_projWeight is not null) yield return _projWeight;
        if (_projBias is not null) yield return _projBias;
    }

    /// <summary>Upsamples <paramref name="x"/> <c>[t·h·w, in]</c> to <c>[outT·outH·outW, OutChannels]</c>. Pointwise
    /// along the temporal axis, so <paramref name="chunkBytes"/> bounds the widened projection without changing a
    /// single output value.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int t, int h, int w, bool dropLeadingFrame,
        long chunkBytes, out int outT, out int outH, out int outW)
    {
        int p1 = _stride.T, p2 = _stride.H, p3 = _stride.W;
        int dropped = p1 == 2 && dropLeadingFrame ? 1 : 0;
        outT = t * p1 - dropped;
        outH = h * p2;
        outW = w * p3;

        int outChannels = OutChannels;
        long plane = (long)h * w;
        Tensor result = new Tensor(new TensorShape((long)outT * outH * outW, outChannels), DType.F32);
        long bytesPerToken = (long)(_inChannels + _projChannels) * sizeof(float);
        int step = LtxVideo25TemporalChunks.FramesPerChunk(chunkBytes, plane, bytesPerToken, t);
        for (int start = 0; start < t; start += step)
        {
            int count = Math.Min(step, t - start);
            using Tensor projected = new Tensor(new TensorShape((long)count * plane, _projChannels), DType.F32);
            using Tensor? slice = count == t ? null : new Tensor(new TensorShape((long)count * plane, _inChannels), DType.F32);
            if (slice is not null) backend.SliceRowsGeneric(slice, x, checked((int)((long)start * plane)));
            backend.Linear(projected, slice ?? x, _projWeight!, _projBias);
            backend.Ltx25PixelShuffle(result, projected, start, h, w, p1, p2, p3, outChannels, dropped, outH, outW);
        }
        return result;
    }
}
