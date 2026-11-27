using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>LTX-2 encoder downsampler (<c>LTXVideoDownsampler3d</c> in <c>AutoencoderKLLTX2Video</c>), the encoder mirror of
/// <see cref="LtxVaeUpsampler3d"/>. The input gets <c>stride_t − 1</c> leading frames copied from its start, then a
/// stride-1 causal conv produces <c>out/∏stride</c> channels that are space-to-depth packed by the stride. A residual
/// (space-to-depth of the same input, averaged over groups of <c>in·∏stride/out</c> channels) is added.</summary>
public sealed class LtxVaeDownsampler3d
{
    private readonly int _inC;
    private readonly int _outC;
    private readonly (int T, int H, int W) _stride;
    private readonly bool _isCausal;
    private readonly int _convOutC;
    private readonly int _groupSize;
    private CausalConv3d? _conv;

    public LtxVaeDownsampler3d(int inC, int outC, (int T, int H, int W) stride, bool isCausal = true)
    {
        if (inC < 1 || outC < 1)
            throw new ArgumentException($"Downsampler channels must be positive; got {inC}→{outC}.");
        if (stride.T < 1 || stride.H < 1 || stride.W < 1)
            throw new ArgumentException($"Downsampler stride must be positive; got ({stride.T},{stride.H},{stride.W}).", nameof(stride));
        int stProd = stride.T * stride.H * stride.W;
        if (outC % stProd != 0 || inC * stProd % outC != 0)
            throw new ArgumentException($"Downsampler {inC}→{outC} needs out divisible by {stProd} and in·{stProd} divisible by out.");
        _inC = inC;
        _outC = outC;
        _stride = stride;
        _isCausal = isCausal;
        _convOutC = outC / stProd;
        _groupSize = inC * stProd / outC;
    }

    /// <summary>Loads <c>{prefix}.conv.conv.weight</c> and its bias.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _conv = new CausalConv3d(w[$"{prefix}.conv.conv.weight"], VaeOps.Bias(w, $"{prefix}.conv.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv is not null) foreach (Tensor t in _conv.EnumerateWeights()) yield return t;
    }

    /// <summary>Downsamples <c>[B, in, F, H, W]</c> to <c>[B, out, F', H/st1, W/st2]</c>, with <c>F' = (F + st0 − 1) / st0</c>.
    /// The caller owns <paramref name="x"/> and the returned tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        if (_conv is null)
            throw new InvalidOperationException("LoadWeights must run before Forward.");
        int b = (int)x.Shape[0], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int st0 = _stride.T, st1 = _stride.H, st2 = _stride.W;
        if ((f + st0 - 1) % st0 != 0 || h % st1 != 0 || w % st2 != 0)
            throw new ArgumentException($"Input {x.Shape} is not divisible by stride ({st0},{st1},{st2}) after the frame prepend.", nameof(x));
        int fOut = (f + st0 - 1) / st0, hOut = h / st1, wOut = w / st2;

        bool prepend = st0 > 1;
        Tensor source = prepend ? PrependLeadingFrames(backend, x, st0 - 1) : x;
        Tensor residual = Residual(backend, source, b, fOut, hOut, wOut);
        Tensor conv = _conv.Forward(backend, source);
        if (prepend) source.Dispose();

        Tensor packed = new Tensor(new TensorShape([b, _outC, fOut, hOut, wOut]), DType.F32);
        PixelShuffles.SpaceToDepth(backend, packed, conv,
            ShuffleSpec.ChannelsFirst(b, fOut, hOut, wOut, _convOutC, st0, st1, st2, ChannelOrder.CTHW));
        conv.Dispose();
        backend.Add(packed, packed, residual);
        residual.Dispose();
        return packed;
    }

    /// <summary>Copies the first <paramref name="count"/> frames of <paramref name="x"/> in front of it (<c>torch.cat</c> on T).</summary>
    private static Tensor PrependLeadingFrames(IBackend backend, Tensor x, int count)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor head = new Tensor(new TensorShape([b, c, count, h, w]), DType.F32);
        StridedCopies.SliceAxis(backend, head, x, 2, 0, count);
        Tensor padded = new Tensor(new TensorShape([b, c, f + count, h, w]), DType.F32);
        Tensor[] parts = [head, x];
        backend.Concat(padded, parts, dim: 2);
        head.Dispose();
        return padded;
    }

    /// <summary>Space-to-depth of <paramref name="source"/>, then the mean of each <c>group</c> of consecutive channels.
    /// Output channel <c>c</c> averages stacked channels <c>c·group + g</c>, the order of diffusers'
    /// <c>unflatten(1, (-1, group)).mean(dim=2)</c>.</summary>
    private Tensor Residual(IBackend backend, Tensor source, int b, int fOut, int hOut, int wOut)
    {
        int st0 = _stride.T, st1 = _stride.H, st2 = _stride.W;
        Tensor stacked = new Tensor(new TensorShape([b, _inC * st0 * st1 * st2, fOut, hOut, wOut]), DType.F32);
        PixelShuffles.SpaceToDepth(backend, stacked, source,
            ShuffleSpec.ChannelsFirst(b, fOut, hOut, wOut, _inC, st0, st1, st2, ChannelOrder.CTHW));

        Tensor mean = new Tensor(new TensorShape([b, _outC, fOut, hOut, wOut]), DType.F32);
        backend.Fill(mean, 0f);
        for (int g = 0; g < _groupSize; g++)
        {
            StridedView members = StridedView.Of(stacked.Shape).Slice(1, g, _outC, _groupSize);
            backend.AccumulateStrided(mean, stacked, StridedCopySpec.Gather(members));
        }
        stacked.Dispose();
        backend.Scale(mean, mean, 1f / _groupSize);
        return mean;
    }
}
