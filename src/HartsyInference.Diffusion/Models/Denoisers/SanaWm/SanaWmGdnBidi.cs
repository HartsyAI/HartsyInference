using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Main branch of one SANA-WM bidirectional Gated-DeltaNet attention layer. Returns the branch before the camera
/// sum, the output gate and the projection. The forward scan includes the current frame; the reverse scan excludes it.</summary>
public sealed class SanaWmGdnBidi : IDisposable
{
    private const float RmsEpsilon = 1e-5f;
    private const float DenominatorEpsilon = 1e-8f;
    private const int ConvTaps = 4;
    private const int ConvHalfWidth = ConvTaps - 1;

    // Offsets -3..3 of conv(x) + conv(flip x).flip - w[3] x, which is the symmetric kernel [w0 w1 w2 w3 w2 w1 w0].
    private static readonly int[] SymmetricTap = [0, 1, 2, 3, 2, 1, 0];

    private readonly IBackend _backend;
    private readonly List<Tensor> _owned = [];
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _channels;
    private readonly Tensor _qkvWeight;
    private readonly Tensor _convTaps;
    private readonly Tensor _betaWeight;
    private readonly Tensor _betaBias;
    private readonly Tensor _gateWeight;
    private readonly Tensor _gateBias;
    private readonly Tensor _negRate;
    private readonly Tensor _qNorm;
    private readonly Tensor _kNorm;
    private bool _disposed;

    /// <param name="backend">Backend the layer runs on; weights of another device are not converted here.</param>
    /// <param name="weights">Layer weights. F32 tensors are referenced; BF16 tensors are converted into owned copies.</param>
    /// <param name="heads">Attention heads (20 for SANA-WM).</param>
    /// <param name="headDim">Head dimension (112 for SANA-WM); <c>heads * headDim</c> is the channel count.</param>
    public SanaWmGdnBidi(IBackend backend, SanaWmGdnWeights weights, int heads, int headDim)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(weights);
        if (heads <= 0 || headDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(heads), "Heads and head dim must be positive.");
        }

        int channels = checked(heads * headDim);
        RequireShape(weights.Qkv, nameof(weights.Qkv), [3L * channels, channels]);
        RequireShape(weights.ConvK, nameof(weights.ConvK), [channels, 1, ConvTaps]);
        RequireShape(weights.BetaWeight, nameof(weights.BetaWeight), [heads, channels]);
        RequireShape(weights.BetaBias, nameof(weights.BetaBias), [heads]);
        RequireShape(weights.GateWeight, nameof(weights.GateWeight), [heads, channels]);
        RequireShape(weights.GateBias, nameof(weights.GateBias), [heads]);
        RequireShape(weights.ALog, nameof(weights.ALog), [heads]);
        RequireShape(weights.DtBias, nameof(weights.DtBias), [heads]);
        RequireShape(weights.QNorm, nameof(weights.QNorm), [channels]);
        RequireShape(weights.KNorm, nameof(weights.KNorm), [channels]);

        _backend = backend;
        _heads = heads;
        _headDim = headDim;
        _channels = channels;
        bool built = false;
        try
        {
            _qkvWeight = Prepare(weights.Qkv);
            _betaWeight = Prepare(weights.BetaWeight);
            _betaBias = Prepare(weights.BetaBias);
            _gateWeight = Prepare(weights.GateWeight);
            _qNorm = Prepare(weights.QNorm);
            _kNorm = Prepare(weights.KNorm);

            Tensor conv = Prepare(weights.ConvK);
            _convTaps = NewTensor([ConvTaps, channels]);
            backend.Transpose2D(_convTaps, conv.Reshape(new TensorShape([1, channels, ConvTaps])), channels, ConvTaps);

            Tensor gateBias = Prepare(weights.GateBias);
            Tensor dtBias = Prepare(weights.DtBias);
            _gateBias = NewTensor([heads]);
            backend.Add(_gateBias, gateBias, dtBias);

            Tensor aLog = Prepare(weights.ALog);
            using Tensor rate = new(new TensorShape([heads]), DType.F32, backend.Device);
            SanaWmGdnElementwise.Exp(backend, rate, aLog);
            _negRate = NewTensor([heads]);
            backend.Scale(_negRate, rate, -1f);
            built = true;
        }
        finally
        {
            if (!built)
            {
                ReleaseOwned();
            }
        }
    }

    /// <summary>Main-branch output <c>[B, N, C]</c> in F32, before the camera sum, output gate and projection. Caller disposes it.</summary>
    /// <param name="x">Layer input <c>[B, N, C]</c>, F32 or BF16, with <c>N = frames * latentHeight * latentWidth</c> in frame-major order.</param>
    /// <param name="ropeCos">Cos table <c>[N, headDim]</c> in interleaved-pair layout.</param>
    /// <param name="ropeSin">Sin table in the same layout; with <paramref name="ropeCos"/> it is the complex table <c>cos + i sin</c>.</param>
    public Tensor Forward(Tensor x, int frames, int latentHeight, int latentWidth, Tensor ropeCos, Tensor ropeSin)
        => Run(x, frames, latentHeight, latentWidth, ropeCos, ropeSin, beta: null, decay: null);

    /// <summary>Main-branch output plus the frame gates the camera branch shares: beta <c>[B, H, T, S]</c> and decay
    /// <c>[B, H, T]</c>, both F32 and head-major. The caller disposes all three tensors.</summary>
    public (Tensor Output, Tensor Beta, Tensor Decay) ForwardWithGates(Tensor x, int frames, int latentHeight, int latentWidth,
        Tensor ropeCos, Tensor ropeSin)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(x);
        if (x.Shape.Rank != 3 || frames <= 0 || latentHeight <= 0 || latentWidth <= 0)
        {
            throw new ArgumentException("x must be [B, N, C] and the latent grid must be positive.", nameof(x));
        }

        int batch = (int)x.Shape[0];
        int tokensPerFrame = checked(latentHeight * latentWidth);
        Tensor beta = new(new TensorShape([batch, _heads, frames, tokensPerFrame]), DType.F32, _backend.Device);
        Tensor decay = new(new TensorShape([batch, _heads, frames]), DType.F32, _backend.Device);
        bool succeeded = false;
        try
        {
            Tensor output = Run(x, frames, latentHeight, latentWidth, ropeCos, ropeSin, beta, decay);
            succeeded = true;
            return (output, beta, decay);
        }
        finally
        {
            if (!succeeded)
            {
                beta.Dispose();
                decay.Dispose();
            }
        }
    }

    private Tensor Run(Tensor x, int frames, int latentHeight, int latentWidth, Tensor ropeCos, Tensor ropeSin, Tensor? beta,
        Tensor? decay)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(ropeCos);
        ArgumentNullException.ThrowIfNull(ropeSin);
        if (frames <= 0 || latentHeight <= 0 || latentWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(frames), "Frames and latent grid must be positive.");
        }

        int tokensPerFrame = checked(latentHeight * latentWidth);
        int tokens = checked(frames * tokensPerFrame);
        if (x.Shape.Rank != 3 || x.Shape[1] != tokens || x.Shape[2] != _channels)
        {
            throw new ArgumentException($"x must be [B, {tokens}, {_channels}]; got {x.Shape}.", nameof(x));
        }

        if (x.DType != DType.F32 && x.DType != DType.BF16)
        {
            throw new ArgumentException($"x must be F32 or BF16; got {x.DType.Name}.", nameof(x));
        }

        RequireShape(ropeCos, nameof(ropeCos), [tokens, _headDim]);
        RequireShape(ropeSin, nameof(ropeSin), [tokens, _headDim]);
        if (ropeCos.DType != DType.F32 || ropeSin.DType != DType.F32)
        {
            throw new ArgumentException("The RoPE tables must be F32.", nameof(ropeCos));
        }

        int batch = (int)x.Shape[0];
        List<Tensor> scratch = [];
        Tensor output = new(new TensorShape([batch, tokens, _channels]), DType.F32, _backend.Device);
        bool succeeded = false;
        try
        {
            Tensor input = x.DType == DType.F32 ? x : Track(scratch, ConvertToF32(x));
            Tensor inputRows = Track(scratch, input.Reshape(new TensorShape([(long)batch * tokens, _channels])));
            Tensor outputRows = Track(scratch, output.Reshape(new TensorShape([(long)batch * tokens, _channels])));
            for (int b = 0; b < batch; b++)
            {
                Tensor xb = Track(scratch, inputRows.SliceRows((long)b * tokens, tokens));
                Tensor ob = Track(scratch, outputRows.SliceRows((long)b * tokens, tokens));
                ForwardBatch(xb, ob, ropeCos, ropeSin, frames, tokensPerFrame, scratch, beta, decay, b);
            }

            succeeded = true;
            return output;
        }
        finally
        {
            if (!succeeded)
            {
                output.Dispose();
            }

            foreach (Tensor tensor in scratch)
            {
                tensor.Dispose();
            }
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        ReleaseOwned();
    }

    private void ForwardBatch(Tensor xb, Tensor ob, Tensor cos, Tensor sin, int frames, int s, List<Tensor> scratch,
        Tensor? betaOut, Tensor? decayOut, int batchIndex)
    {
        int n = frames * s;
        Tensor q = NewScratch(scratch, [n, _channels]);
        _backend.LinearWeightRows(q, xb, _qkvWeight, null, 0, _channels);
        Tensor v = NewScratch(scratch, [n, _channels]);
        _backend.LinearWeightRows(v, xb, _qkvWeight, null, 2 * _channels, _channels);
        Tensor kConv = KConv(xb, frames, s, scratch);
        (Tensor beta, Tensor decayCol, Tensor decayRep) = Gates(xb, frames, s, scratch);
        if (betaOut is not null && decayOut is not null)
        {
            ExportGates(betaOut, decayOut, batchIndex, frames, s, beta, decayCol, scratch);
        }

        Tensor qUnrot = NormActivate(q, _qNorm, 1f, n, scratch);
        Tensor kUnrot = NormActivate(kConv, _kNorm, KeyScale(s), n, scratch);
        Tensor qRot = Rotate(qUnrot, cos, sin, n, scratch);
        Tensor kRot = Rotate(kUnrot, cos, sin, n, scratch);

        Tensor kHeads = HeadMajor(kUnrot, frames, s, scratch);
        Tensor kRotHeads = HeadMajor(kRot, frames, s, scratch);
        GdnHeadMajor g = new(
            HeadMajor(qUnrot, frames, s, scratch),
            HeadMajor(qRot, frames, s, scratch),
            kHeads,
            kRotHeads,
            HeadMajor(v, frames, s, scratch),
            beta,
            decayCol,
            decayRep,
            Transposed(kHeads, frames, s, scratch),
            Transposed(kRotHeads, frames, s, scratch));

        int rows = frames * _heads * s;
        Tensor numF = NewScratch(scratch, [rows, _headDim]);
        Tensor denF = NewScratch(scratch, [rows, 1]);
        Tensor numB = NewScratch(scratch, [rows, _headDim]);
        Tensor denB = NewScratch(scratch, [rows, 1]);
        Scan(g, frames, s, false, numF, denF, scratch);
        Scan(g, frames, s, true, numB, denB, scratch);
        Combine(numF, denF, numB, denB, ob, frames, s, scratch);
    }

    // Gates leave the scan frame-major ([T·H, ·]); the camera branch reads them head-major ([H, T, ·]) per batch item.
    private void ExportGates(Tensor betaOut, Tensor decayOut, int batchIndex, int frames, int s, Tensor beta, Tensor decay,
        List<Tensor> scratch)
    {
        long batch = betaOut.Shape[0];
        long headFrames = (long)_heads * frames;
        Tensor betaRows = Track(scratch, betaOut.Reshape(new TensorShape([batch * headFrames, s])));
        Tensor betaSlice = Track(scratch, betaRows.SliceRows((long)batchIndex * headFrames, headFrames));
        ReadOnlySpan<long> betaDims = [_heads, frames, s];
        ReadOnlySpan<long> betaStrides = [s, (long)_heads * s, 1];
        _backend.CopyStrided(betaSlice, beta, StridedCopySpec.Gather(new StridedView(betaDims, betaStrides, 0)));

        Tensor decayRows = Track(scratch, decayOut.Reshape(new TensorShape([batch * _heads, frames])));
        Tensor decaySlice = Track(scratch, decayRows.SliceRows((long)batchIndex * _heads, _heads));
        ReadOnlySpan<long> decayDims = [_heads, frames];
        ReadOnlySpan<long> decayStrides = [1, _heads];
        _backend.CopyStrided(decaySlice, decay, StridedCopySpec.Gather(new StridedView(decayDims, decayStrides, 0)));
    }

    // Bidirectional depthwise temporal conv on K. The seven taps are views of a zero-padded K, so the
    // conv costs one projection, one fill, seven broadcast scales and six adds per batch element.
    private Tensor KConv(Tensor xb, int frames, int s, List<Tensor> scratch)
    {
        int n = frames * s;
        Tensor padded = NewScratch(scratch, [(frames + 2 * ConvHalfWidth) * s, _channels]);
        _backend.Fill(padded, 0f);
        Tensor kMiddle = Track(scratch, padded.SliceRows((long)ConvHalfWidth * s, n));
        _backend.LinearWeightRows(kMiddle, xb, _qkvWeight, null, _channels, _channels);

        TensorShape frameShape = new([1, n, _channels]);
        Tensor accA = NewScratch(scratch, frameShape);
        Tensor accB = NewScratch(scratch, frameShape);
        Tensor term = NewScratch(scratch, frameShape);
        for (int i = 0; i < SymmetricTap.Length; i++)
        {
            Tensor window = Track(scratch, padded.SliceRows((long)i * s, n).Reshape(frameShape));
            Tensor tap = Track(scratch, _convTaps.SliceRows(SymmetricTap[i], 1).Reshape(new TensorShape([_channels])));
            if (i == 0)
            {
                _backend.AffineBroadcastLastDim(accA, window, tap, null);
                continue;
            }

            _backend.AffineBroadcastLastDim(term, window, tap, null);
            _backend.Add(accB, accA, term);
            (accA, accB) = (accB, accA);
        }

        return Track(scratch, accA.Reshape(new TensorShape([n, _channels])));
    }

    // Frame gates are per frame and per head. The decay is exp(-A * softplus(Linear(mean_S x) + dt)); exp(z) for z <= 0
    // is ELU(z) + 1, so the decay needs no exp kernel. The gate bias already carries dt_bias.
    private (Tensor Beta, Tensor DecayCol, Tensor DecayRep) Gates(Tensor xb, int frames, int s, List<Tensor> scratch)
    {
        int th = frames * _heads;
        Tensor betaLinear = NewScratch(scratch, [frames * s, _heads]);
        _backend.Linear(betaLinear, xb, _betaWeight, _betaBias);
        Tensor beta = NewScratch(scratch, [frames * s, _heads]);
        _backend.Sigmoid(beta, betaLinear);
        Tensor betaTh = NewScratch(scratch, [th, s]);
        _backend.Transpose2D(betaTh, Track(scratch, beta.Reshape(new TensorShape([frames, s, _heads]))), s, _heads);

        Tensor meanWeights = Filled(scratch, [frames, 1, s], 1f / s);
        Tensor frameMean = NewScratch(scratch, [frames, 1, _channels]);
        _backend.BatchedMatMul(frameMean, meanWeights, Track(scratch, xb.Reshape(new TensorShape([frames, s, _channels]))));
        Tensor gateLinear = NewScratch(scratch, [frames, _heads]);
        _backend.Linear(gateLinear, Track(scratch, frameMean.Reshape(new TensorShape([frames, _channels]))), _gateWeight, _gateBias);
        Tensor softplus = NewScratch(scratch, [frames, _heads]);
        _backend.Softplus(softplus, gateLinear);
        Tensor exponent = NewScratch(scratch, [frames, _heads]);
        _backend.AffineBroadcastLastDim(exponent, softplus, _negRate, null);
        Tensor elu = NewScratch(scratch, [frames, _heads]);
        _backend.Elu(elu, exponent, 1f);

        Tensor decayCol = NewScratch(scratch, [th, 1]);
        _backend.AffineMix(decayCol, Track(scratch, elu.Reshape(new TensorShape([th, 1]))), Filled(scratch, [th, 1], 1f), 1f, 1f);
        Tensor decayRep = NewScratch(scratch, [th, _headDim]);
        Tensor unitRow = Filled(scratch, [1, _headDim], 1f);
        _backend.BatchedMatMul(
            Track(scratch, decayRep.Reshape(new TensorShape([th, 1, _headDim]))),
            Track(scratch, decayCol.Reshape(new TensorShape([th, 1, 1]))),
            unitRow);
        return (betaTh, decayCol, decayRep);
    }

    private Tensor NormActivate(Tensor input, Tensor normWeight, float scale, int n, List<Tensor> scratch)
    {
        Tensor normed = NewScratch(scratch, [n, _channels]);
        _backend.RmsNorm(normed, input, normWeight, RmsEpsilon);
        Tensor activated = NewScratch(scratch, [n, _channels]);
        _backend.LeakyRelu(activated, normed, 0f);
        Tensor scaled = NewScratch(scratch, [n, _channels]);
        _backend.Scale(scaled, activated, scale);
        return scaled;
    }

    private Tensor Rotate(Tensor source, Tensor cos, Tensor sin, int n, List<Tensor> scratch)
    {
        Tensor rotated = NewScratch(scratch, [n, _channels]);
        _backend.CopyTo(rotated, source);
        _backend.Rotary(RotaryCall.Rows(rotated, cos, sin, n, _heads, _headDim, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair));
        return rotated;
    }

    private Tensor HeadMajor(Tensor tokenMajor, int frames, int s, List<Tensor> scratch)
    {
        Tensor headMajor = NewScratch(scratch, [frames * _heads, s, _headDim]);
        _backend.Permute0213(headMajor, tokenMajor, s, _heads, _headDim);
        return headMajor;
    }

    private Tensor Transposed(Tensor headMajor, int frames, int s, List<Tensor> scratch)
    {
        Tensor transposed = NewScratch(scratch, [frames * _heads, _headDim, s]);
        _backend.Transpose2D(transposed, headMajor, s, _headDim);
        return transposed;
    }

    // One pass over the frames. Forward step t reads and writes frame t. Reverse step j reads frame T-j (the shifted
    // source) and writes frame T-1-j, so frame T-1 receives nothing and the state starts empty.
    private void Scan(in GdnHeadMajor g, int frames, int s, bool reverse, Tensor num, Tensor den, List<Tensor> scratch)
    {
        int h = _heads;
        int d = _headDim;
        TensorShape matDd = new([h, d, d]);
        TensorShape vecD1 = new([h, d, 1]);
        TensorShape matSd = new([h, s, d]);
        TensorShape matDs = new([h, d, s]);
        TensorShape vecS1 = new([h, s, 1]);

        Tensor aState = NewScratch(scratch, matDd);
        Tensor zState = NewScratch(scratch, vecD1);
        _backend.Fill(aState, 0f);
        _backend.Fill(zState, 0f);
        Tensor aBar = NewScratch(scratch, matDd);
        Tensor zBar = NewScratch(scratch, vecD1);
        Tensor pred = NewScratch(scratch, matSd);
        Tensor residual = NewScratch(scratch, matSd);
        Tensor aUpdate = NewScratch(scratch, matDd);
        Tensor zUpdate = NewScratch(scratch, vecD1);
        Tensor betaKeyA = NewScratch(scratch, matDs);
        Tensor betaKeyZ = NewScratch(scratch, matDs);
        Tensor zPred = NewScratch(scratch, vecS1);
        Tensor zResidual = NewScratch(scratch, vecS1);
        Tensor unitTokens = Filled(scratch, vecS1, 1f);

        if (reverse)
        {
            _backend.Fill(Track(scratch, FrameOf(num, frames - 1, h * s).Reshape(matSd)), 0f);
            _backend.Fill(Track(scratch, FrameOf(den, frames - 1, h * s).Reshape(vecS1)), 0f);
        }

        for (int step = reverse ? 1 : 0; step < frames; step++)
        {
            int src = reverse ? frames - step : step;
            int dst = reverse ? frames - 1 - step : step;
            Tensor decayCol = Track(scratch, g.DecayCol.SliceRows((long)src * h, h));
            Tensor decayRep = Track(scratch, g.DecayRep.SliceRows((long)src * h, h));
            Tensor beta = Track(scratch, g.Beta.SliceRows((long)src * h, h));
            Tensor kRot = Track(scratch, g.KRot.SliceRows((long)src * h, h));
            Tensor k = Track(scratch, g.K.SliceRows((long)src * h, h));
            Tensor v = Track(scratch, g.V.SliceRows((long)src * h, h));
            Tensor kRotT = Track(scratch, g.KRotT.SliceRows((long)src * h, h));
            Tensor kT = Track(scratch, g.KT.SliceRows((long)src * h, h));
            Tensor qRot = Track(scratch, g.QRot.SliceRows((long)dst * h, h));
            Tensor q = Track(scratch, g.Q.SliceRows((long)dst * h, h));
            Tensor numOut = Track(scratch, FrameOf(num, dst, h * s).Reshape(matSd));
            Tensor denOut = Track(scratch, FrameOf(den, dst, h * s).Reshape(vecS1));

            // Numerator state (key x value, stored transposed) and denominator state, decayed then updated.
            _backend.AffineBroadcastLastDim(aBar, aState, decayRep, null);
            _backend.AffineBroadcastLastDim(zBar, zState, decayCol, null);
            _backend.BatchedMatMul(pred, kRot, aBar);
            _backend.AffineMix(residual, v, pred, 1f, -1f);
            _backend.AffineBroadcastLastDim(betaKeyA, kRotT, beta, null);
            _backend.BatchedMatMul(aUpdate, betaKeyA, residual);
            _backend.Add(aState, aBar, aUpdate);
            _backend.BatchedMatMul(zPred, k, zBar);
            _backend.AffineMix(zResidual, unitTokens, zPred, 1f, -1f);
            _backend.AffineBroadcastLastDim(betaKeyZ, kT, beta, null);
            _backend.BatchedMatMul(zUpdate, betaKeyZ, zResidual);
            _backend.Add(zState, zBar, zUpdate);

            _backend.BatchedMatMul(numOut, qRot, aState);
            _backend.BatchedMatMul(denOut, q, zState);
        }
    }

    // out = (num_f + num_b) / (den_f + den_b + eps), with the division as a reciprocal times a row broadcast.
    private void Combine(Tensor numF, Tensor denF, Tensor numB, Tensor denB, Tensor ob, int frames, int s, List<Tensor> scratch)
    {
        int rows = frames * _heads * s;
        Tensor num = NewScratch(scratch, [rows, _headDim]);
        _backend.Add(num, numF, numB);
        Tensor denSum = NewScratch(scratch, [rows, 1]);
        _backend.Add(denSum, denF, denB);
        Tensor denEps = NewScratch(scratch, [rows, 1]);
        _backend.AffineBroadcastLastDim(denEps, denSum, Filled(scratch, [1], 1f), Filled(scratch, [1], DenominatorEpsilon));
        Tensor reciprocal = NewScratch(scratch, [rows, 1]);
        SanaWmGdnElementwise.Reciprocal(_backend, reciprocal, denEps);
        Tensor reciprocalRows = NewScratch(scratch, [rows, _headDim]);
        _backend.BatchedMatMul(
            Track(scratch, reciprocalRows.Reshape(new TensorShape([rows, 1, _headDim]))),
            Track(scratch, reciprocal.Reshape(new TensorShape([rows, 1, 1]))),
            Filled(scratch, [1, _headDim], 1f));
        Tensor normalized = NewScratch(scratch, [rows, _headDim]);
        _backend.Mul(normalized, num, reciprocalRows);
        _backend.Permute0213(
            Track(scratch, ob.Reshape(new TensorShape([frames, s, _heads, _headDim]))),
            Track(scratch, normalized.Reshape(new TensorShape([frames, _heads, s, _headDim]))),
            _heads,
            s,
            _headDim);
    }

    private Tensor Prepare(Tensor source)
    {
        if (source.DType == DType.F32)
        {
            return source;
        }

        Tensor converted = ConvertToF32(source);
        _owned.Add(converted);
        return converted;
    }

    // The CPU default of IBackend.CastToF32 reads its input as F16, so BF16 on the CPU takes the dtype-aware host cast.
    private Tensor ConvertToF32(Tensor source)
    {
        if (_backend.Device.IsCpu)
        {
            return source.CastTo(DType.F32);
        }

        Tensor converted = new(source.Shape, DType.F32, _backend.Device);
        _backend.CastToF32(converted, source);
        return converted;
    }

    private Tensor Filled(List<Tensor> scratch, ReadOnlySpan<long> dims, float value) => Filled(scratch, new TensorShape(dims), value);

    private Tensor Filled(List<Tensor> scratch, TensorShape shape, float value)
    {
        Tensor filled = NewScratch(scratch, shape);
        _backend.Fill(filled, value);
        return filled;
    }

    private Tensor NewTensor(ReadOnlySpan<long> dims) => NewTensor(new TensorShape(dims));

    private Tensor NewTensor(TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32, _backend.Device);
        _owned.Add(tensor);
        return tensor;
    }

    private Tensor NewScratch(List<Tensor> scratch, ReadOnlySpan<long> dims) => NewScratch(scratch, new TensorShape(dims));

    private Tensor NewScratch(List<Tensor> scratch, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32, _backend.Device);
        scratch.Add(tensor);
        return tensor;
    }

    private static Tensor Track(List<Tensor> scratch, Tensor view)
    {
        scratch.Add(view);
        return view;
    }

    private static Tensor FrameOf(Tensor rows, int frame, int rowsPerFrame) => rows.SliceRows((long)frame * rowsPerFrame, rowsPerFrame);

    private float KeyScale(int tokensPerFrame) => (float)(Math.Pow(_headDim, -0.5) * Math.Pow(tokensPerFrame, -0.5));

    private void ReleaseOwned()
    {
        foreach (Tensor tensor in _owned)
        {
            tensor.Dispose();
        }

        _owned.Clear();
    }

    private static void RequireShape(Tensor tensor, string name, ReadOnlySpan<long> expected)
    {
        if (tensor.Shape.Rank != expected.Length)
        {
            throw new ArgumentException($"{name} must have rank {expected.Length}; got {tensor.Shape}.", name);
        }

        for (int i = 0; i < expected.Length; i++)
        {
            if (tensor.Shape[i] != expected[i])
            {
                throw new ArgumentException($"{name} must be {FormatDims(expected)}; got {tensor.Shape}.", name);
            }
        }
    }

    private static string FormatDims(ReadOnlySpan<long> dims) => string.Join("x", dims.ToArray());

    private readonly record struct GdnHeadMajor(
        Tensor Q,
        Tensor QRot,
        Tensor K,
        Tensor KRot,
        Tensor V,
        Tensor Beta,
        Tensor DecayCol,
        Tensor DecayRep,
        Tensor KT,
        Tensor KRotT);
}
