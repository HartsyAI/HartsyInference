using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Main branch of one SANA-WM bidirectional Gated-DeltaNet attention layer. Returns the branch before the camera
/// sum, the output gate and the projection. The forward scan includes the current frame; the reverse scan excludes it.</summary>
/// <remarks>Runs one batch item at a time. q, k and v stay token-major in F32 and each scan step gathers its frame head-major,
/// so the item's live set is five <c>[N, C]</c> operands and the numerator. The states and accumulators are F32.</remarks>
public sealed class SanaWmGdnBidi : IDisposable
{
    private const float RmsEpsilon = 1e-5f;
    private const float DenominatorEpsilon = 1e-8f;

    private readonly IBackend _backend;
    private readonly List<Tensor> _owned = [];
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _channels;
    private readonly Tensor _qkvWeight;
    private readonly Tensor _convKernel;
    private readonly Tensor _betaWeight;
    private readonly Tensor _betaBias;
    private readonly Tensor _gateWeight;
    private readonly Tensor _gateBias;
    private readonly Tensor _negRate;
    private readonly Tensor _qNorm;
    private readonly Tensor _kNorm;
    private readonly Tensor _denominatorScale;
    private readonly Tensor _denominatorShift;
    private bool _disposed;

    /// <param name="backend">Backend the layer runs on.</param>
    /// <param name="weights">Layer weights. The qkv matrix is borrowed in its stored dtype (F32 or BF16); the small parameters
    /// become host F32 tensors the layer owns when they are not F32 already.</param>
    /// <param name="heads">Attention heads (20 for SANA-WM).</param>
    /// <param name="headDim">Head dimension (112 for SANA-WM); <c>heads * headDim</c> is the channel count.</param>
    public SanaWmGdnBidi(IBackend backend, SanaWmGdnWeights weights, int heads, int headDim)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(weights);
        if (heads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(heads), "Heads and head dim must be positive.");

        int channels = checked(heads * headDim);
        RequireShape(weights.Qkv, nameof(weights.Qkv), [3L * channels, channels]);
        RequireShape(weights.ConvK, nameof(weights.ConvK), [channels, 1, SanaWmTemporalConv.SourceTaps]);
        RequireShape(weights.BetaWeight, nameof(weights.BetaWeight), [heads, channels]);
        RequireShape(weights.BetaBias, nameof(weights.BetaBias), [heads]);
        RequireShape(weights.GateWeight, nameof(weights.GateWeight), [heads, channels]);
        RequireShape(weights.GateBias, nameof(weights.GateBias), [heads]);
        RequireShape(weights.ALog, nameof(weights.ALog), [heads]);
        RequireShape(weights.DtBias, nameof(weights.DtBias), [heads]);
        RequireShape(weights.QNorm, nameof(weights.QNorm), [channels]);
        RequireShape(weights.KNorm, nameof(weights.KNorm), [channels]);
        if (weights.Qkv.DType != DType.F32 && weights.Qkv.DType != DType.BF16)
            throw new ArgumentException($"The qkv weight must be F32 or BF16; got {weights.Qkv.DType.Name}.", nameof(weights));

        _backend = backend;
        _heads = heads;
        _headDim = headDim;
        _channels = channels;
        bool built = false;
        try
        {
            _qkvWeight = weights.Qkv;
            _betaWeight = SanaWmTensors.HostF32(weights.BetaWeight, _owned);
            _betaBias = SanaWmTensors.HostF32(weights.BetaBias, _owned);
            _gateWeight = SanaWmTensors.HostF32(weights.GateWeight, _owned);
            _qNorm = SanaWmTensors.HostF32(weights.QNorm, _owned);
            _kNorm = SanaWmTensors.HostF32(weights.KNorm, _owned);
            _convKernel = Own(SanaWmTemporalConv.SymmetricKernel(weights.ConvK, channels, nameof(weights.ConvK)));

            // Parameters are folded on the host: a long-lived device activation would not survive a backend's FreeActivations.
            ReadOnlySpan<float> gateBias = SanaWmTensors.HostF32(weights.GateBias, _owned).AsReadOnlySpan<float>();
            ReadOnlySpan<float> dtBias = SanaWmTensors.HostF32(weights.DtBias, _owned).AsReadOnlySpan<float>();
            ReadOnlySpan<float> aLog = SanaWmTensors.HostF32(weights.ALog, _owned).AsReadOnlySpan<float>();
            _gateBias = Own(new Tensor(new TensorShape(heads), DType.F32));
            _negRate = Own(new Tensor(new TensorShape(heads), DType.F32));
            Span<float> combinedBias = _gateBias.AsSpan<float>();
            Span<float> negRate = _negRate.AsSpan<float>();
            for (int h = 0; h < heads; h++)
            {
                combinedBias[h] = gateBias[h] + dtBias[h];
                negRate[h] = -MathF.Exp(aLog[h]);
            }

            _denominatorScale = Own(SanaWmTensors.Filled(new TensorShape(heads), 1f));
            _denominatorShift = Own(SanaWmTensors.Filled(new TensorShape(heads), DenominatorEpsilon));
            built = true;
        }
        finally
        {
            if (!built)
                SanaWmTensors.DisposeAll(_owned);
        }
    }

    /// <summary>Main-branch output <c>[B, N, C]</c> in F32, before the camera sum, output gate and projection. Caller disposes it.</summary>
    /// <param name="x">Layer input <c>[B, N, C]</c>, F32 or BF16, with <c>N = frames * latentHeight * latentWidth</c> in frame-major order.</param>
    /// <param name="ropeCos">Cos table <c>[N, headDim]</c> in interleaved-pair layout.</param>
    /// <param name="ropeSin">Sin table in the same layout; with <paramref name="ropeCos"/> it is the complex table <c>cos + i sin</c>.</param>
    public Tensor Forward(Tensor x, int frames, int latentHeight, int latentWidth, Tensor ropeCos, Tensor ropeSin)
    {
        (Tensor output, Tensor beta, Tensor decay) = ForwardWithGates(x, frames, latentHeight, latentWidth, ropeCos, ropeSin);
        beta.Dispose();
        decay.Dispose();
        return output;
    }

    /// <summary>Main-branch output plus the frame gates the camera branch shares: beta <c>[B, H, T, S]</c> and decay
    /// <c>[B, H, T]</c>, both F32 and head-major. The caller disposes all three tensors.</summary>
    public (Tensor Output, Tensor Beta, Tensor Decay) ForwardWithGates(Tensor x, int frames, int latentHeight, int latentWidth,
        Tensor ropeCos, Tensor ropeSin)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(x);
        if (frames <= 0 || latentHeight <= 0 || latentWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(frames), "Frames and latent grid must be positive.");
        int spatial = checked(latentHeight * latentWidth);
        int tokens = checked(frames * spatial);
        if (x.Shape.Rank != 3 || x.Shape[1] != tokens || x.Shape[2] != _channels)
            throw new ArgumentException($"x must be [B, {tokens}, {_channels}]; got {x.Shape}.", nameof(x));
        if (x.DType != DType.F32 && x.DType != DType.BF16)
            throw new ArgumentException($"x must be F32 or BF16; got {x.DType.Name}.", nameof(x));
        ValidateTables(ropeCos, ropeSin, tokens);

        int batch = (int)x.Shape[0];
        long itemElements = (long)tokens * _channels;
        using SanaWmScratch scratch = new();
        Tensor input = x.DType == DType.F32 ? x : scratch.Track(SanaWmTensors.ToF32(_backend, x));
        Tensor output = scratch.New(new TensorShape(batch, tokens, _channels));
        Tensor beta = scratch.New(batch, _heads, frames, spatial);
        Tensor decay = scratch.New(batch, _heads, frames);
        for (int b = 0; b < batch; b++)
        {
            using SanaWmScratch item = new();
            Tensor a = item.Track(SanaWmTensors.Slice(_backend, input, b * itemElements, new TensorShape(frames, spatial, _channels)));
            (Tensor itemOutput, Tensor itemBeta, Tensor itemDecay) = ForwardItem(a, frames, spatial, ropeCos, ropeSin);
            item.Track(itemOutput);
            item.Track(itemBeta);
            item.Track(itemDecay);
            SanaWmTensors.Write(_backend, output, itemOutput, b * itemElements);
            SanaWmTensors.Write(_backend, beta, itemBeta, (long)b * itemBeta.ElementCount);
            SanaWmTensors.Write(_backend, decay, itemDecay, (long)b * itemDecay.ElementCount);
        }

        return (scratch.Detach(output), scratch.Detach(beta), scratch.Detach(decay));
    }

    /// <summary>One batch item: <paramref name="a"/> is F32 <c>[frames, spatial, C]</c>. Returns the branch output <c>[N, C]</c>, beta
    /// <c>[H, frames, spatial]</c> and decay <c>[H, frames]</c>; the caller disposes all three.</summary>
    /// <param name="projectionInput">The same tokens in the dtype the q, k and v GEMMs read (BF16 on a device), shared with the
    /// block's other projections so the input is converted once; null projects from <paramref name="a"/>.</param>
    internal (Tensor Output, Tensor Beta, Tensor Decay) ForwardItem(Tensor a, int frames, int spatial, Tensor ropeCos, Tensor ropeSin,
        string profileTag = "gdn", Tensor? projectionInput = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int tokens = checked(frames * spatial);
        SanaWmTensors.Require(a, nameof(a), DType.F32, [frames, spatial, _channels]);
        Tensor projected = projectionInput ?? a;
        if (projected.ElementCount != (long)tokens * _channels)
            throw new ArgumentException($"The projection input must hold {tokens}x{_channels} elements; got {projected.Shape}.", nameof(projectionInput));
        ValidateTables(ropeCos, ropeSin, tokens);

        using SanaWmScratch scratch = new();
        Tensor betaRows, decay, beta, decayHeads;
        using (StageProfiler.Begin(_backend, profileTag, "attn.compute_frame_gates"))
        {
            (betaRows, decay) = Gates(scratch, a, frames, spatial);
            beta = scratch.New(_heads, frames, spatial);
            ReadOnlySpan<long> betaDims = [_heads, frames, spatial];
            ReadOnlySpan<long> betaStrides = [spatial, (long)_heads * spatial, 1];
            _backend.CopyStrided(beta, betaRows, StridedCopySpec.Gather(new StridedView(betaDims, betaStrides, 0)));
            decayHeads = scratch.New(_heads, frames);
            ReadOnlySpan<long> decayDims = [_heads, frames];
            ReadOnlySpan<long> decayStrides = [1, _heads];
            _backend.CopyStrided(decayHeads, decay, StridedCopySpec.Gather(new StridedView(decayDims, decayStrides, 0)));
        }

        Tensor kUnrot, kRot, qUnrot, qRot, v;
        using (StageProfiler.Begin(_backend, profileTag, "attn.main.qkv"))
        {
            kUnrot = scratch.Track(KeyPath(projected, frames, spatial));
            kRot = scratch.Track(Rotated(kUnrot, ropeCos, ropeSin, tokens));
            qUnrot = scratch.Track(QueryPath(projected, tokens));
            qRot = scratch.Track(Rotated(qUnrot, ropeCos, ropeSin, tokens));
            v = scratch.New(tokens, _channels);
            _backend.LinearWeightRows(v, projected, _qkvWeight, null, 2 * _channels, _channels);
        }

        scratch.Release(betaRows);
        scratch.Release(decay);
        // Numerator over the rotated q and k; denominator over the unrotated ones with a unit value.
        Tensor num = scratch.New(tokens, _channels);
        Tensor den = scratch.New(tokens, _heads);
        using (StageProfiler.Begin(_backend, profileTag, "attn.main.scan"))
        {
            _backend.FrameDeltaScan(num, qRot, kRot, v, beta, decayHeads, frames, spatial, _heads, _headDim, _headDim);
            _backend.FrameDeltaScan(den, qUnrot, kUnrot, null, beta, decayHeads, frames, spatial, _heads, _headDim, 1);
        }

        foreach (Tensor consumed in new[] { qUnrot, qRot, kUnrot, kRot, v })
            scratch.Release(consumed);

        Tensor output;
        using (StageProfiler.Begin(_backend, profileTag, "attn.main.combine"))
            output = Combine(scratch, num, den, tokens);
        return (scratch.Detach(output), scratch.Detach(beta), scratch.Detach(decayHeads));
    }

    /// <summary>The weights the layer reads on every call, for preloading on a device backend.</summary>
    public IEnumerable<Tensor> EnumerateWeights() =>
        [_qkvWeight, _convKernel, _betaWeight, _betaBias, _gateWeight, _gateBias, _negRate, _qNorm, _kNorm, _denominatorScale,
            _denominatorShift];

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        SanaWmTensors.DisposeAll(_owned);
    }

    private Tensor Own(Tensor tensor)
    {
        _owned.Add(tensor);
        return tensor;
    }

    // Beta per token and head, frame-major [T·H, S]; decay exp(-exp(A_log)·softplus(Linear(mean_S x) + dt)) per frame, [T·H, 1].
    private (Tensor BetaRows, Tensor Decay) Gates(SanaWmScratch scratch, Tensor a, int frames, int spatial)
    {
        int tokens = frames * spatial;
        Tensor betaLinear = scratch.New(tokens, _heads);
        _backend.Linear(betaLinear, a, _betaWeight, _betaBias);
        Tensor betaTokens = scratch.New(tokens, _heads);
        _backend.Sigmoid(betaTokens, betaLinear);
        scratch.Release(betaLinear);
        Tensor betaRows = scratch.New((long)frames * _heads, spatial);
        _backend.Transpose2D(betaRows, betaTokens, spatial, _heads);
        scratch.Release(betaTokens);

        Tensor meanWeights = scratch.Track(SanaWmTensors.Filled(new TensorShape(frames, 1, spatial), 1f / spatial));
        Tensor frameMean = scratch.New(frames, 1, _channels);
        _backend.BatchedMatMul(frameMean, meanWeights, a);
        scratch.Release(meanWeights);
        Tensor gateLinear = scratch.New(frames, _heads);
        _backend.Linear(gateLinear, frameMean, _gateWeight, _gateBias);
        scratch.Release(frameMean);
        Tensor softplus = scratch.New(frames, _heads);
        _backend.Softplus(softplus, gateLinear);
        scratch.Release(gateLinear);
        Tensor exponent = scratch.New(frames, _heads);
        _backend.AffineBroadcastLastDim(exponent, softplus, _negRate, null);
        scratch.Release(softplus);
        Tensor decay = scratch.New((long)frames * _heads, 1);
        _backend.Exp(decay, exponent);
        scratch.Release(exponent);
        return (betaRows, decay);
    }

    // k = ReLU(RMSNorm(conv(Linear(x)))) · D^-1/2 · S^-1/2, token-major [N, C].
    private Tensor KeyPath(Tensor a, int frames, int spatial)
    {
        int tokens = frames * spatial;
        using SanaWmScratch scratch = new();
        Tensor raw = scratch.New(tokens, _channels);
        _backend.LinearWeightRows(raw, a, _qkvWeight, null, _channels, _channels);
        Tensor convolved = scratch.Track(SanaWmTemporalConv.Apply(_backend, raw, _convKernel, frames, spatial, _channels));
        scratch.Release(raw);
        return scratch.Detach(NormActivate(scratch, convolved, _kNorm, tokens, KeyScale(spatial)));
    }

    // q = ReLU(RMSNorm(Linear(x))), token-major [N, C].
    private Tensor QueryPath(Tensor a, int tokens)
    {
        using SanaWmScratch scratch = new();
        Tensor raw = scratch.New(tokens, _channels);
        _backend.LinearWeightRows(raw, a, _qkvWeight, null, 0, _channels);
        return scratch.Detach(NormActivate(scratch, raw, _qNorm, tokens, 1f));
    }

    // Releases input; returns a tracked ReLU(RMSNorm(input))·scale, one fused op.
    private Tensor NormActivate(SanaWmScratch scratch, Tensor input, Tensor normWeight, int tokens, float scale)
    {
        Tensor activated = scratch.New(tokens, _channels);
        _backend.RmsNormRelu(activated, input, normWeight, RmsEpsilon, scale);
        scratch.Release(input);
        return activated;
    }

    private Tensor Rotated(Tensor source, Tensor cos, Tensor sin, int tokens)
    {
        Tensor rotated = SanaWmTensors.Copy(_backend, source, source.Shape);
        try
        {
            _backend.Rotary(RotaryCall.Rows(rotated, cos, sin, tokens, _heads, _headDim, RotaryPairing.Interleaved,
                RopeTableLayout.InterleavedPair));
            return rotated;
        }
        catch
        {
            rotated.Dispose();
            throw;
        }
    }

    // out = (num_f + num_b) / (den_f + den_b + eps): the reciprocal of each token's per-head denominator scales its head's lanes.
    private Tensor Combine(SanaWmScratch scratch, Tensor num, Tensor den, int tokens)
    {
        Tensor shifted = scratch.New(tokens, _heads);
        _backend.AffineBroadcastLastDim(shifted, den, _denominatorScale, _denominatorShift);
        scratch.Release(den);
        Tensor reciprocal = scratch.New(tokens, _heads);
        _backend.Reciprocal(reciprocal, shifted);
        scratch.Release(shifted);
        Tensor output = scratch.New(tokens, _channels);
        _backend.GroupScale(output, num, reciprocal, GroupScaleSpec.PerGroup(_headDim, _heads, GateFunction.Identity));
        scratch.Release(num);
        scratch.Release(reciprocal);
        return output;
    }

    private void ValidateTables(Tensor ropeCos, Tensor ropeSin, int tokens)
    {
        ArgumentNullException.ThrowIfNull(ropeCos);
        ArgumentNullException.ThrowIfNull(ropeSin);
        RequireShape(ropeCos, nameof(ropeCos), [tokens, _headDim]);
        RequireShape(ropeSin, nameof(ropeSin), [tokens, _headDim]);
        if (ropeCos.DType != DType.F32 || ropeSin.DType != DType.F32)
            throw new ArgumentException("The RoPE tables must be F32.", nameof(ropeCos));
    }

    private float KeyScale(int tokensPerFrame) => (float)(Math.Pow(_headDim, -0.5) * Math.Pow(tokensPerFrame, -0.5));

    private static void RequireShape(Tensor tensor, string name, ReadOnlySpan<long> expected)
    {
        if (tensor.Shape.Rank != expected.Length)
            throw new ArgumentException($"{name} must have rank {expected.Length}; got {tensor.Shape}.", name);
        for (int i = 0; i < expected.Length; i++)
        {
            if (tensor.Shape[i] != expected[i])
                throw new ArgumentException($"{name} must be {string.Join("x", expected.ToArray())}; got {tensor.Shape}.", name);
        }
    }
}
