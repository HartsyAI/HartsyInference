using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>SANA-WM camera branch of one attention layer (upstream UCPE single-path camera branch). It takes the same
/// post-adaLN input as the main branch, projects it with its own Q/K/V, applies the per-token ray transform and RoPE, and
/// runs the numerator-only bidirectional scan that shares the main branch's beta and decay. The result is
/// <c>out_proj_cam(·)</c>; the block sums it with the main branch output.</summary>
public sealed class SanaWmCamBranch : IDisposable
{
    private const float NormEpsilon = 1e-5f;
    private const float ProbeEpsilon = 1e-30f;

    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _channels;

    private Tensor? _queryWeight;
    private Tensor? _queryBias;
    private Tensor? _keyWeight;
    private Tensor? _keyBias;
    private Tensor? _valueWeight;
    private Tensor? _valueBias;
    private Tensor? _convKernel;
    private Tensor? _queryNorm;
    private Tensor? _keyNorm;
    private Tensor? _outputWeight;
    private Tensor? _outputBias;
    private int _disposed;

    /// <param name="heads">Attention heads; the camera branch shares the main branch's head split.</param>
    /// <param name="headDim">Per-head dim. Must be a multiple of 8 and at least 24 so the UCPE and RoPE halves split.</param>
    public SanaWmCamBranch(int heads, int headDim)
    {
        if (heads <= 0)
            throw new ArgumentOutOfRangeException(nameof(heads), heads, "Head count must be positive.");
        if (headDim < 24 || headDim % 8 != 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "Head dim must be a multiple of 8 and at least 24.");
        _heads = heads;
        _headDim = headDim;
        _channels = checked(heads * headDim);
    }

    /// <summary>Loads <c>q_proj_cam</c>, <c>k_proj_cam</c>, <c>v_proj_cam</c>, <c>conv_k_cam</c>, <c>q_norm_cam</c>,
    /// <c>k_norm_cam</c> and <c>out_proj_cam</c> under <paramref name="prefix"/>. The depthwise k=4 conv becomes a symmetric
    /// 7-tap kernel, which is the bidirectional trick folded into one filter.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(weights);
        Tensor conv = TensorCasts.EnsureF32(weights[$"{prefix}conv_k_cam.weight"]);
        if (conv.Shape.Rank != 3 || conv.Shape[0] != _channels || conv.Shape[1] != 1 || conv.Shape[2] != 4)
            throw new ArgumentException($"conv_k_cam.weight must be [{_channels}, 1, 4]; got {conv.Shape}.");

        _queryWeight = Matrix(weights, prefix, "q_proj_cam.weight");
        _queryBias = Vector(weights, prefix, "q_proj_cam.bias");
        _keyWeight = Matrix(weights, prefix, "k_proj_cam.weight");
        _keyBias = Vector(weights, prefix, "k_proj_cam.bias");
        _valueWeight = Matrix(weights, prefix, "v_proj_cam.weight");
        _valueBias = Vector(weights, prefix, "v_proj_cam.bias");
        _queryNorm = Vector(weights, prefix, "q_norm_cam.weight");
        _keyNorm = Vector(weights, prefix, "k_norm_cam.weight");
        _outputWeight = Matrix(weights, prefix, "out_proj_cam.weight");
        _outputBias = Vector(weights, prefix, "out_proj_cam.bias");
        _convKernel?.Dispose();
        _convKernel = SymmetricKernel(conv);
    }

    /// <summary>Weights loaded by <see cref="LoadWeights"/>; the derived conv kernel is owned and released by <see cref="Dispose"/>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_queryWeight, _queryBias, _keyWeight, _keyBias, _valueWeight, _valueBias, _queryNorm, _keyNorm,
            _outputWeight, _outputBias];
        foreach (Tensor? tensor in all)
        {
            if (tensor is not null)
                yield return tensor;
        }
    }

    /// <summary>Runs the camera branch over one latent sequence and returns <c>out_proj_cam</c> of the scan, shaped <c>[B, N, C]</c>.
    /// The caller owns the result.</summary>
    /// <param name="x">F32 <c>[B, N, C]</c> post-adaLN tokens, frame-major with <c>N = frames·latentHeight·latentWidth</c>.</param>
    /// <param name="cameraConditions">Per frame, 16 row-major camera-to-world floats then fx, fy, cx, cy in latent pixels.</param>
    /// <param name="ropeCos">Main 3D RoPE cosine table <c>[N, headDim/2]</c>; the camera slice is taken from it.</param>
    /// <param name="ropeSin">Main 3D RoPE sine table <c>[N, headDim/2]</c>.</param>
    /// <param name="beta">Shared write gate <c>[B, H, T, S]</c>, before camera discounting.</param>
    /// <param name="decay">Shared per-frame decay <c>[B, H, T]</c>.</param>
    public Tensor Forward(IBackend backend, Tensor x, ReadOnlySpan<float> cameraConditions, int frames, int latentHeight,
        int latentWidth, Tensor ropeCos, Tensor ropeSin, Tensor beta, Tensor decay)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(ropeCos);
        ArgumentNullException.ThrowIfNull(ropeSin);
        ArgumentNullException.ThrowIfNull(beta);
        ArgumentNullException.ThrowIfNull(decay);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_queryWeight is null || _convKernel is null)
            throw new InvalidOperationException("LoadWeights must run before Forward.");
        if (frames <= 0 || latentHeight <= 0 || latentWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(frames), "Frame and latent grid extents must be positive.");

        int batch = (int)x.Shape[0];
        int spatial = latentHeight * latentWidth;
        int tokens = frames * spatial;
        if (x.DType != DType.F32 || x.Shape.Rank != 3 || x.Shape[1] != tokens || x.Shape[2] != _channels)
            throw new ArgumentException($"x must be F32 [B, {tokens}, {_channels}]; got {x.DType} {x.Shape}.", nameof(x));
        if (cameraConditions.Length != batch * frames * SanaWmCamRays.ConditionWidth)
            throw new ArgumentException(
                $"Camera conditions must hold {batch * frames * SanaWmCamRays.ConditionWidth} floats.", nameof(cameraConditions));
        ValidateGate(beta, [batch, _heads, frames, spatial], nameof(beta));
        ValidateGate(decay, [batch, _heads, frames], nameof(decay));

        TensorShape activation = new(batch, tokens, _channels);
        TensorShape rayShape = new(batch * tokens, 4, 4);
        float keyScale = MathF.Pow(_headDim, -0.5f) * MathF.Pow(spatial, -0.5f);

        using Tensor hostP = new(rayShape, DType.F32, DeviceKind.Cpu);
        using Tensor hostPT = new(rayShape, DType.F32, DeviceKind.Cpu);
        using Tensor hostPI = new(rayShape, DType.F32, DeviceKind.Cpu);
        SanaWmCamRays.Build(cameraConditions, batch, frames, latentHeight, latentWidth,
            hostP.AsSpan<float>(), hostPT.AsSpan<float>(), hostPI.AsSpan<float>());
        using Tensor rays = Upload(backend, hostP);
        using Tensor raysT = Upload(backend, hostPT);
        using Tensor raysInv = Upload(backend, hostPI);

        (Tensor Cos, Tensor Sin) tables = SanaWmCamRope.BuildCameraTables(backend, ropeCos, ropeSin, _headDim);
        using Tensor cos = tables.Cos;
        using Tensor sin = tables.Sin;
        using Tensor negSin = new(sin.Shape, DType.F32, backend.Device);
        backend.Scale(negSin, sin, -1f);

        Tensor? queryFinal = null;
        Tensor? keyFinal = null;
        Tensor? valueFinal = null;
        Tensor? betaCam = null;
        Tensor? scanned = null;
        try
        {
            queryFinal = QueryPath(backend, x, activation, raysT, cos, sin, batch, tokens);
            (betaCam, keyFinal) = KeyPath(backend, x, activation, keyScale, raysInv, cos, sin, beta, batch, frames, spatial);
            valueFinal = ValuePath(backend, x, activation, raysInv, cos, sin, batch, tokens);

            scanned = new Tensor(activation, DType.F32, backend.Device);
            SanaWmCamScan.Run(backend, scanned, queryFinal, keyFinal, valueFinal, betaCam, decay, batch, frames, spatial, _heads, _headDim);
            queryFinal.Dispose();
            keyFinal.Dispose();
            valueFinal.Dispose();
            betaCam.Dispose();
            queryFinal = keyFinal = valueFinal = betaCam = null;

            using Tensor readout = SanaWmCamUcpe.Apply(backend, scanned, rays, cos, negSin, batch, tokens, _heads, _headDim);
            scanned.Dispose();
            scanned = null;

            Tensor camRaw = new(activation, DType.F32, backend.Device);
            bool built = false;
            try
            {
                backend.Linear(camRaw, readout, _outputWeight!, _outputBias);
                built = true;
                return camRaw;
            }
            finally
            {
                if (!built)
                    camRaw.Dispose();
            }
        }
        finally
        {
            queryFinal?.Dispose();
            keyFinal?.Dispose();
            valueFinal?.Dispose();
            betaCam?.Dispose();
            scanned?.Dispose();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Tensor? kernel = Interlocked.Exchange(ref _convKernel, null);
        kernel?.Dispose();
    }

    /// <summary>Q path: <c>UCPE_{P^T}(ReLU(RMSNorm(Linear(x))))</c>, with RoPE on the second half.</summary>
    private Tensor QueryPath(IBackend backend, Tensor x, TensorShape activation, Tensor raysT, Tensor cos, Tensor sin,
        int batch, int tokens)
    {
        Tensor activated = ProjectQuery(backend, x, activation);
        try
        {
            return SanaWmCamUcpe.Apply(backend, activated, raysT, cos, sin, batch, tokens, _heads, _headDim);
        }
        finally
        {
            activated.Dispose();
        }
    }

    /// <summary>K path: <c>k' = ReLU(RMSNorm(conv(Linear(x)))) · k_scale</c>. Returns the discounted beta and the transformed
    /// key <c>UCPE_{P^-1}(k')</c>; the discount reads <c>k'</c> before it is released.</summary>
    private (Tensor Beta, Tensor Key) KeyPath(IBackend backend, Tensor x, TensorShape activation, float keyScale,
        Tensor raysInv, Tensor cos, Tensor sin, Tensor beta, int batch, int frames, int spatial)
    {
        int tokens = frames * spatial;
        Tensor keyScaled = ProjectKey(backend, x, activation, keyScale, batch, frames, spatial);
        try
        {
            Tensor key = SanaWmCamUcpe.Apply(backend, keyScaled, raysInv, cos, sin, batch, tokens, _heads, _headDim);
            bool returned = false;
            try
            {
                Tensor discounted = DiscountBeta(backend, keyScaled, raysInv, cos, sin, beta, batch, frames, spatial);
                returned = true;
                return (discounted, key);
            }
            finally
            {
                if (!returned)
                    key.Dispose();
            }
        }
        finally
        {
            keyScaled.Dispose();
        }
    }

    /// <summary>V path: <c>UCPE_{P^-1}(Linear(x))</c>; values are neither normalized nor rectified.</summary>
    private Tensor ValuePath(IBackend backend, Tensor x, TensorShape activation, Tensor raysInv, Tensor cos, Tensor sin,
        int batch, int tokens)
    {
        Tensor raw = new(activation, DType.F32, backend.Device);
        try
        {
            backend.Linear(raw, x, _valueWeight!, _valueBias);
            return SanaWmCamUcpe.Apply(backend, raw, raysInv, cos, sin, batch, tokens, _heads, _headDim);
        }
        finally
        {
            raw.Dispose();
        }
    }

    /// <summary>Linear, then RMSNorm over all channels, then ReLU: the query activation.</summary>
    private Tensor ProjectQuery(IBackend backend, Tensor x, TensorShape activation)
    {
        Tensor raw = new(activation, DType.F32, backend.Device);
        try
        {
            backend.Linear(raw, x, _queryWeight!, _queryBias);
            return NormActivate(backend, raw, _queryNorm!, activation);
        }
        finally
        {
            raw.Dispose();
        }
    }

    /// <summary>Linear, then the depthwise temporal conv, then RMSNorm and ReLU, then the k-scale: the key activation.</summary>
    private Tensor ProjectKey(IBackend backend, Tensor x, TensorShape activation, float keyScale, int batch, int frames, int spatial)
    {
        Tensor raw = new(activation, DType.F32, backend.Device);
        try
        {
            backend.Linear(raw, x, _keyWeight!, _keyBias);
            Tensor convolved = new(activation, DType.F32, backend.Device);
            try
            {
                TemporalConvolve(backend, convolved, raw, _convKernel!, batch, frames, spatial, _channels);
                Tensor activated = NormActivate(backend, convolved, _keyNorm!, activation);
                bool built = false;
                try
                {
                    backend.Scale(activated, activated, keyScale);
                    built = true;
                    return activated;
                }
                finally
                {
                    if (!built)
                        activated.Dispose();
                }
            }
            finally
            {
                convolved.Dispose();
            }
        }
        finally
        {
            raw.Dispose();
        }
    }

    /// <summary>ReLU(RMSNorm(input)) over the full channel dim, into a new tensor.</summary>
    private Tensor NormActivate(IBackend backend, Tensor input, Tensor norm, TensorShape activation)
    {
        Tensor normed = new(activation, DType.F32, backend.Device);
        try
        {
            backend.RmsNorm(normed, input, norm, NormEpsilon);
            Tensor activated = new(activation, DType.F32, backend.Device);
            bool built = false;
            try
            {
                backend.LeakyRelu(activated, normed, 0f);
                built = true;
                return activated;
            }
            finally
            {
                if (!built)
                    activated.Dispose();
            }
        }
        finally
        {
            normed.Dispose();
        }
    }

    /// <summary>Divides the shared beta per frame by <c>max(mean over S of ||k_post||² / ||k_pre||², 1)</c>. The per-token ratio
    /// comes from the key direction normalized per head: with <c>k̂</c> unit-RMS per head, <c>post(k̂)/D</c> equals
    /// <c>post(k)/pre(k)</c> up to the RMS epsilon, so no device division is needed. <see cref="IBackend"/> has no
    /// elementwise divide, so the per-frame reciprocal is inverted on the host (<c>B·H·T</c> values read back).</summary>
    private Tensor DiscountBeta(IBackend backend, Tensor keyScaled, Tensor raysInv, Tensor cos, Tensor sin, Tensor beta,
        int batch, int frames, int spatial)
    {
        int tokens = frames * spatial;
        int rows = batch * tokens * _heads;
        int groups = batch * _heads * frames;
        TensorShape headRows = new(rows, _headDim);
        using Tensor onesHead = Ones(backend, 1, _headDim);
        using Tensor onesSpatial = Ones(backend, 1, spatial);

        using Tensor keyHat = new(headRows, DType.F32, backend.Device);
        backend.RmsNorm(keyHat, keyScaled.Reshape(headRows), onesHead, ProbeEpsilon);
        using Tensor probe = SanaWmCamUcpe.Apply(backend, keyHat, raysInv, cos, sin, batch, tokens, _heads, _headDim);
        using Tensor squares = new(headRows, DType.F32, backend.Device);
        backend.Mul(squares, probe, probe);
        using Tensor perToken = new(new TensorShape(rows, 1), DType.F32, backend.Device);
        backend.Linear(perToken, squares, onesHead, null);

        using Tensor grouped = new(new TensorShape(groups, spatial), DType.F32, backend.Device);
        ReadOnlySpan<long> groupDims = [batch, _heads, frames, spatial];
        ReadOnlySpan<long> groupStrides = [(long)tokens * _heads, 1, (long)spatial * _heads, _heads];
        backend.CopyStrided(grouped, perToken, StridedCopySpec.Gather(new StridedView(groupDims, groupStrides, 0)));
        using Tensor sums = new(new TensorShape(groups, 1), DType.F32, backend.Device);
        backend.Linear(sums, grouped, onesSpatial, null);

        using Tensor factor = InvertPerFrame(backend, sums, groups, batch, frames, spatial);
        using Tensor factorFull = new(beta.Shape, DType.F32, backend.Device);
        ReadOnlySpan<long> frameDims = [batch, _heads, frames, spatial];
        ReadOnlySpan<long> frameStrides = [(long)_heads * frames, frames, 1, 0];
        backend.CopyStrided(factorFull, factor, StridedCopySpec.Gather(new StridedView(frameDims, frameStrides, 0)));
        Tensor discounted = new(beta.Shape, DType.F32, backend.Device);
        bool built = false;
        try
        {
            backend.Mul(discounted, beta, factorFull);
            built = true;
            return discounted;
        }
        finally
        {
            if (!built)
                discounted.Dispose();
        }
    }

    /// <summary>Returns <c>[B, H, T]</c> factors <c>1 / max(sum / (D·S), 1)</c> from per-frame key sums.</summary>
    private Tensor InvertPerFrame(IBackend backend, Tensor sums, int groups, int batch, int frames, int spatial)
    {
        using Tensor hostSums = new(sums.Shape, DType.F32, DeviceKind.Cpu);
        backend.CopyTo(hostSums, sums);
        using Tensor hostFactor = new(new TensorShape(batch, _heads, frames), DType.F32, DeviceKind.Cpu);
        ReadOnlySpan<float> totals = hostSums.AsReadOnlySpan<float>();
        Span<float> factors = hostFactor.AsSpan<float>();
        float perFrame = 1f / ((float)_headDim * spatial);
        for (int i = 0; i < groups; i++)
            factors[i] = 1f / MathF.Max(totals[i] * perFrame, 1f);
        return Upload(backend, hostFactor);
    }

    /// <summary>Depthwise causal conv along frames for each (batch, token-in-frame) lane of a <c>[B, N, C]</c> tensor.</summary>
    private static void TemporalConvolve(IBackend backend, Tensor output, Tensor input, Tensor kernel, int batch, int frames,
        int spatial, int channels)
    {
        ReadOnlySpan<long> dims = [batch, spatial, channels, frames];
        ReadOnlySpan<long> strides = [(long)frames * spatial * channels, channels, 1, (long)spatial * channels];
        StridedView lanes = new(dims, strides, 0);
        using Tensor sequences = new(new TensorShape(batch * spatial, channels, frames), DType.F32, backend.Device);
        backend.CopyStrided(sequences, input, StridedCopySpec.Gather(lanes));
        using Tensor convolved = new(sequences.Shape, DType.F32, backend.Device);
        backend.Conv1d(convolved, sequences, kernel, null, 1, 3, 3, 1, channels);
        backend.CopyStrided(output, convolved, StridedCopySpec.Scatter(lanes));
    }

    private static void ValidateGate(Tensor gate, ReadOnlySpan<long> expected, string name)
    {
        if (gate.DType != DType.F32 || gate.Shape.Rank != expected.Length)
            throw new ArgumentException($"{name} must be F32 with rank {expected.Length}; got {gate.DType} {gate.Shape}.", name);
        for (int i = 0; i < expected.Length; i++)
        {
            if (gate.Shape[i] != expected[i])
                throw new ArgumentException($"{name} axis {i} must be {expected[i]}; got {gate.Shape}.", name);
        }
    }

    private Tensor Matrix(IReadOnlyDictionary<string, Tensor> weights, string prefix, string key)
    {
        Tensor tensor = TensorCasts.EnsureF32(weights[$"{prefix}{key}"]);
        if (tensor.Shape.Rank != 2 || tensor.Shape[0] != _channels || tensor.Shape[1] != _channels)
            throw new ArgumentException($"{prefix}{key} must be [{_channels}, {_channels}]; got {tensor.Shape}.");
        return tensor;
    }

    private Tensor Vector(IReadOnlyDictionary<string, Tensor> weights, string prefix, string key)
    {
        Tensor tensor = TensorCasts.EnsureF32(weights[$"{prefix}{key}"]);
        if (tensor.Shape.Rank != 1 || tensor.Shape[0] != _channels)
            throw new ArgumentException($"{prefix}{key} must be [{_channels}]; got {tensor.Shape}.");
        return tensor;
    }

    /// <summary>Folds <c>[w0, w1, w2, w3]</c> into the symmetric taps <c>[w0, w1, w2, w3, w2, w1, w0]</c>, which equals
    /// <c>causal(x) + causal(flip x)·flip − w3·x</c> with zero padding.</summary>
    private Tensor SymmetricKernel(Tensor conv)
    {
        ReadOnlySpan<float> source = conv.AsReadOnlySpan<float>();
        Tensor kernel = new(new TensorShape(_channels, 1, 7), DType.F32, DeviceKind.Cpu);
        Span<float> taps = kernel.AsSpan<float>();
        for (int c = 0; c < _channels; c++)
        {
            float w0 = source[c * 4], w1 = source[c * 4 + 1], w2 = source[c * 4 + 2], w3 = source[c * 4 + 3];
            Span<float> row = taps.Slice(c * 7, 7);
            row[0] = w0; row[1] = w1; row[2] = w2; row[3] = w3; row[4] = w2; row[5] = w1; row[6] = w0;
        }
        return kernel;
    }

    private static Tensor Upload(IBackend backend, Tensor host)
    {
        Tensor device = new(host.Shape, DType.F32, backend.Device);
        backend.CopyTo(device, host);
        return device;
    }

    private static Tensor Ones(IBackend backend, int rows, int columns)
    {
        using Tensor host = new(new TensorShape(rows, columns), DType.F32, DeviceKind.Cpu);
        host.AsSpan<float>().Fill(1f);
        return Upload(backend, host);
    }
}
