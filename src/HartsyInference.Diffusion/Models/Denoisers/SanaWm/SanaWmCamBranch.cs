using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>SANA-WM camera branch of one attention layer (upstream UCPE single-path camera branch). It takes the same
/// post-adaLN input as the main branch, projects it with its own Q/K/V, applies the per-token ray transform and RoPE, and
/// runs the numerator-only bidirectional scan that shares the main branch's beta and decay. The result is
/// <c>out_proj_cam(·)</c>; the block sums it with the main branch output. Runs one batch item at a time.</summary>
public sealed class SanaWmCamBranch : IDisposable
{
    private const float NormEpsilon = 1e-5f;
    private const float ProbeEpsilon = 1e-30f;

    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _channels;
    private readonly List<Tensor> _owned = [];

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
    private Tensor? _headOnes;
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
    /// <c>k_norm_cam</c> and <c>out_proj_cam</c> under <paramref name="prefix"/>. Matrices are borrowed in their stored dtype (F32
    /// or BF16); vectors become host F32 copies when they are not F32. The depthwise k=4 conv becomes a symmetric 7-tap kernel,
    /// which is the bidirectional trick folded into one filter.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(weights);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        SanaWmTensors.DisposeAll(_owned);
        _convKernel = Own(SanaWmTemporalConv.SymmetricKernel(weights[$"{prefix}conv_k_cam.weight"], _channels, "conv_k_cam.weight"));
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
        _headOnes = Own(SanaWmTensors.Filled(new TensorShape(1, _headDim), 1f));
    }

    /// <summary>The tensors <see cref="Forward"/> reads, for preloading on a device backend.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_queryWeight, _queryBias, _keyWeight, _keyBias, _valueWeight, _valueBias, _queryNorm, _keyNorm,
            _outputWeight, _outputBias, _convKernel, _headOnes];
        foreach (Tensor? tensor in all)
        {
            if (tensor is not null)
                yield return tensor;
        }
    }

    /// <summary>Runs the camera branch over every batch item and returns <c>out_proj_cam</c> of the scan, shaped <c>[B, N, C]</c>.
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
        ArgumentNullException.ThrowIfNull(beta);
        ArgumentNullException.ThrowIfNull(decay);
        if (frames <= 0 || latentHeight <= 0 || latentWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(frames), "Frame and latent grid extents must be positive.");

        int batch = (int)x.Shape[0];
        int spatial = latentHeight * latentWidth;
        int tokens = frames * spatial;
        if (x.DType != DType.F32 || x.Shape.Rank != 3 || x.Shape[1] != tokens || x.Shape[2] != _channels)
            throw new ArgumentException($"x must be F32 [B, {tokens}, {_channels}]; got {x.DType} {x.Shape}.", nameof(x));
        SanaWmTensors.Require(beta, nameof(beta), DType.F32, [batch, _heads, frames, spatial]);
        SanaWmTensors.Require(decay, nameof(decay), DType.F32, [batch, _heads, frames]);

        using SanaWmCameraInputs camera = SanaWmCameraInputs.Create(backend, ropeCos, ropeSin, _headDim, cameraConditions, batch,
            frames, latentHeight, latentWidth);
        long itemElements = (long)tokens * _channels;
        using SanaWmScratch scratch = new();
        Tensor output = scratch.New(new TensorShape(batch, tokens, _channels));
        for (int b = 0; b < batch; b++)
        {
            using SanaWmScratch item = new();
            Tensor a = item.Track(SanaWmTensors.Slice(backend, x, b * itemElements, new TensorShape(tokens, _channels)));
            Tensor itemBeta = item.Track(SanaWmTensors.Slice(backend, beta, (long)b * _heads * tokens, new TensorShape(_heads, frames, spatial)));
            Tensor itemDecay = item.Track(SanaWmTensors.Slice(backend, decay, (long)b * _heads * frames, new TensorShape(_heads, frames)));
            Tensor cam = item.Track(ForwardItem(backend, a, camera, b, itemBeta, itemDecay, frames, spatial));
            SanaWmTensors.Write(backend, output, cam, b * itemElements);
        }

        return scratch.Detach(output);
    }

    /// <summary>One batch item: <paramref name="a"/> is F32 token-major <c>[N, C]</c> (any shape of that size), <paramref name="beta"/>
    /// <c>[H, T, S]</c> and <paramref name="decay"/> <c>[H, T]</c>. Returns <c>out_proj_cam</c> of the scan, <c>[N, C]</c>.</summary>
    internal Tensor ForwardItem(IBackend backend, Tensor a, SanaWmCameraInputs camera, int item, Tensor beta, Tensor decay, int frames,
        int spatial)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_queryWeight is null || _convKernel is null)
            throw new InvalidOperationException("LoadWeights must run before Forward.");
        int tokens = checked(frames * spatial);
        if (a.DType != DType.F32 || a.ElementCount != (long)tokens * _channels)
            throw new ArgumentException($"The item input must be F32 with {tokens}x{_channels} elements; got {a.DType} {a.Shape}.", nameof(a));
        float keyScale = MathF.Pow(_headDim, -0.5f) * MathF.Pow(spatial, -0.5f);

        using SanaWmScratch scratch = new();
        Tensor keyScaled = scratch.Track(ProjectKey(backend, a, keyScale, frames, spatial));
        Tensor discounted = scratch.Track(DiscountBeta(backend, keyScaled, camera, item, beta, frames, spatial));
        Tensor key = Transform(backend, scratch, keyScaled, camera.RaysInverse(item), camera.Cos, camera.Sin, tokens);

        Tensor queryRaw = scratch.New(tokens, _channels);
        backend.Linear(queryRaw, a, _queryWeight, _queryBias);
        Tensor queryActivated = NormActivate(backend, scratch, queryRaw, _queryNorm!, new TensorShape(tokens, _channels));
        Tensor query = Transform(backend, scratch, queryActivated, camera.RaysTransposed(item), camera.Cos, camera.Sin, tokens);

        Tensor valueRaw = scratch.New(tokens, _channels);
        backend.Linear(valueRaw, a, _valueWeight!, _valueBias);
        Tensor value = Transform(backend, scratch, valueRaw, camera.RaysInverse(item), camera.Cos, camera.Sin, tokens);

        Tensor scanned = scratch.Track(SanaWmCamScan.Run(backend, query, key, value, discounted, decay, frames, spatial, _heads, _headDim));
        scratch.Release(query);
        scratch.Release(key);
        scratch.Release(value);
        scratch.Release(discounted);
        Tensor readout = Transform(backend, scratch, scanned, camera.Rays(item), camera.Cos, camera.NegSin, tokens);
        Tensor output = scratch.New(tokens, _channels);
        backend.Linear(output, readout, _outputWeight!, _outputBias);
        return scratch.Detach(output);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        SanaWmTensors.DisposeAll(_owned);
    }

    // Releases input; returns the tracked UCPE transform of it.
    private Tensor Transform(IBackend backend, SanaWmScratch scratch, Tensor input, Tensor matrices, Tensor cos, Tensor sin, int tokens)
    {
        Tensor transformed = scratch.Track(SanaWmCamUcpe.Apply(backend, input, matrices, cos, sin, tokens, _heads, _headDim));
        scratch.Release(input);
        return transformed;
    }

    /// <summary>K path: <c>k' = ReLU(RMSNorm(conv(Linear(x)))) · k_scale</c>, shaped <c>[N·H, D]</c> so the discount's per-head
    /// norm reads it directly.</summary>
    private Tensor ProjectKey(IBackend backend, Tensor a, float keyScale, int frames, int spatial)
    {
        int tokens = frames * spatial;
        using SanaWmScratch scratch = new();
        Tensor raw = scratch.New(tokens, _channels);
        backend.Linear(raw, a, _keyWeight!, _keyBias);
        Tensor convolved = scratch.Track(SanaWmTemporalConv.Apply(backend, raw, _convKernel!, frames, spatial, _channels));
        scratch.Release(raw);
        Tensor activated = NormActivate(backend, scratch, convolved, _keyNorm!, new TensorShape(tokens, _channels));
        Tensor scaled = scratch.New((long)tokens * _heads, _headDim);
        backend.Scale(scaled, activated, keyScale);
        return scratch.Detach(scaled);
    }

    // Releases input; returns a tracked ReLU(RMSNorm(input)) over the full channel dim.
    private static Tensor NormActivate(IBackend backend, SanaWmScratch scratch, Tensor input, Tensor norm, TensorShape shape)
    {
        Tensor normed = scratch.New(shape);
        backend.RmsNorm(normed, input, norm, NormEpsilon);
        scratch.Release(input);
        Tensor activated = scratch.New(shape);
        backend.LeakyRelu(activated, normed, 0f);
        scratch.Release(normed);
        return activated;
    }

    /// <summary>Divides the shared beta per frame by <c>max(mean over S of ||k_post||² / ||k_pre||², 1)</c>. The per-token ratio
    /// comes from the key direction normalized per head: with <c>k̂</c> unit-RMS per head, <c>post(k̂)/D</c> equals
    /// <c>post(k)/pre(k)</c> up to the RMS epsilon. Returns <c>[H, T, S]</c>.</summary>
    private Tensor DiscountBeta(IBackend backend, Tensor keyScaled, SanaWmCameraInputs camera, int item, Tensor beta, int frames,
        int spatial)
    {
        int tokens = frames * spatial;
        long rows = (long)tokens * _heads;
        int groups = _heads * frames;
        using SanaWmScratch scratch = new();
        Tensor keyHat = scratch.New(rows, _headDim);
        backend.RmsNorm(keyHat, keyScaled, _headOnes!, ProbeEpsilon);
        Tensor probe = Transform(backend, scratch, keyHat, camera.RaysInverse(item), camera.Cos, camera.Sin, tokens);
        Tensor squares = scratch.New(rows, _headDim);
        backend.Mul(squares, probe, probe);
        scratch.Release(probe);
        Tensor perToken = scratch.New(rows, 1);
        backend.Linear(perToken, squares, _headOnes!, null);
        scratch.Release(squares);

        Tensor grouped = scratch.New(groups, spatial);
        ReadOnlySpan<long> groupDims = [_heads, frames, spatial];
        ReadOnlySpan<long> groupStrides = [1, (long)spatial * _heads, _heads];
        backend.CopyStrided(grouped, perToken, StridedCopySpec.Gather(new StridedView(groupDims, groupStrides, 0)));
        scratch.Release(perToken);
        Tensor spatialOnes = scratch.Track(SanaWmTensors.Filled(new TensorShape(1, spatial), 1f));
        Tensor sums = scratch.New(groups, 1);
        backend.Linear(sums, grouped, spatialOnes, null);
        scratch.Release(grouped);

        // factor = 1 / max(sum / (D·S), 1), on the device.
        Tensor mean = scratch.New(groups, 1);
        backend.Scale(mean, sums, 1f / ((float)_headDim * spatial));
        Tensor clamped = scratch.New(groups, 1);
        backend.Clamp(clamped, mean, 1f, float.MaxValue);
        Tensor factor = scratch.New(groups, 1);
        backend.Reciprocal(factor, clamped);
        Tensor factorFull = scratch.New(_heads, frames, spatial);
        ReadOnlySpan<long> frameDims = [_heads, frames, spatial];
        ReadOnlySpan<long> frameStrides = [frames, 1, 0];
        backend.CopyStrided(factorFull, factor, StridedCopySpec.Gather(new StridedView(frameDims, frameStrides, 0)));
        Tensor discounted = scratch.New(_heads, frames, spatial);
        backend.Mul(discounted, beta, factorFull);
        return scratch.Detach(discounted);
    }

    private Tensor Matrix(IReadOnlyDictionary<string, Tensor> weights, string prefix, string key)
    {
        Tensor tensor = weights[$"{prefix}{key}"];
        SanaWmTensors.RequireMatrix(tensor, $"{prefix}{key}", _channels, _channels);
        return tensor;
    }

    private Tensor Vector(IReadOnlyDictionary<string, Tensor> weights, string prefix, string key)
    {
        Tensor tensor = SanaWmTensors.HostF32(weights[$"{prefix}{key}"], _owned);
        if (tensor.Shape.Rank != 1 || tensor.Shape[0] != _channels)
            throw new ArgumentException($"{prefix}{key} must be [{_channels}]; got {tensor.Shape}.");
        return tensor;
    }

    private Tensor Own(Tensor tensor)
    {
        _owned.Add(tensor);
        return tensor;
    }
}
