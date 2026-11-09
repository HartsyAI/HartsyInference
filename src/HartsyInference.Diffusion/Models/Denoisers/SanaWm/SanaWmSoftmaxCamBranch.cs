using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Camera branch of a SANA-WM softmax attention layer (upstream <c>_forward_cam_branch_softmax</c>). Q/K/V are
/// projected from the same input, normalized, UCPE-transformed, then attended with softmax over every token. There is no
/// temporal conv, ReLU, key scale or beta/decay gate, unlike <see cref="SanaWmCamBranch"/>. The result is
/// <c>out_proj_cam(·)</c>, which the block sums with the main branch before the output gate. Runs one batch item at a time.</summary>
public sealed class SanaWmSoftmaxCamBranch : IDisposable
{
    private const float NormEpsilon = 1e-5f;

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
    private Tensor? _queryNorm;
    private Tensor? _keyNorm;
    private Tensor? _outputWeight;
    private Tensor? _outputBias;
    private int _disposed;

    /// <param name="heads">Attention heads; the camera branch shares the main branch's head split.</param>
    /// <param name="headDim">Per-head dim. Must be a multiple of 8 and at least 24 so the UCPE and RoPE halves split.</param>
    public SanaWmSoftmaxCamBranch(int heads, int headDim)
    {
        if (heads <= 0)
            throw new ArgumentOutOfRangeException(nameof(heads), heads, "Head count must be positive.");
        if (headDim < 24 || headDim % 8 != 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "Head dim must be a multiple of 8 and at least 24.");
        _heads = heads;
        _headDim = headDim;
        _channels = checked(heads * headDim);
    }

    /// <summary>Binds <c>q_proj_cam</c>, <c>k_proj_cam</c>, <c>v_proj_cam</c>, <c>q_norm_cam</c>, <c>k_norm_cam</c> and
    /// <c>out_proj_cam</c>. Matrices are borrowed in their stored dtype (F32 or BF16) and must outlive the branch; vectors become
    /// host F32 copies when they are not F32.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(weights);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        SanaWmTensors.DisposeAll(_owned);
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
    }

    /// <summary>The tensors <see cref="Forward"/> reads, for preloading on a device backend.</summary>
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

    /// <summary>Returns <c>out_proj_cam</c> of the camera softmax, shaped <c>[B, N, C]</c> in F32. The caller owns the result.</summary>
    /// <param name="x">F32 <c>[B, N, C]</c> post-adaLN tokens, frame-major with <c>N = frames·latentHeight·latentWidth</c>.</param>
    /// <param name="cameraConditions">Per frame, 16 row-major camera-to-world floats then fx, fy, cx, cy in latent pixels.</param>
    /// <param name="cameraCos">Per-complex 3D RoPE cosine table <c>[N, headDim/2]</c>; the camera slice is taken from it.</param>
    /// <param name="cameraSin">Per-complex 3D RoPE sine table <c>[N, headDim/2]</c>.</param>
    public Tensor Forward(IBackend backend, Tensor x, ReadOnlySpan<float> cameraConditions, int frames, int latentHeight,
        int latentWidth, Tensor cameraCos, Tensor cameraSin)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(x);
        if (frames <= 0 || latentHeight <= 0 || latentWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(frames), "Frame and latent grid extents must be positive.");
        int batch = (int)x.Shape[0];
        int tokens = frames * latentHeight * latentWidth;
        if (x.DType != DType.F32 || x.Shape.Rank != 3 || x.Shape[1] != tokens || x.Shape[2] != _channels)
            throw new ArgumentException($"x must be F32 [B, {tokens}, {_channels}]; got {x.DType} {x.Shape}.", nameof(x));

        using SanaWmCameraInputs camera = SanaWmCameraInputs.Create(backend, cameraCos, cameraSin, _headDim, cameraConditions, batch,
            frames, latentHeight, latentWidth);
        long itemElements = (long)tokens * _channels;
        using SanaWmScratch scratch = new();
        Tensor output = scratch.New(x.Shape);
        for (int b = 0; b < batch; b++)
        {
            using SanaWmScratch item = new();
            Tensor a = item.Track(SanaWmTensors.Slice(backend, x, b * itemElements, new TensorShape(tokens, _channels)));
            Tensor cam = item.Track(ForwardItem(backend, a, camera, b, tokens));
            SanaWmTensors.Write(backend, output, cam, b * itemElements);
        }

        return scratch.Detach(output);
    }

    /// <summary>One batch item: <paramref name="a"/> is F32 token-major with <c>tokens·C</c> elements. Returns <c>[N, C]</c>.</summary>
    internal Tensor ForwardItem(IBackend backend, Tensor a, SanaWmCameraInputs camera, int item, int tokens, Tensor? projectionInput = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_queryWeight is null || _outputWeight is null)
            throw new InvalidOperationException("LoadWeights must run before Forward.");
        if (a.DType != DType.F32 || a.ElementCount != (long)tokens * _channels)
            throw new ArgumentException($"The item input must be F32 with {tokens}x{_channels} elements; got {a.DType} {a.Shape}.", nameof(a));

        using SanaWmScratch scratch = new();
        Tensor projected = projectionInput ?? a;
        Tensor qh = Heads(backend, scratch, projected, _queryWeight, _queryBias, _queryNorm, camera.RaysTransposed(item), camera, tokens);
        Tensor kh = Heads(backend, scratch, projected, _keyWeight!, _keyBias, _keyNorm, camera.RaysInverse(item), camera, tokens);
        Tensor vh = Heads(backend, scratch, projected, _valueWeight!, _valueBias, null, camera.RaysInverse(item), camera, tokens);
        Tensor attended = scratch.New(1, _heads, tokens, _headDim);
        // BF16 operands with F32 softmax and accumulation, as upstream's SDPA (it casts the camera q, k and v to BF16).
        backend.Attention(AttentionCall.FromScaledDotProductAttention(attended, qh, kh, vh, null, MathF.Pow(_headDim, -0.5f), allowF16: false)
            with { Numerics = AttentionNumerics.Bf16Storage });
        scratch.Release(qh);
        scratch.Release(kh);
        scratch.Release(vh);
        Tensor merged = scratch.New(tokens, _channels);
        backend.Permute0213(merged, attended, _heads, tokens, _headDim);
        scratch.Release(attended);
        Tensor readout = scratch.Track(SanaWmCamUcpe.Apply(backend, merged, camera.Rays(item), camera.Cos, camera.NegSin, tokens, _heads,
            _headDim));
        scratch.Release(merged);
        Tensor output = scratch.New(tokens, _channels);
        backend.Linear(output, readout, _outputWeight, _outputBias);
        return scratch.Detach(output);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        SanaWmTensors.DisposeAll(_owned);
    }

    // Linear, the optional full-channel RMSNorm, the UCPE transform, then head-major [1, H, N, D].
    private Tensor Heads(IBackend backend, SanaWmScratch scratch, Tensor a, Tensor weight, Tensor? bias, Tensor? norm, Tensor matrices,
        SanaWmCameraInputs camera, int tokens)
    {
        Tensor projected = scratch.New(tokens, _channels);
        backend.Linear(projected, a, weight, bias);
        if (norm is not null)
        {
            Tensor normed = scratch.New(tokens, _channels);
            backend.RmsNorm(normed, projected, norm, NormEpsilon);
            scratch.Release(projected);
            projected = normed;
        }

        Tensor transformed = scratch.Track(SanaWmCamUcpe.Apply(backend, projected, matrices, camera.Cos, camera.Sin, tokens, _heads,
            _headDim));
        scratch.Release(projected);
        Tensor heads = scratch.New(1, _heads, tokens, _headDim);
        backend.Permute0213(heads, transformed, tokens, _heads, _headDim);
        scratch.Release(transformed);
        return heads;
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
}
