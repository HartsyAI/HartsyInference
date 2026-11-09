using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Camera branch of a SANA-WM softmax attention layer (upstream <c>_forward_cam_branch_softmax</c>). Q/K/V are
/// projected from the same input, normalized, UCPE-transformed, then attended with softmax over every token. There is no
/// temporal conv, ReLU, key scale or beta/decay gate, unlike <see cref="SanaWmCamBranch"/>. The result is
/// <c>out_proj_cam(·)</c>, which the block sums with the main branch before the output gate.</summary>
public sealed class SanaWmSoftmaxCamBranch : IDisposable
{
    private const float NormEpsilon = 1e-5f;

    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _channels;

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
    /// <c>out_proj_cam</c>. Every tensor must already be F32 and owned by the caller, which keeps them alive.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(weights);
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
        ArgumentNullException.ThrowIfNull(cameraCos);
        ArgumentNullException.ThrowIfNull(cameraSin);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_queryWeight is null || _outputWeight is null)
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

        TensorShape activation = new(batch, tokens, _channels);
        TensorShape rayShape = new(batch * tokens, 4, 4);
        using Tensor hostP = new(rayShape, DType.F32, DeviceKind.Cpu);
        using Tensor hostPT = new(rayShape, DType.F32, DeviceKind.Cpu);
        using Tensor hostPI = new(rayShape, DType.F32, DeviceKind.Cpu);
        SanaWmCamRays.Build(cameraConditions, batch, frames, latentHeight, latentWidth,
            hostP.AsSpan<float>(), hostPT.AsSpan<float>(), hostPI.AsSpan<float>());
        using Tensor rays = SanaWmTensors.Upload(backend, hostP);
        using Tensor raysT = SanaWmTensors.Upload(backend, hostPT);
        using Tensor raysInv = SanaWmTensors.Upload(backend, hostPI);

        (Tensor Cos, Tensor Sin) tables = SanaWmCamRope.BuildCameraTables(backend, cameraCos, cameraSin, _headDim);
        using Tensor cos = tables.Cos;
        using Tensor sin = tables.Sin;
        using Tensor negSin = new(sin.Shape, DType.F32, backend.Device);
        backend.Scale(negSin, sin, -1f);

        List<Tensor> scratch = [];
        try
        {
            Tensor queryRaw = Projected(backend, scratch, x, _queryWeight!, _queryBias, activation);
            Tensor queries = Transformed(backend, scratch, Normed(backend, scratch, queryRaw, _queryNorm!, activation), raysT, cos, sin,
                batch, tokens);
            Tensor keyRaw = Projected(backend, scratch, x, _keyWeight!, _keyBias, activation);
            Tensor keys = Transformed(backend, scratch, Normed(backend, scratch, keyRaw, _keyNorm!, activation), raysInv, cos, sin,
                batch, tokens);
            Tensor valueRaw = Projected(backend, scratch, x, _valueWeight!, _valueBias, activation);
            Tensor values = Transformed(backend, scratch, valueRaw, raysInv, cos, sin, batch, tokens);

            TensorShape tokenMajor = new(batch, tokens, _heads, _headDim);
            TensorShape headMajor = new(batch, _heads, tokens, _headDim);
            Tensor qh = NewTensor(backend, scratch, headMajor);
            Tensor kh = NewTensor(backend, scratch, headMajor);
            Tensor vh = NewTensor(backend, scratch, headMajor);
            backend.Permute0213(qh, queries.Reshape(tokenMajor), tokens, _heads, _headDim);
            backend.Permute0213(kh, keys.Reshape(tokenMajor), tokens, _heads, _headDim);
            backend.Permute0213(vh, values.Reshape(tokenMajor), tokens, _heads, _headDim);

            Tensor attended = NewTensor(backend, scratch, headMajor);
            backend.ScaledDotProductAttention(attended, qh, kh, vh, null, MathF.Pow(_headDim, -0.5f));
            Tensor merged = NewTensor(backend, scratch, activation);
            backend.Permute0213(merged.Reshape(tokenMajor), attended, _heads, tokens, _headDim);

            using Tensor readout = SanaWmCamUcpe.Apply(backend, merged, rays, cos, negSin, batch, tokens, _heads, _headDim);
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
            SanaWmTensors.DisposeAll(scratch);
        }
    }

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }

    private static Tensor Projected(IBackend backend, List<Tensor> scratch, Tensor x, Tensor weight, Tensor? bias, TensorShape activation)
    {
        Tensor raw = NewTensor(backend, scratch, activation);
        backend.Linear(raw, x, weight, bias);
        return raw;
    }

    private Tensor Normed(IBackend backend, List<Tensor> scratch, Tensor input, Tensor norm, TensorShape activation)
    {
        Tensor normed = NewTensor(backend, scratch, activation);
        backend.RmsNorm(normed, input, norm, NormEpsilon);
        return normed;
    }

    private Tensor Transformed(IBackend backend, List<Tensor> scratch, Tensor input, Tensor matrices, Tensor cos, Tensor sin,
        int batch, int tokens)
    {
        Tensor transformed = SanaWmCamUcpe.Apply(backend, input, matrices, cos, sin, batch, tokens, _heads, _headDim);
        scratch.Add(transformed);
        return transformed;
    }

    private static Tensor NewTensor(IBackend backend, List<Tensor> scratch, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32, backend.Device);
        scratch.Add(tensor);
        return tensor;
    }

    private Tensor Matrix(IReadOnlyDictionary<string, Tensor> weights, string prefix, string key)
    {
        Tensor tensor = Fetch(weights, prefix, key);
        if (tensor.Shape.Rank != 2 || tensor.Shape[0] != _channels || tensor.Shape[1] != _channels)
            throw new ArgumentException($"{prefix}{key} must be [{_channels}, {_channels}]; got {tensor.Shape}.");
        return tensor;
    }

    private Tensor Vector(IReadOnlyDictionary<string, Tensor> weights, string prefix, string key)
    {
        Tensor tensor = Fetch(weights, prefix, key);
        if (tensor.Shape.Rank != 1 || tensor.Shape[0] != _channels)
            throw new ArgumentException($"{prefix}{key} must be [{_channels}]; got {tensor.Shape}.");
        return tensor;
    }

    private static Tensor Fetch(IReadOnlyDictionary<string, Tensor> weights, string prefix, string key)
    {
        Tensor tensor = weights[$"{prefix}{key}"];
        if (tensor.DType != DType.F32)
            throw new ArgumentException($"{prefix}{key} must be F32; convert it before LoadWeights.");
        return tensor;
    }
}
