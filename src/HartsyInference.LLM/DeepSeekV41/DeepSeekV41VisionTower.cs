using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Host reference for the V4.1 vision tower: a patch embedding, pre-norm blocks of full bidirectional attention with a two-dimensional half-split rotary and a SwiGLU MLP, and a final RMSNorm.</summary>
/// <remarks>Follows upstream <c>ViT.forward</c> in float32, one image at a time. The tower owns the weight tensors it is given and disposes them. A call allocates its own workspace, so it is
/// reentrant apart from <see cref="Probe"/>.</remarks>
public sealed class DeepSeekV41VisionTower : IDisposable
{
    // upstream's RMSNorm default; the config has no field for it
    private const float NormEps = 1e-6f;

    private readonly IBackend _backend;
    private readonly DeepSeekV41VisionWeights _weights;
    private bool _disposed;

    /// <param name="backend">Provides linear, norm, rotary, attention and activation ops.</param>
    /// <param name="config">Tower dimensions.</param>
    /// <param name="weights">F32 weights shaped by <paramref name="config"/>; the tower takes ownership.</param>
    /// <exception cref="HartsyInference.Core.Exceptions.HartsyInferenceException">The config is not runnable or a weight is missing, not F32 or mis-shaped.</exception>
    public DeepSeekV41VisionTower(IBackend backend, DeepSeekV41VisionConfig config, DeepSeekV41VisionWeights weights)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        config.Validate();
        weights.Validate(config);
        _backend = backend;
        Config = config;
        _weights = weights;
    }

    /// <summary>The dimensions the tower was built with.</summary>
    public DeepSeekV41VisionConfig Config { get; }

    /// <summary>Diagnostic tap called with a stage name and a copy of its values: <c>patch_embed</c>, <c>block.{i}</c> (the residual stream after block i) and <c>norm</c>. Null (the default) copies nothing.</summary>
    public Action<string, float[]>? Probe { get; set; }

    /// <summary>Runs one image's patches through the tower and returns the normed features, <c>[gridHeight * gridWidth, HiddenSize]</c>.</summary>
    /// <param name="patches">Flattened patches in row-major grid order, <c>[gridHeight * gridWidth, 3 * patch * patch]</c>, channel-major inside a patch.</param>
    /// <param name="gridHeight">Patch rows.</param>
    /// <param name="gridWidth">Patch columns.</param>
    public float[] Forward(ReadOnlySpan<float> patches, int gridHeight, int gridWidth)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfLessThan(gridHeight, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(gridWidth, 1);
        int n = checked(gridHeight * gridWidth);
        if (patches.Length != (long)n * Config.PatchInputDim)
        {
            throw new ArgumentException(
                $"Expected {n} patches of {Config.PatchInputDim} values ({(long)n * Config.PatchInputDim}), got {patches.Length}.", nameof(patches));
        }
        int dim = Config.HiddenSize, heads = Config.NumHeads, headDim = Config.HeadDim, inter = Config.IntermediateSize;

        DeepSeekV41VisionRopeTable rope = DeepSeekV41VisionRopeTable.Build(gridHeight, gridWidth, headDim, Config.RopeTheta);
        using Tensor cos = DeepSeekV41HostMath.Tensor(rope.Cos, 1, n, headDim);
        using Tensor sin = DeepSeekV41HostMath.Tensor(rope.Sin, 1, n, headDim);
        using Tensor input = DeepSeekV41HostMath.Tensor(patches, n, Config.PatchInputDim);
        using Workspace w = new(n, dim, heads, headDim, inter);

        _backend.Linear(w.X, input, _weights.PatchProj, _weights.PatchBias);
        Emit("patch_embed", w.X);
        for (int i = 0; i < _weights.Blocks.Count; i++)
        {
            Attention(_weights.Blocks[i], w, cos, sin, n);
            Mlp(_weights.Blocks[i], w, inter);
            Emit($"block.{i}", w.X);
        }
        _backend.RmsNorm(w.Normed, w.X, _weights.FinalNorm, NormEps);
        Emit("norm", w.Normed);
        return w.Normed.AsReadOnlySpan<float>().ToArray();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _weights.Dispose();
    }

    // x += wo(attention(norm1(x))); q and k are rotated token-major, attention runs head-major
    private void Attention(DeepSeekV41VisionBlockWeights b, Workspace w, Tensor cos, Tensor sin, int n)
    {
        int dim = Config.HiddenSize, heads = Config.NumHeads, headDim = Config.HeadDim;
        _backend.RmsNorm(w.Normed, w.X, b.Norm1, NormEps);
        _backend.Linear(w.Qkv, w.Normed, b.Wqkv, b.WqkvBias);
        _backend.SliceLastDim(w.Q, w.Qkv, 0);
        _backend.SliceLastDim(w.K, w.Qkv, dim);
        _backend.SliceLastDim(w.V, w.Qkv, 2 * dim);
        _backend.ApplyRopeSingle(w.Q, cos, sin);
        _backend.ApplyRopeSingle(w.K, cos, sin);
        _backend.Permute0213(w.QHeads, w.Q, n, heads, headDim);
        _backend.Permute0213(w.KHeads, w.K, n, heads, headDim);
        _backend.Permute0213(w.VHeads, w.V, n, heads, headDim);
        _backend.ScaledDotProductAttention(w.OHeads, w.QHeads, w.KHeads, w.VHeads, null, 1f / MathF.Sqrt(headDim));
        _backend.Permute0213(w.O, w.OHeads, heads, n, headDim);
        _backend.Linear(w.Branch, w.O, b.Wo, b.WoBias);
        _backend.Add(w.X, w.X, w.Branch);
    }

    // x += w2(silu(gate) * up) with gate and up the two halves of w1(norm2(x))
    private void Mlp(DeepSeekV41VisionBlockWeights b, Workspace w, int inter)
    {
        _backend.RmsNorm(w.Normed, w.X, b.Norm2, NormEps);
        _backend.Linear(w.GateUp, w.Normed, b.W1, null);
        _backend.GluActivate(w.Act, w.GateUp, inter, gelu: false);
        _backend.Linear(w.Branch, w.Act, b.W2, null);
        _backend.Add(w.X, w.X, w.Branch);
    }

    private void Emit(string stage, Tensor values) => Probe?.Invoke(stage, values.AsReadOnlySpan<float>().ToArray());

    // The activations of one call, allocated together and freed together.
    private sealed class Workspace : IDisposable
    {
        public Workspace(int n, int dim, int heads, int headDim, int inter)
        {
            X = Alloc(n, dim);
            Normed = Alloc(n, dim);
            Branch = Alloc(n, dim);
            Qkv = Alloc(n, 3 * dim);
            Q = Alloc(1, n, heads, headDim);
            K = Alloc(1, n, heads, headDim);
            V = Alloc(1, n, heads, headDim);
            O = Alloc(1, n, heads, headDim);
            QHeads = Alloc(1, heads, n, headDim);
            KHeads = Alloc(1, heads, n, headDim);
            VHeads = Alloc(1, heads, n, headDim);
            OHeads = Alloc(1, heads, n, headDim);
            GateUp = Alloc(n, 2 * inter);
            Act = Alloc(n, inter);
        }

        public Tensor X { get; }
        public Tensor Normed { get; }
        public Tensor Branch { get; }
        public Tensor Qkv { get; }
        public Tensor Q { get; }
        public Tensor K { get; }
        public Tensor V { get; }
        public Tensor O { get; }
        public Tensor QHeads { get; }
        public Tensor KHeads { get; }
        public Tensor VHeads { get; }
        public Tensor OHeads { get; }
        public Tensor GateUp { get; }
        public Tensor Act { get; }

        public void Dispose()
        {
            foreach (Tensor tensor in new[] { X, Normed, Branch, Qkv, Q, K, V, O, QHeads, KHeads, VHeads, OHeads, GateUp, Act }) tensor.Dispose();
        }

        private static Tensor Alloc(params long[] shape) => new(new TensorShape(shape), DType.F32);
    }
}
