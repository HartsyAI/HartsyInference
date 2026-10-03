using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>BigVGAN's <c>AMPBlock1</c> (HiFi-GAN ResBlock1 with anti-aliased periodic activations): three
/// dilated-conv branches per <paramref name="dilations"/>, each wrapped by its own paired bias-free
/// <see cref="AntiAliasedSnake"/> activation before and after the branch's two convs, residual-accumulated.</summary>
/// <remarks>Reuses <see cref="AntiAliasedSnake"/> directly (already shipped with CUDA/CPU/Vulkan kernels) for each
/// of the 6 activation sites rather than porting <c>LtxBigVganGenerator</c>'s private anti-alias math a second
/// time — only the top-level assembly shape is new.</remarks>
internal sealed unsafe class IndexTtsBigVganResBlock : IDisposable
{
    private readonly int _channels, _kernel;
    private readonly int[] _dilations;
    private readonly AntiAliasedSnake[] _activations;   // 6: interleaved acts1[0..2] / acts2[0..2]
    private Tensor?[] _convs1W = [], _convs1B = [];
    private Tensor?[] _convs2W = [], _convs2B = [];
    private int _disposed;

    public IndexTtsBigVganResBlock(int channels, int kernel, int[] dilations)
    {
        _channels = channels;
        _kernel = kernel;
        _dilations = dilations;
        _activations = new AntiAliasedSnake[2 * dilations.Length];
        for (int i = 0; i < _activations.Length; i++) _activations[i] = new AntiAliasedSnake(channels);
        _convs1W = new Tensor?[dilations.Length]; _convs1B = new Tensor?[dilations.Length];
        _convs2W = new Tensor?[dilations.Length]; _convs2B = new Tensor?[dilations.Length];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int i = 0; i < _dilations.Length; i++)
        {
            _convs1W[i] = WeightNormFusion.Compose(w, $"{prefix}.convs1.{i}");
            _convs1B[i] = EnsureF32(w[$"{prefix}.convs1.{i}.bias"]);
            _convs2W[i] = WeightNormFusion.Compose(w, $"{prefix}.convs2.{i}");
            _convs2B[i] = EnsureF32(w[$"{prefix}.convs2.{i}.bias"]);
        }
        for (int i = 0; i < _activations.Length; i++) _activations[i].LoadWeights(w, $"{prefix}.activations.{i}");
    }

    /// <summary>Runs over <c>[1, channels, T]</c>, returning a new tensor of the same shape.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        int t = (int)x.Shape[2];
        Tensor cur = new(x.Shape, DType.F32);
        backend.CopyTo(cur, x);

        for (int i = 0; i < _dilations.Length; i++)
        {
            int dilation = _dilations[i];
            int pad = (_kernel - 1) * dilation / 2;

            Tensor a1 = _activations[2 * i].Forward(backend, cur);
            Tensor c1 = new(new TensorShape(1, _channels, t), DType.F32);
            backend.Conv1d(c1, a1, _convs1W[i]!, _convs1B[i], stride: 1, padLeft: pad, padRight: pad, dilation: dilation, groups: 1);
            a1.Dispose();

            Tensor a2 = _activations[2 * i + 1].Forward(backend, c1);
            c1.Dispose();
            int pad2 = (_kernel - 1) / 2;
            Tensor c2 = new(new TensorShape(1, _channels, t), DType.F32);
            backend.Conv1d(c2, a2, _convs2W[i]!, _convs2B[i], stride: 1, padLeft: pad2, padRight: pad2, dilation: 1, groups: 1);
            a2.Dispose();

            Tensor next = new(cur.Shape, DType.F32);
            backend.Add(next, cur, c2);
            c2.Dispose();
            cur.Dispose();
            cur = next;
        }
        return cur;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < _dilations.Length; i++)
        {
            yield return _convs1W[i]!; yield return _convs1B[i]!;
            yield return _convs2W[i]!; yield return _convs2B[i]!;
        }
        foreach (AntiAliasedSnake act in _activations) foreach (Tensor t in act.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (AntiAliasedSnake act in _activations) act.Dispose();
        GC.SuppressFinalize(this);
    }
}
