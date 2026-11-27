using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Host reference of the Hy4 Flash vision tower and projector (the HY4-VL tower of transformers PR 50903 at <c>eb1d5db</c>, checkpoint prefix <c>vit.</c>): a patch convolution plus a learned
/// position grid interpolated to the image, pre-LayerNorm blocks of full bidirectional attention and a GELU MLP, then the merger that norms, folds each 2x2 patch window into one token and
/// appends the learned newline token after every merged row.</summary>
/// <remarks>The tower has no rotary embedding; its config fields for one are unused by the reference. The position grid is interpolated bilinearly with the reference <c>scale_factor</c> of
/// <c>(size + 0.1) / edge</c> (align_corners false), which shifts the sampling grid slightly off a plain resize. One image at a time, row-major patches, channel-major inside a patch.</remarks>
public sealed class Hy4FlashVisionTower : IDisposable
{
    private readonly IBackend _backend;
    private readonly Hy4FlashVisionConfig _cfg;
    private readonly List<Tensor> _owned = [];
    private readonly Tensor _patchW, _patchB, _positions, _mergeNorm, _mergeW1, _mergeB1, _mergeW2, _mergeB2, _newline;
    private readonly Block[] _blocks;
    private int _disposed;

    /// <param name="backend">Provides linear, norm, attention and activation ops.</param>
    /// <param name="config">The model config; supplies the vision shape.</param>
    /// <param name="source">Reads the <c>vit.*</c> tensors.</param>
    public Hy4FlashVisionTower(IBackend backend, Hy4FlashVisionConfig config, IModelWeightSource source)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(source);
        _backend = backend;
        _cfg = config;
        int d = config.HiddenSize, m = config.SpatialMergeSize, p = config.PatchSize;
        _blocks = new Block[config.NumHiddenLayers];
        try
        {
            _patchW = Flat(source.Read("vit.embeddings.patch_embedding.weight"), d, 3 * p * p);
            _patchB = F32(source, "vit.embeddings.patch_embedding.bias");
            _positions = F32(source, "vit.embeddings.position_embedding.weight");
            for (int i = 0; i < _blocks.Length; i++)
            {
                string b = $"vit.layers.{i}.";
                _blocks[i] = new Block(
                    F32(source, b + "input_layernorm.weight"), F32(source, b + "input_layernorm.bias"),
                    Own(source.Read(b + "self_attn.q_proj.weight")), F32(source, b + "self_attn.q_proj.bias"),
                    Own(source.Read(b + "self_attn.k_proj.weight")), F32(source, b + "self_attn.k_proj.bias"),
                    Own(source.Read(b + "self_attn.v_proj.weight")), F32(source, b + "self_attn.v_proj.bias"),
                    Own(source.Read(b + "self_attn.o_proj.weight")), F32(source, b + "self_attn.o_proj.bias"),
                    F32(source, b + "post_attention_layernorm.weight"), F32(source, b + "post_attention_layernorm.bias"),
                    Own(source.Read(b + "mlp.dense_h_to_4h.weight")), F32(source, b + "mlp.dense_h_to_4h.bias"),
                    Own(source.Read(b + "mlp.dense_4h_to_h.weight")), F32(source, b + "mlp.dense_4h_to_h.bias"));
            }
            _mergeNorm = F32(source, "vit.perceive.norm.weight");
            _mergeW1 = Flat(source.Read("vit.perceive.proj.0.weight"), d * m * m, d * m * m);
            _mergeB1 = F32(source, "vit.perceive.proj.0.bias");
            _mergeW2 = Flat(source.Read("vit.perceive.proj.2.weight"), config.OutHiddenSize, d * m * m);
            _mergeB2 = F32(source, "vit.perceive.proj.2.bias");
            _newline = F32(source, "vit.perceive.image_newline");
            // The position lookup skips a leading row (1 + y * edge + x), so the table must hold PositionRows rows; a checkpoint with another max_image_size fails here, not inside a read.
            RequireShape(_positions, "vit.embeddings.position_embedding.weight", config.PositionRows, d);
            RequireShape(_newline, "vit.perceive.image_newline", config.OutHiddenSize);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private static void RequireShape(Tensor t, string name, params long[] dims)
    {
        bool ok = t.Shape.Rank == dims.Length;
        for (int i = 0; ok && i < dims.Length; i++) ok = t.Shape[i] == dims[i];
        if (!ok)
            throw new InvalidDataException($"{name} must be [{string.Join(", ", dims)}] but the checkpoint holds [{string.Join(", ", Enumerable.Range(0, t.Shape.Rank).Select(i => t.Shape[i]))}].");
    }

    /// <summary>Tokens a <paramref name="gridHeight"/> by <paramref name="gridWidth"/> patch image yields: merged rows of merged columns plus one newline each.</summary>
    public int TokenCount(int gridHeight, int gridWidth) => gridHeight / _cfg.SpatialMergeSize * (gridWidth / _cfg.SpatialMergeSize + 1);

    /// <summary>Runs one image and returns its projected tokens, <c>[TokenCount, OutHiddenSize]</c>.</summary>
    /// <param name="patches">Flattened patches, row-major grid order, <c>[gridHeight * gridWidth, 3 * patch * patch]</c>, channel-major inside a patch.</param>
    public float[] Forward(ReadOnlySpan<float> patches, int gridHeight, int gridWidth)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int m = _cfg.SpatialMergeSize, p = _cfg.PatchSize, d = _cfg.HiddenSize, heads = _cfg.NumAttentionHeads, hd = d / heads, n = checked(gridHeight * gridWidth);
        if (gridHeight < 1 || gridWidth < 1 || gridHeight % m != 0 || gridWidth % m != 0) throw new ArgumentException($"The patch grid must be a positive multiple of {m} on each side.");
        if (gridHeight > _cfg.PositionEdge || gridWidth > _cfg.PositionEdge) throw new ArgumentException($"The patch grid exceeds the {_cfg.PositionEdge}-patch position grid.");
        if (patches.Length != (long)n * 3 * p * p) throw new ArgumentException($"Expected {n} patches of {3 * p * p} values.", nameof(patches));
        float eps = (float)_cfg.RmsNormEps;

        using Tensor input = HostMath.Tensor(patches, n, 3 * p * p);
        using Tensor x = Alloc(n, d), normed = Alloc(n, d), branch = Alloc(n, d), q = Alloc(n, d), k = Alloc(n, d), v = Alloc(n, d), o = Alloc(n, d);
        using Tensor qh = Alloc(1, heads, n, hd), kh = Alloc(1, heads, n, hd), vh = Alloc(1, heads, n, hd), oh = Alloc(1, heads, n, hd);
        using Tensor ff = Alloc(n, _cfg.IntermediateSize);
        _backend.Linear(x, input, _patchW, _patchB);
        AddPositions(x.AsSpan<float>(), gridHeight, gridWidth);

        foreach (Block b in _blocks)
        {
            _backend.LayerNorm(normed, x, b.Ln1W, b.Ln1B, eps);
            _backend.Linear(q, normed, b.Q, b.QB);
            _backend.Linear(k, normed, b.K, b.KB);
            _backend.Linear(v, normed, b.V, b.VB);
            _backend.Permute0213(qh, q, n, heads, hd);
            _backend.Permute0213(kh, k, n, heads, hd);
            _backend.Permute0213(vh, v, n, heads, hd);
            _backend.ScaledDotProductAttention(oh, qh, kh, vh, null, 1f / MathF.Sqrt(hd));
            _backend.Permute0213(o, oh, heads, n, hd);
            _backend.Linear(branch, o, b.O, b.OB);
            _backend.Add(x, x, branch);
            _backend.LayerNorm(normed, x, b.Ln2W, b.Ln2B, eps);
            _backend.Linear(ff, normed, b.Fc1, b.Fc1B);
            _backend.GeluErf(ff, ff);
            _backend.Linear(branch, ff, b.Fc2, b.Fc2B);
            _backend.Add(x, x, branch);
        }

        _backend.RmsNorm(normed, x, _mergeNorm, eps);
        return Merge(normed.AsReadOnlySpan<float>(), gridHeight, gridWidth);
    }

    // x += position grid resampled to (h, w): the reference interpolates the learned square grid bilinearly with scale_factor (size + 0.1) / edge, align_corners false.
    private void AddPositions(Span<float> x, int gridHeight, int gridWidth)
    {
        int edge = _cfg.PositionEdge, d = _cfg.HiddenSize;
        ReadOnlySpan<float> table = _positions.AsReadOnlySpan<float>();
        (int I0, int I1, float L)[] rows = Axis(gridHeight, edge), cols = Axis(gridWidth, edge);
        for (int y = 0; y < gridHeight; y++)
            for (int xx = 0; xx < gridWidth; xx++)
            {
                (int y0, int y1, float ly) = rows[y];
                (int x0, int x1, float lx) = cols[xx];
                for (int c = 0; c < d; c++)
                {
                    float v00 = table[(1 + y0 * edge + x0) * d + c], v01 = table[(1 + y0 * edge + x1) * d + c];
                    float v10 = table[(1 + y1 * edge + x0) * d + c], v11 = table[(1 + y1 * edge + x1) * d + c];
                    float top = v00 * (1f - lx) + v01 * lx, bottom = v10 * (1f - lx) + v11 * lx;
                    x[(y * gridWidth + xx) * d + c] += top * (1f - ly) + bottom * ly;
                }
            }
    }

    private static (int I0, int I1, float L)[] Axis(int size, int edge)
    {
        float scale = (size + 0.1f) / edge;
        (int, int, float)[] axis = new (int, int, float)[size];
        for (int i = 0; i < size; i++)
        {
            float real = MathF.Max((i + 0.5f) / scale - 0.5f, 0f);
            int i0 = (int)real, i1 = i0 + (i0 < edge - 1 ? 1 : 0);
            axis[i] = (i0, i1, real - i0);
        }
        return axis;
    }

    // Folds every 2x2 window of the normed grid into one vector (channel-major, then window row, then window column: the Conv2d weight order), projects, and appends the newline token to each merged row.
    private float[] Merge(ReadOnlySpan<float> normed, int gridHeight, int gridWidth)
    {
        int m = _cfg.SpatialMergeSize, d = _cfg.HiddenSize, rows = gridHeight / m, cols = gridWidth / m, folded = d * m * m, outDim = _cfg.OutHiddenSize;
        float[] windows = new float[rows * cols * folded];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                for (int ch = 0; ch < d; ch++)
                    for (int kh = 0; kh < m; kh++)
                        for (int kw = 0; kw < m; kw++)
                            windows[(r * cols + c) * folded + ch * m * m + kh * m + kw] = normed[((r * m + kh) * gridWidth + c * m + kw) * d + ch];
        using Tensor input = HostMath.Tensor(windows, rows * cols, folded);
        using Tensor hidden = Alloc(rows * cols, folded);
        using Tensor projected = Alloc(rows * cols, outDim);
        _backend.Linear(hidden, input, _mergeW1, _mergeB1);
        _backend.GeluErf(hidden, hidden);
        _backend.Linear(projected, hidden, _mergeW2, _mergeB2);
        ReadOnlySpan<float> merged = projected.AsReadOnlySpan<float>(), newline = _newline.AsReadOnlySpan<float>();
        float[] tokens = new float[TokenCount(gridHeight, gridWidth) * outDim];
        for (int r = 0; r < rows; r++)
        {
            merged.Slice(r * cols * outDim, cols * outDim).CopyTo(tokens.AsSpan(r * (cols + 1) * outDim, cols * outDim));
            newline.CopyTo(tokens.AsSpan((r * (cols + 1) + cols) * outDim, outDim));
        }
        return tokens;
    }

    private static Tensor Alloc(params long[] shape) => new(new TensorShape(shape), DType.F32);

    private Tensor Flat(Tensor t, long rows, long cols)
    {
        Tensor view = t.Reshape(new TensorShape(rows, cols));
        _owned.Add(t);
        _owned.Add(view);
        return view;
    }

    private Tensor F32(IModelWeightSource source, string name) => Own(WeightReads.ReadF32(source, name));

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }

    private sealed record Block(Tensor Ln1W, Tensor Ln1B, Tensor Q, Tensor QB, Tensor K, Tensor KB, Tensor V, Tensor VB, Tensor O, Tensor OB,
        Tensor Ln2W, Tensor Ln2B, Tensor Fc1, Tensor Fc1B, Tensor Fc2, Tensor Fc2B);
}
