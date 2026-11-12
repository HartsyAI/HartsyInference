using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>Shape of the X-Part DiT (<c>PartFormerDITPlain</c>, <c>infer.yaml</c>): 21 blocks of width 2048 and 16 heads over 1024 latent tokens of 64 channels per part, a mixture of
/// experts (8 experts, top 2) in the last 6 blocks, U-skips after the middle, a learned embedding per part id (50 ids).</summary>
public sealed record XPartDitConfig(int Width = 2048, int Depth = 21, int Heads = 16, int LatentChannels = 64, int MoeBlocks = 6, int Experts = 8, int TopK = 2, int ValidParts = 50);

/// <summary>The X-Part DiT: all parts of one object are denoised together as a batch of <c>K</c> latent sets. Each block runs self-attention over its own part (even layers) or over the
/// tokens of every part at once (odd layers, upstream's <c>LocalGlobalProcessor(use_global=True)</c>), then cross-attends to the object tokens and to the part-in-box tokens, which
/// are summed with the hidden state. The velocity <c>[K, N, 64]</c> comes from the final norm and projection after the time token is dropped.</summary>
/// <remarks>The timestep is the flow sigma itself (upstream feeds <c>t / 1000</c> and embeds it with no time factor), as in the 2.1 plain DiT. The part ids are the caller's:
/// upstream draws <c>torch.randperm(50)[:K]</c> at every call, which is not reproducible, so a caller passes the ids it wants. Activations are F32; weights widen to F32 on load.</remarks>
public sealed class XPartDit : IDisposable
{
    private readonly XPartDitConfig _config;
    private readonly Hunyuan3DPlainBlock[] _blocks;
    private readonly List<Tensor> _owned = [];
    private readonly HashSet<Tensor> _callerOwned = new(ReferenceEqualityComparer.Instance);
    private Tensor _xW = null!, _xB = null!, _t1W = null!, _t1B = null!, _t2W = null!, _t2B = null!, _finalNormW = null!, _finalNormB = null!, _finalW = null!, _finalB = null!, _partEmbed = null!;
    private bool _disposed;

    /// <summary>Creates a DiT with no weights; call <see cref="LoadWeights"/>.</summary>
    public XPartDit(XPartDitConfig? config = null)
    {
        _config = config ?? new XPartDitConfig();
        _blocks = new Hunyuan3DPlainBlock[_config.Depth];
        for (int layer = 0; layer < _blocks.Length; layer++)
        {
            bool skip = layer > _config.Depth / 2, moe = _config.Depth - layer <= _config.MoeBlocks;
            _blocks[layer] = new Hunyuan3DPlainBlock(_config.Width, _config.Heads, skip, moe, _config.Experts, _config.TopK);
        }
    }

    /// <summary>Called with the layer index and the hidden state <c>[K, N + 1, W]</c> after each block (a parity probe); the tensor is the block's output and must not be disposed.</summary>
    public Action<int, Tensor>? BlockProbe { get; set; }

    /// <summary>Loads the <c>model.safetensors</c> tensors (<paramref name="prefix"/> <c>""</c>). BF16 and F16 widen to F32; the caller keeps ownership of <paramref name="weights"/>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        LoadEdgeWeights(weights, prefix);
        string p = prefix.Length == 0 ? string.Empty : prefix + ".";
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].LoadWeights(weights, $"{p}blocks.{i}", halfWeights: false);
            if (!_blocks[i].HasSecondCross)
                throw new InvalidDataException($"X-Part block {i} has no attn2_2 second cross-attention.");
        }
    }

    // The embedding, the time embedder, the part embedding and the final layer: everything outside the blocks.
    internal void LoadEdgeWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(weights);
        string p = prefix.Length == 0 ? string.Empty : prefix + ".";
        int w = _config.Width;
        foreach (Tensor t in weights.Values)
            _callerOwned.Add(t);
        _xW = Own(weights, $"{p}x_embedder.weight", w, _config.LatentChannels); _xB = Own(weights, $"{p}x_embedder.bias", w);
        _t1W = Own(weights, $"{p}t_embedder.mlp.0.weight", 4 * w, w); _t1B = Own(weights, $"{p}t_embedder.mlp.0.bias", 4 * w);
        _t2W = Own(weights, $"{p}t_embedder.mlp.2.weight", w, 4 * w); _t2B = Own(weights, $"{p}t_embedder.mlp.2.bias", w);
        _partEmbed = Own(weights, $"{p}part_embed", _config.ValidParts, w);
        _finalNormW = Own(weights, $"{p}final_layer.norm_final.weight", w); _finalNormB = Own(weights, $"{p}final_layer.norm_final.bias", w);
        _finalW = Own(weights, $"{p}final_layer.linear.weight", _config.LatentChannels, w); _finalB = Own(weights, $"{p}final_layer.linear.bias", _config.LatentChannels);
    }

    /// <summary>One velocity prediction. <paramref name="latents"/> is <c>[K, N, 64]</c>, <paramref name="sigma"/> the flow time (<c>t / 1000</c>), <paramref name="objectContext"/> and
    /// <paramref name="partContext"/> are <c>[K, Nc, 1024]</c> (the object tokens, repeated per part, and each part's in-box tokens), <paramref name="partIds"/> one id below 50 per part.
    /// Returns <c>[K, N, 64]</c>; the caller disposes it. Nothing passed in is consumed.</summary>
    public Tensor Forward(IBackend backend, Tensor latents, float sigma, Tensor objectContext, Tensor partContext, ReadOnlySpan<int> partIds)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(backend);
        if (_xW is null)
            throw new InvalidOperationException("Call LoadWeights first.");
        int k = (int)latents.Shape[0], c = _config.LatentChannels;
        if (latents.Shape.Rank != 3 || latents.Shape[2] != c || partIds.Length != k || objectContext.Shape[0] != k || partContext.Shape[0] != k)
            throw new ArgumentException($"Latents {latents.Shape}, contexts {objectContext.Shape} and {partContext.Shape} and {partIds.Length} part ids do not describe one set of parts.");
        foreach (int id in partIds)
        {
            if ((uint)id >= (uint)_config.ValidParts)
                throw new ArgumentOutOfRangeException(nameof(partIds), $"Part id {id} is outside 0..{_config.ValidParts - 1}.");
        }

        Tensor x = Embed(backend, latents, sigma, partIds);
        Stack<Tensor> skips = new();
        int half = _config.Depth / 2;
        for (int layer = 0; layer < _blocks.Length; layer++)
        {
            Tensor? skip = layer > half ? skips.Pop() : null;
            bool global = (layer + 1) % 2 == 0;
            Tensor next = _blocks[layer].ForwardParts(backend, x, objectContext, partContext, skip, global);
            skip?.Dispose();
            if (layer < half)
            {
                Tensor copy = new(next.Shape, DType.F32);
                backend.CopyInto(copy, next);
                skips.Push(copy);
            }

            x = next;
            BlockProbe?.Invoke(layer, x);
        }

        Tensor velocity = Project(backend, x);
        x.Dispose();
        return velocity;
    }

    /// <summary>The final layer: LayerNorm, the time token dropped, then the linear to the latent channels. <paramref name="x"/> is <c>[K, N + 1, W]</c>; returns <c>[K, N, 64]</c>.</summary>
    internal Tensor Project(IBackend backend, Tensor x)
    {
        int k = (int)x.Shape[0], n = (int)x.Shape[1] - 1, w = _config.Width;
        using Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, _finalNormW, _finalNormB, 1e-6f);
        using Tensor body = new(new TensorShape(k, n, w), DType.F32);
        for (int part = 0; part < k; part++)
            normed.AsSpan<float>().Slice((part * (n + 1) + 1) * w, n * w).CopyTo(body.AsSpan<float>().Slice(part * n * w, n * w));
        Tensor velocity = new(new TensorShape(k, n, _config.LatentChannels), DType.F32);
        backend.Linear(velocity, body, _finalW, _finalB);
        return velocity;
    }

    /// <summary>The input of the first block: the time token followed by <c>x_embedder(latent) + part_embed[id]</c>, <c>[K, N + 1, W]</c>.</summary>
    internal Tensor Embed(IBackend backend, Tensor latents, float sigma, ReadOnlySpan<int> partIds)
    {
        int k = (int)latents.Shape[0], n = (int)latents.Shape[1], w = _config.Width;
        using Tensor sin = new(new TensorShape(1, w), DType.F32);
        TimestepEmbedding(sin.AsSpan<float>(), sigma, w);
        using Tensor t1 = new(new TensorShape(1, 4 * w), DType.F32);
        backend.Linear(t1, sin, _t1W, _t1B);
        using Tensor t1a = new(t1.Shape, DType.F32);
        backend.GeluErf(t1a, t1);
        using Tensor token = new(new TensorShape(1, 1, w), DType.F32);
        backend.Linear(token, t1a, _t2W, _t2B);

        using Tensor embedded = new(new TensorShape(k, n, w), DType.F32);
        backend.Linear(embedded, latents, _xW, _xB);
        Tensor x = new(new TensorShape(k, n + 1, w), DType.F32);
        Span<float> target = x.AsSpan<float>();
        ReadOnlySpan<float> body = embedded.AsSpan<float>(), time = token.AsSpan<float>(), parts = _partEmbed.AsSpan<float>();
        for (int part = 0; part < k; part++)
        {
            Span<float> rows = target.Slice(part * (n + 1) * w, (n + 1) * w);
            time.CopyTo(rows[..w]);
            ReadOnlySpan<float> embed = parts.Slice(partIds[part] * w, w), source = body.Slice(part * n * w, n * w);
            for (int row = 0; row < n; row++)
            {
                Span<float> dst = rows.Slice((row + 1) * w, w);
                ReadOnlySpan<float> src = source.Slice(row * w, w);
                for (int i = 0; i < w; i++)
                    dst[i] = src[i] + embed[i];
            }
        }

        return x;
    }

    /// <summary>Releases the converted weights and the blocks.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (Tensor t in _owned)
            t.Dispose();
        foreach (Hunyuan3DPlainBlock block in _blocks)
        {
            foreach (Tensor t in block.EnumerateWeights())
            {
                if (!_callerOwned.Contains(t))
                    t.Dispose();
            }
        }
    }

    // Timesteps(width): frequencies exp(-ln(10000) i / half), sines then cosines, scale 1.
    private static void TimestepEmbedding(Span<float> output, float timestep, int width)
    {
        int half = width / 2;
        float logMax = MathF.Log(10000f);
        for (int i = 0; i < half; i++)
        {
            float angle = timestep * MathF.Exp(-logMax * i / half);
            output[i] = MathF.Sin(angle);
            output[half + i] = MathF.Cos(angle);
        }
    }

    private Tensor Own(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape)
    {
        if (!weights.TryGetValue(key, out Tensor? source))
            throw new KeyNotFoundException($"The X-Part DiT tensor '{key}' is missing.");
        if (!source.Shape.Equals(new TensorShape(shape)))
            throw new InvalidDataException($"The X-Part DiT tensor '{key}' has shape {source.Shape}, expected {new TensorShape(shape)}.");
        Tensor widened = Hunyuan3DShapeVae.F32(source);
        if (ReferenceEquals(widened, source))
            return source;
        _owned.Add(widened);
        return widened;
    }
}
