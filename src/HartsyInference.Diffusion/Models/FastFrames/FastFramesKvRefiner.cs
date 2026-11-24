using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.FastFrames;

/// <summary>Shape of a FastFrames KV refinement network. <c>Dim</c>, <c>Heads</c> and <c>Layers</c> are the generator DiT's;
/// <c>AdapterDim</c> is the refiner's own width (768 in both released models).</summary>
public sealed record FastFramesKvRefinerConfig(int Dim, int Heads, int Layers, int AdapterDim = 768, int StemBlocks = 4,
    int StemHeads = 8, int StemMlpMultiplier = 4, int HeadMlpMultiplier = 2, int FrequencyDim = 256)
{
    /// <summary>Per-head width of the generator's keys.</summary>
    public int HeadDim => Dim / Heads;
}

/// <summary>The KV refinement network of FastFrames (Sec. 3.3 of arXiv 2611.23226; <c>fastframes/models/kv_refinement.py</c> in
/// ml-fastframes a65c592). After a frame's last denoising step the generator's cache slot holds <i>noisy</i> keys and values; this
/// network maps them, together with features of the frame's clean latent, to clean-context keys and values and overwrites the
/// slot, in place of a second clean pass of the whole generator. A shared stem turns the patch-embedded clean latent into
/// features <c>phi</c> (four pre-norm adaLN transformer blocks over the frame's tokens); one light head per generator layer
/// then computes <c>K' = renorm(K + dK)</c> (per-head norm preserved) and <c>V' = (V + dV) · exp(s_V)</c>.
/// <para>Weights load from the released safetensors names (<c>stem_in</c>, <c>stem_blocks.N.*</c>, <c>stem_norm</c>,
/// <c>t_embed.mlp.*</c>, <c>heads_grouped.gemms.0.*</c>, <c>heads_grouped.phi_proj.&lt;d&gt;.weight</c>); the per-layer head
/// weights are stored <c>[layers, in, out]</c> and are transposed once to the engine's <c>[out, in]</c> layout. All compute is
/// F32 over host-built weights; a half-precision weight path is a measured follow-up, not assumed here.</para></summary>
public sealed class FastFramesKvRefiner : IDisposable
{
    private const float LayerNormEps = 1e-5f;   // nn.LayerNorm default, used by the stem blocks, stem_norm and the heads
    private const float RenormEps = 1e-6f;

    private readonly FastFramesKvRefinerConfig _config;
    private readonly List<Tensor> _owned = [];
    private Tensor _tEmbed0W = null!, _tEmbed0B = null!, _tEmbed2W = null!, _tEmbed2B = null!;
    private Tensor _stemInW = null!, _stemInB = null!, _stemNormW = null!, _stemNormB = null!, _phiProjW = null!;
    private StemBlock[] _stem = [];
    private HeadLayer[] _heads = [];
    private int _disposed;
    private bool _loaded;

    /// <summary>A refiner of the given shape; call <see cref="LoadWeights"/> before use.</summary>
    public FastFramesKvRefiner(FastFramesKvRefinerConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Dim != config.Heads * config.HeadDim || config.AdapterDim % config.StemHeads != 0)
            throw new ArgumentException("Dim must be Heads × HeadDim and AdapterDim a multiple of StemHeads.", nameof(config));
        _config = config;
    }

    /// <summary>The refiner's shape.</summary>
    public FastFramesKvRefinerConfig Config => _config;

    /// <summary>Reads a refiner's shape from its checkpoint: the adapter width from <c>stem_in.weight</c> (<c>[d_a, dim]</c>), the
    /// layer count from the stacked head weights, <c>dim</c>-dependent head count from <paramref name="heads"/>.</summary>
    public static FastFramesKvRefinerConfig DetectConfig(IReadOnlyDictionary<string, Tensor> weights, int heads)
    {
        Tensor stemIn = weights["stem_in.weight"];
        Tensor dK = weights["heads_grouped.gemms.0.dK_w"];
        int stemBlocks = 0;
        while (weights.ContainsKey($"stem_blocks.{stemBlocks}.qkv.weight"))
            stemBlocks++;
        return new FastFramesKvRefinerConfig(Dim: (int)stemIn.Shape[1], Heads: heads, Layers: (int)dK.Shape[0],
            AdapterDim: (int)stemIn.Shape[0], StemBlocks: stemBlocks);
    }

    /// <summary>Every tensor name this refiner reads and its checkpoint shape, so a checkpoint can be checked against the loader
    /// (and the loader against the released files) without loading weights.</summary>
    public static IReadOnlyDictionary<string, long[]> ExpectedShapes(FastFramesKvRefinerConfig c)
    {
        int dA = c.AdapterDim, dim = c.Dim, layers = c.Layers, h = dA * c.HeadMlpMultiplier, m = dA * c.StemMlpMultiplier;
        Dictionary<string, long[]> shapes = new()
        {
            ["t_embed.mlp.0.weight"] = [dA, c.FrequencyDim], ["t_embed.mlp.0.bias"] = [dA],
            ["t_embed.mlp.2.weight"] = [dA, dA], ["t_embed.mlp.2.bias"] = [dA],
            ["stem_in.weight"] = [dA, dim], ["stem_in.bias"] = [dA],
            ["stem_norm.weight"] = [dA], ["stem_norm.bias"] = [dA],
            [$"heads_grouped.phi_proj.{dA}.weight"] = [dA, dA],
        };
        for (int b = 0; b < c.StemBlocks; b++)
        {
            string p = $"stem_blocks.{b}.";
            shapes[p + "qkv.weight"] = [3 * dA, dA]; shapes[p + "qkv.bias"] = [3 * dA];
            shapes[p + "proj.weight"] = [dA, dA]; shapes[p + "proj.bias"] = [dA];
            shapes[p + "mlp.0.weight"] = [m, dA]; shapes[p + "mlp.0.bias"] = [m];
            shapes[p + "mlp.2.weight"] = [dA, m]; shapes[p + "mlp.2.bias"] = [dA];
            shapes[p + "mod.lin.weight"] = [6 * dA, dA]; shapes[p + "mod.lin.bias"] = [6 * dA];
        }
        const string g = "heads_grouped.gemms.0.";
        foreach ((string name, long[] shape) in new (string, long[])[]
        {
            ("proj_k_w", [layers, dim, dA]), ("proj_k_b", [layers, 1, dA]), ("proj_v_w", [layers, dim, dA]), ("proj_v_b", [layers, 1, dA]),
            ("fuse_w", [layers, 3 * dA, dA]), ("fuse_b", [layers, 1, dA]), ("mod_w", [layers, dA, 2 * dA]), ("mod_b", [layers, 1, 2 * dA]),
            ("m0_w", [layers, dA, h]), ("m0_b", [layers, 1, h]), ("m2_w", [layers, h, dA]), ("m2_b", [layers, 1, dA]),
            ("dK_w", [layers, dA, dim]), ("dK_b", [layers, 1, dim]), ("dV_w", [layers, dA, dim]), ("dV_b", [layers, 1, dim]),
            ("v_logscale", [layers, 1, dim]),
        })
        {
            shapes[g + name] = shape;
        }
        return shapes;
    }

    /// <summary>Loads and repacks the weights. The tensors in <paramref name="weights"/> are only read.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_loaded)
            throw new InvalidOperationException("The refiner's weights are already loaded; a refiner loads once.");
        int dA = _config.AdapterDim, dim = _config.Dim, layers = _config.Layers;
        _tEmbed0W = Own(F32(weights, "t_embed.mlp.0.weight")); _tEmbed0B = Own(F32(weights, "t_embed.mlp.0.bias"));
        _tEmbed2W = Own(F32(weights, "t_embed.mlp.2.weight")); _tEmbed2B = Own(F32(weights, "t_embed.mlp.2.bias"));
        _stemInW = Own(F32(weights, "stem_in.weight")); _stemInB = Own(F32(weights, "stem_in.bias"));
        _stemNormW = Own(F32(weights, "stem_norm.weight")); _stemNormB = Own(F32(weights, "stem_norm.bias"));
        _phiProjW = Own(F32(weights, $"heads_grouped.phi_proj.{dA}.weight"));
        if (_stemInW.Shape[0] != dA || _stemInW.Shape[1] != dim)
            throw new ArgumentException($"stem_in.weight is {_stemInW.Shape}; expected [{dA}, {dim}].");

        _stem = new StemBlock[_config.StemBlocks];
        for (int i = 0; i < _stem.Length; i++)
        {
            string p = $"stem_blocks.{i}.";
            _stem[i] = new StemBlock(Own(F32(weights, p + "qkv.weight")), Own(F32(weights, p + "qkv.bias")),
                Own(F32(weights, p + "proj.weight")), Own(F32(weights, p + "proj.bias")),
                Own(F32(weights, p + "mlp.0.weight")), Own(F32(weights, p + "mlp.0.bias")),
                Own(F32(weights, p + "mlp.2.weight")), Own(F32(weights, p + "mlp.2.bias")),
                Own(F32(weights, p + "mod.lin.weight")), Own(F32(weights, p + "mod.lin.bias")));
        }

        const string g = "heads_grouped.gemms.0.";
        _heads = new HeadLayer[layers];
        for (int l = 0; l < layers; l++)
        {
            _heads[l] = new HeadLayer(
                ProjK: Own(FastFramesOps.TransposedLayer(weights[g + "proj_k_w"], l)), ProjKB: Own(FastFramesOps.LayerVector(weights[g + "proj_k_b"], l)),
                ProjV: Own(FastFramesOps.TransposedLayer(weights[g + "proj_v_w"], l)), ProjVB: Own(FastFramesOps.LayerVector(weights[g + "proj_v_b"], l)),
                Fuse: Own(FastFramesOps.TransposedLayer(weights[g + "fuse_w"], l)), FuseB: Own(FastFramesOps.LayerVector(weights[g + "fuse_b"], l)),
                Mod: Own(FastFramesOps.TransposedLayer(weights[g + "mod_w"], l)), ModB: Own(FastFramesOps.LayerVector(weights[g + "mod_b"], l)),
                M0: Own(FastFramesOps.TransposedLayer(weights[g + "m0_w"], l)), M0B: Own(FastFramesOps.LayerVector(weights[g + "m0_b"], l)),
                M2: Own(FastFramesOps.TransposedLayer(weights[g + "m2_w"], l)), M2B: Own(FastFramesOps.LayerVector(weights[g + "m2_b"], l)),
                DK: Own(FastFramesOps.TransposedLayer(weights[g + "dK_w"], l)), DKB: Own(FastFramesOps.LayerVector(weights[g + "dK_b"], l)),
                DV: Own(FastFramesOps.TransposedLayer(weights[g + "dV_w"], l)), DVB: Own(FastFramesOps.LayerVector(weights[g + "dV_b"], l)),
                VScale: Own(ExpVector(weights[g + "v_logscale"], l)));
        }
        _loaded = true;
    }

    /// <summary>The per-run conditioning of a refinement at timestep <paramref name="timestep"/> (833.33 in the released
    /// models): the timestep embedding and, for every layer, the head's modulation vectors. The caller disposes it.</summary>
    public Conditioning Prepare(IBackend backend, float timestep)
    {
        RequireLoaded();
        int dA = _config.AdapterDim;
        using Tensor sinusoid = FastFramesOps.SinusoidalEmbedding(timestep, _config.FrequencyDim);
        using Tensor h0 = FastFramesOps.Linear(backend, sinusoid, _tEmbed0W, _tEmbed0B);
        using Tensor h1 = FastFramesOps.Silu(backend, h0);
        Tensor emb = FastFramesOps.Linear(backend, h1, _tEmbed2W, _tEmbed2B);
        Tensor[] shifts = new Tensor[_heads.Length];
        Tensor[] scales = new Tensor[_heads.Length];
        using Tensor siluEmb = FastFramesOps.Silu(backend, emb);
        for (int l = 0; l < _heads.Length; l++)
        {
            using Tensor modulation = FastFramesOps.Linear(backend, siluEmb, _heads[l].Mod, _heads[l].ModB);   // [1, 2·dA]
            ReadOnlySpan<float> row = modulation.AsReadOnlySpan<float>();
            shifts[l] = FastFramesOps.Vector(row[..dA]);
            scales[l] = FastFramesOps.OnePlus(row[dA..]);
        }
        return new Conditioning(emb, shifts, scales);
    }

    /// <summary>The stem features of a frame: <paramref name="stemTokens"/> is the clean latent patch-embedded by the generator,
    /// <c>[S, dim]</c>; the result is <c>[S, d_a]</c> after the stem's final norm; every layer's head consumes it through the shared <c>phi_proj</c>.</summary>
    public Tensor ComputePhi(IBackend backend, Tensor stemTokens, Conditioning conditioning)
    {
        RequireLoaded();
        Tensor x = FastFramesOps.Linear(backend, stemTokens, _stemInW, _stemInB);
        foreach (StemBlock block in _stem)
        {
            Tensor next = RunStemBlock(backend, block, x, conditioning.Embedding);
            x.Dispose();
            x = next;
        }
        Tensor phi = FastFramesOps.LayerNormAffine(backend, x, _stemNormW, _stemNormB, LayerNormEps);
        x.Dispose();
        return phi;
    }

    /// <summary>The shared <c>phi_proj</c> of the stem features <paramref name="phi"/> (<see cref="ComputePhi"/>'s result),
    /// <c>[S, d_a]</c>; every layer's head reads it, so it is computed once per frame. The caller disposes it.</summary>
    public Tensor ProjectPhi(IBackend backend, Tensor phi)
    {
        RequireLoaded();
        return FastFramesOps.Linear(backend, phi, _phiProjW, null);
    }

    /// <summary>Refines layer <paramref name="layer"/>'s noisy keys and values (<c>[S, dim]</c> F32, as the generator's cache
    /// holds them after the frame's last step) into clean-context ones. The inputs are only read; the caller disposes both
    /// results and writes them back into the cache slot, converting to the slot's dtype.</summary>
    public (Tensor K, Tensor V) RepairLayer(IBackend backend, int layer, Tensor noisyK, Tensor noisyV, Tensor phiV, Conditioning conditioning)
    {
        RequireLoaded();
        if (layer < 0 || layer >= _heads.Length)
            throw new ArgumentOutOfRangeException(nameof(layer), layer, $"The refiner has {_heads.Length} layers.");
        return RunHead(backend, _heads[layer], noisyK, noisyV, phiV, conditioning, layer);
    }

    private (Tensor K, Tensor V) RunHead(IBackend backend, HeadLayer head, Tensor noisyK, Tensor noisyV, Tensor phiV,
        Conditioning conditioning, int layer)
    {
        using Tensor pk = FastFramesOps.Linear(backend, noisyK, head.ProjK, head.ProjKB);
        using Tensor pv = FastFramesOps.Linear(backend, noisyV, head.ProjV, head.ProjVB);
        int tokens = (int)noisyK.Shape[0], dA = _config.AdapterDim;
        using Tensor joined = new(new TensorShape(tokens, 3 * dA), DType.F32);
        backend.Concat(joined, [pk, pv, phiV], 1);
        using Tensor fused = FastFramesOps.Linear(backend, joined, head.Fuse, head.FuseB);
        using Tensor modulated = FastFramesOps.Modulate(backend, fused, conditioning.Shifts[layer], conditioning.ScalesPlusOne[layer], LayerNormEps);
        using Tensor m0 = FastFramesOps.Linear(backend, modulated, head.M0, head.M0B);
        using Tensor act = FastFramesOps.GeluTanh(backend, m0);
        using Tensor h = FastFramesOps.Linear(backend, act, head.M2, head.M2B);

        using Tensor dK = FastFramesOps.Linear(backend, h, head.DK, head.DKB);
        using Tensor kSum = FastFramesOps.Add(backend, noisyK, dK);
        Tensor refinedK = RenormPerHead(kSum, noisyK);

        using Tensor dV = FastFramesOps.Linear(backend, h, head.DV, head.DVB);
        using Tensor vSum = FastFramesOps.Add(backend, noisyV, dV);
        Tensor refinedV = new(vSum.Shape, DType.F32);
        backend.AffineBroadcastLastDim(refinedV, vSum, head.VScale, null);
        return (refinedK, refinedV);
    }

    /// <summary>Scales each head's <c>HeadDim</c> slice of <paramref name="x"/> to the norm of the same slice of
    /// <paramref name="reference"/>: <c>x · (‖ref‖ / (‖x‖ + 1e-6))</c>. No backend operation covers this exact form, so it runs on
    /// the host.</summary>
    private Tensor RenormPerHead(Tensor x, Tensor reference)
    {
        int tokens = (int)x.Shape[0], heads = _config.Heads, headDim = _config.HeadDim;
        Tensor output = new(x.Shape, DType.F32);
        ReadOnlySpan<float> xs = x.AsReadOnlySpan<float>(), rs = reference.AsReadOnlySpan<float>();
        Span<float> os = output.AsSpan<float>();
        for (int t = 0; t < tokens; t++)
        {
            for (int h = 0; h < heads; h++)
            {
                int offset = (t * heads + h) * headDim;
                double xNorm = 0, rNorm = 0;
                for (int d = 0; d < headDim; d++)
                {
                    xNorm += (double)xs[offset + d] * xs[offset + d];
                    rNorm += (double)rs[offset + d] * rs[offset + d];
                }
                float scale = (float)(Math.Sqrt(rNorm) / (Math.Sqrt(xNorm) + RenormEps));
                for (int d = 0; d < headDim; d++)
                    os[offset + d] = xs[offset + d] * scale;
            }
        }
        return output;
    }

    private Tensor RunStemBlock(IBackend backend, StemBlock block, Tensor x, Tensor embedding)
    {
        int dA = _config.AdapterDim, heads = _config.StemHeads, headDim = dA / heads, tokens = (int)x.Shape[0];
        using Tensor siluEmb = FastFramesOps.Silu(backend, embedding);
        using Tensor mod = FastFramesOps.Linear(backend, siluEmb, block.ModW, block.ModB);   // [1, 6·dA]
        ReadOnlySpan<float> row = mod.AsReadOnlySpan<float>();
        using Tensor shift1 = FastFramesOps.Vector(row[..dA]);
        using Tensor scale1 = FastFramesOps.OnePlus(row.Slice(dA, dA));
        using Tensor gate1 = FastFramesOps.Vector(row.Slice(2 * dA, dA));
        using Tensor shift2 = FastFramesOps.Vector(row.Slice(3 * dA, dA));
        using Tensor scale2 = FastFramesOps.OnePlus(row.Slice(4 * dA, dA));
        using Tensor gate2 = FastFramesOps.Vector(row.Slice(5 * dA, dA));

        using Tensor h = FastFramesOps.Modulate(backend, x, shift1, scale1, LayerNormEps);
        using Tensor qkv = FastFramesOps.Linear(backend, h, block.QkvW, block.QkvB);   // [S, 3·dA], q | k | v per token
        Tensor[] parts = [new(new TensorShape(tokens, dA), DType.F32), new(new TensorShape(tokens, dA), DType.F32), new(new TensorShape(tokens, dA), DType.F32)];
        backend.Split(parts, qkv, 1);
        using Tensor q = HeadMajor(backend, parts[0], tokens, heads, headDim);
        using Tensor k = HeadMajor(backend, parts[1], tokens, heads, headDim);
        using Tensor v = HeadMajor(backend, parts[2], tokens, heads, headDim);
        using Tensor attended = new(new TensorShape(1, heads, tokens, headDim), DType.F32);
        backend.ScaledDotProductAttention(attended, q, k, v, null, 1f / MathF.Sqrt(headDim));
        using Tensor merged = new(new TensorShape(1, tokens, heads, headDim), DType.F32);
        backend.Permute0213(merged, attended, heads, tokens, headDim);
        using Tensor flat = merged.Reshape(new TensorShape(tokens, dA));
        using Tensor projected = FastFramesOps.Linear(backend, flat, block.ProjW, block.ProjB);
        using Tensor afterAttention = FastFramesOps.GatedAdd(backend, x, projected, gate1);

        using Tensor h2 = FastFramesOps.Modulate(backend, afterAttention, shift2, scale2, LayerNormEps);
        using Tensor m0 = FastFramesOps.Linear(backend, h2, block.Mlp0W, block.Mlp0B);
        using Tensor act = FastFramesOps.GeluTanh(backend, m0);
        using Tensor m2 = FastFramesOps.Linear(backend, act, block.Mlp2W, block.Mlp2B);
        return FastFramesOps.GatedAdd(backend, afterAttention, m2, gate2);
    }

    /// <summary><c>[S, dA]</c> (consumed) to head-major <c>[1, heads, S, headDim]</c>.</summary>
    private static Tensor HeadMajor(IBackend backend, Tensor columns, int tokens, int heads, int headDim)
    {
        using (columns)
        {
            using Tensor shaped = columns.Reshape(new TensorShape(1, tokens, heads, headDim));
            Tensor output = new(new TensorShape(1, heads, tokens, headDim), DType.F32);
            backend.Permute0213(output, shaped, tokens, heads, headDim);
            return output;
        }
    }

    private static Tensor ExpVector(Tensor stacked, int layer)
    {
        using Tensor logScale = FastFramesOps.LayerVector(stacked, layer);
        Span<float> values = logScale.AsSpan<float>();
        Tensor output = new(logScale.Shape, DType.F32);
        Span<float> target = output.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            target[i] = MathF.Exp(values[i]);
        return output;
    }

    private static Tensor F32(IReadOnlyDictionary<string, Tensor> weights, string key) =>
        FastFramesOps.CopyF32(weights[key], weights[key].Shape);

    private void RequireLoaded()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_loaded)
            throw new InvalidOperationException("The refiner has no weights yet; call LoadWeights first.");
    }

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor t in _owned)
            t.Dispose();
        _owned.Clear();
    }

    /// <summary>The timestep-dependent state of one run: the embedding and each layer's head modulation.</summary>
    public sealed class Conditioning(Tensor embedding, Tensor[] shifts, Tensor[] scalesPlusOne) : IDisposable
    {
        /// <summary>The timestep embedding <c>[1, d_a]</c>.</summary>
        public Tensor Embedding { get; } = embedding;

        /// <summary>Each layer's shift vector <c>[d_a]</c>.</summary>
        public Tensor[] Shifts { get; } = shifts;

        /// <summary>Each layer's <c>1 + scale</c> vector <c>[d_a]</c>.</summary>
        public Tensor[] ScalesPlusOne { get; } = scalesPlusOne;

        private bool _disposed;

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            Embedding.Dispose();
            foreach (Tensor t in Shifts)
                t.Dispose();
            foreach (Tensor t in ScalesPlusOne)
                t.Dispose();
        }
    }

    private sealed record StemBlock(Tensor QkvW, Tensor QkvB, Tensor ProjW, Tensor ProjB, Tensor Mlp0W, Tensor Mlp0B,
        Tensor Mlp2W, Tensor Mlp2B, Tensor ModW, Tensor ModB);

    private sealed record HeadLayer(Tensor ProjK, Tensor ProjKB, Tensor ProjV, Tensor ProjVB, Tensor Fuse, Tensor FuseB,
        Tensor Mod, Tensor ModB, Tensor M0, Tensor M0B, Tensor M2, Tensor M2B, Tensor DK, Tensor DKB, Tensor DV, Tensor DVB,
        Tensor VScale);
}
