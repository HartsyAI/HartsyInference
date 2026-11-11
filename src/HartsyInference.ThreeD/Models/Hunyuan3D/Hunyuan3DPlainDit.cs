using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Hunyuan3D-2.1 shape denoiser (<c>HunYuanDiTPlain</c>): the timestep becomes a prepended token, blocks after the middle take U-skips from the first half, and the last blocks are mixtures of experts.</summary>
public sealed unsafe class Hunyuan3DPlainDit : IHunyuan3DDenoiser
{
    private readonly Hunyuan3DConfig _cfg;
    private readonly Hunyuan3DPlainDitConfig _plain;
    private readonly Hunyuan3DPlainBlock[] _blocks;
    private bool _half;
    private Tensor? _xW, _xB, _t1W, _t1B, _t2W, _t2B, _finalNormW, _finalNormB, _finalW, _finalB;

    /// <summary>Test hook: called with each block's output.</summary>
    internal Action<int, Tensor>? BlockProbe { get; set; }

    /// <summary>Test hook: <see cref="Forward"/> returns the hidden state after this block instead of the velocity.</summary>
    internal int StopAfterBlock { get; set; } = int.MaxValue;

    /// <summary>Creates the model for a config read from a 2.1 checkpoint (<see cref="Hunyuan3DConfig.PlainDit"/> must be set).</summary>
    public Hunyuan3DPlainDit(Hunyuan3DConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        _cfg = cfg;
        _plain = cfg.PlainDit ?? throw new ArgumentException("The config is not a Hunyuan3D-2.1 plain DiT config.", nameof(cfg));
        _blocks = new Hunyuan3DPlainBlock[_plain.Depth];
        for (int layer = 0; layer < _plain.Depth; layer++)
        {
            bool skip = layer > _plain.Depth / 2, moe = _plain.Depth - layer <= _plain.MoeBlocks;
            _blocks[layer] = new Hunyuan3DPlainBlock(cfg.Width, cfg.NumHeads, skip, moe, _plain.Experts, _plain.TopK);
        }
    }

    /// <summary>Loads the <c>model.*</c> weights with the prefix removed; with <paramref name="halfWeights"/> the matrices stay F16 (for a backend that multiplies F16 weights).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "", bool halfWeights = false)
    {
        string p = prefix.Length == 0 ? string.Empty : prefix + ".";
        _half = halfWeights;
        _xW = F(w, $"{p}x_embedder.weight"); _xB = F(w, $"{p}x_embedder.bias");
        _t1W = F(w, $"{p}t_embedder.mlp.0.weight"); _t1B = F(w, $"{p}t_embedder.mlp.0.bias");
        _t2W = F(w, $"{p}t_embedder.mlp.2.weight"); _t2B = F(w, $"{p}t_embedder.mlp.2.bias");
        _finalNormW = F(w, $"{p}final_layer.norm_final.weight"); _finalNormB = F(w, $"{p}final_layer.norm_final.bias");
        _finalW = F(w, $"{p}final_layer.linear.weight"); _finalB = F(w, $"{p}final_layer.linear.bias");
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].LoadWeights(w, $"{p}blocks.{i}", halfWeights);
            if (_blocks[i].HiddenDim != _cfg.MlpDim)
                throw new UnsupportedModelException($"Hunyuan3D-2.1 block {i} has MLP width {_blocks[i].HiddenDim}, but the config (hidden_size x mlp_ratio) says {_cfg.MlpDim}.");
        }
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_xW, _xB, _t1W, _t1B, _t2W, _t2B, _finalNormW, _finalNormB, _finalW, _finalB];
        foreach (Tensor? t in head)
        {
            if (t is not null) yield return t;
        }

        foreach (Hunyuan3DPlainBlock block in _blocks)
        {
            foreach (Tensor t in block.EnumerateWeights()) yield return t;
        }
    }

    /// <inheritdoc/>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor cond, float timestep, float? guidance = null)
    {
        if (guidance.HasValue)
            throw new ArgumentException("Hunyuan3D-2.1 has no guidance embedding; run classifier-free guidance instead.", nameof(guidance));
        if (latent.Shape.Rank != 3 || latent.Shape[0] != 1 || cond.Shape[0] != 1)
            throw new ArgumentException($"Batch 1 only; got latent {latent.Shape} and cond {cond.Shape}.");
        int n = (int)latent.Shape[1], width = _cfg.Width;
        Tensor x = Embed(backend, latent, timestep);

        Stack<Tensor> skips = new();
        int half = _plain.Depth / 2;
        for (int layer = 0; layer < _blocks.Length; layer++)
        {
            Tensor? skip = layer > half ? skips.Pop() : null;
            Tensor next = _blocks[layer].Forward(backend, x, cond, skip);
            skip?.Dispose();
            if (layer < half)
            {
                // The block output is both the next block's input and a skip value: keep a copy for the skip.
                Tensor copy = new(next.Shape, DType.F32);
                backend.CopyInto(copy, next);
                skips.Push(copy);
            }

            x = next;
            BlockProbe?.Invoke(layer, x);
            if (layer == StopAfterBlock) return x;
        }

        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, _finalNormW!, _finalNormB!, 1e-6f);
        x.Dispose();
        Tensor body = new(new TensorShape(1, n, width), DType.F32);
        backend.SliceRows(body, normed, 1);
        normed.Dispose();
        Tensor velocity = new(new TensorShape(1, n, _cfg.LatentChannels), DType.F32);
        backend.Linear(velocity, body, _finalW!, _finalB!);
        body.Dispose();
        return velocity;
    }

    /// <summary>The timestep token prepended to the embedded latent: <c>[1, N + 1, W]</c>.</summary>
    internal Tensor Embed(IBackend backend, Tensor latent, float timestep)
    {
        int n = (int)latent.Shape[1], width = _cfg.Width;

        // Timestep token: sin/cos(t * freqs) -> Linear -> GELU -> Linear, prepended to the embedded latent.
        Tensor sin = new(new TensorShape(1, width), DType.F32);
        TimestepEmbedding(sin, timestep, width);
        Tensor t1 = new(new TensorShape(1, _t1W!.Shape[0]), DType.F32);
        backend.Linear(t1, sin, _t1W!, _t1B!);
        sin.Dispose();
        Tensor t1a = new(t1.Shape, DType.F32);
        backend.GeluErf(t1a, t1); // nn.GELU() in upstream's t_embedder: exact erf form
        t1.Dispose();
        Tensor token = new(new TensorShape(1, 1, width), DType.F32);
        backend.Linear(token, t1a, _t2W!, _t2B!);
        t1a.Dispose();

        Tensor embedded = new(new TensorShape(1, n, width), DType.F32);
        backend.Linear(embedded, latent, _xW!, _xB!);
        Tensor x = new(new TensorShape(1, n + 1, width), DType.F32);
        backend.Concat(x, [token, embedded], 1);
        token.Dispose();
        embedded.Dispose();
        return x;
    }

    // diffusers-style Timesteps(width): frequencies exp(-ln(10000) * i / half), sine block then cosine block, no extra scale.
    private static void TimestepEmbedding(Tensor output, float timestep, int width)
    {
        float* p = (float*)output.DataPointer;
        int half = width / 2;
        float logMax = MathF.Log(10000f);
        for (int i = 0; i < half; i++)
        {
            float angle = timestep * MathF.Exp(-logMax * i / half);
            p[i] = MathF.Sin(angle);
            p[half + i] = MathF.Cos(angle);
        }
    }

    private Tensor F(IReadOnlyDictionary<string, Tensor> w, string key) =>
        _half && w[key].DType == DType.F16 && w[key].Shape.Rank == 2 ? w[key] : Hunyuan3DShapeVae.F32(w[key]);
}
