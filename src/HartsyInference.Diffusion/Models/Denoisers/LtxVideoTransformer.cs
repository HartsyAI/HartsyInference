using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>LTX-Video DiT (<c>LTXVideoTransformer3DModel</c>), ported from diffusers. Single-stream over VAE-latent tokens (B=1, <c>[S, 128]</c>): <c>proj_in</c> → 28 blocks (self-attn+RoPE / cross-attn to T5 / FFN, AdaLN-Single) → final AdaLN + <c>proj_out</c>. Reuses <see cref="DiTUtils"/>, backend <c>RmsNorm</c>/<c>ScaledDotProductAttention</c>, and the existing T5 encoder (caption side). See <c>docs/Research/LTX_VIDEO_ARCHITECTURE.md</c>.</summary>
public sealed unsafe class LtxVideoTransformer : IDisposable
{
    private readonly LtxVideoConfig _config;
    private readonly LtxVideoBlock[] _blocks;
    private readonly LtxRope _rope;
    private int _disposed;

    private Tensor? _projInW, _projInB, _projOutW, _projOutB;
    private Tensor? _finalScaleShift;     // [2, inner]
    private Tensor? _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B;   // timestep_embedder
    private Tensor? _timeLinW, _timeLinB;  // → 6*inner
    private Tensor? _capW1, _capB1, _capW2, _capB2;   // caption projection

    public LtxVideoTransformer(LtxVideoConfig config)
    {
        _config = config;
        _blocks = new LtxVideoBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _blocks[i] = new LtxVideoBlock(config);
        _rope = new LtxRope(config.InnerDim, config.RopeTheta, config.RopeBaseNumFrames, config.RopeBaseHeight, config.RopeBaseWidth);
    }

    public LtxVideoConfig Config => _config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _projInW = w["proj_in.weight"]; w.TryGetValue("proj_in.bias", out _projInB);
        _projOutW = w["proj_out.weight"]; w.TryGetValue("proj_out.bias", out _projOutB);
        _finalScaleShift = LoadF32(w, "scale_shift_table");
        _timeEmb1W = w["time_embed.emb.timestep_embedder.linear_1.weight"]; w.TryGetValue("time_embed.emb.timestep_embedder.linear_1.bias", out _timeEmb1B);
        _timeEmb2W = w["time_embed.emb.timestep_embedder.linear_2.weight"]; w.TryGetValue("time_embed.emb.timestep_embedder.linear_2.bias", out _timeEmb2B);
        _timeLinW = w["time_embed.linear.weight"]; w.TryGetValue("time_embed.linear.bias", out _timeLinB);
        _capW1 = w["caption_projection.linear_1.weight"]; w.TryGetValue("caption_projection.linear_1.bias", out _capB1);
        _capW2 = w["caption_projection.linear_2.weight"]; w.TryGetValue("caption_projection.linear_2.bias", out _capB2);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"transformer_blocks.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _projInW, _projInB, _projOutW, _projOutB, _finalScaleShift,
            _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeLinW, _timeLinB, _capW1, _capB1, _capW2, _capB2 })
            if (t is not null) yield return t;
        for (int i = 0; i < _blocks.Length; i++) foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
    }

    /// <summary>Velocity prediction over VAE-latent tokens. <paramref name="latentTokens"/> is <c>[S, inChannels]</c> with <c>S = numFrames·height·width</c> in <c>(f,h,w)</c> order; <paramref name="encoder"/> is raw T5 features <c>[L, captionChannels]</c>; <paramref name="encoderMask"/> is an optional additive cross-attn mask. Returns <c>[S, outChannels]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latentTokens, Tensor encoder, float timestep,
        (int Frames, int Height, int Width) grid, (double T, double H, double W) interpScale, Tensor? encoderMask)
    {
        int s = (int)latentTokens.Shape[0];
        int dim = _config.InnerDim;

        (Tensor cos, Tensor sin) = _rope.BuildCosSin(grid.Frames, grid.Height, grid.Width, interpScale);

        Tensor hidden = new Tensor(new TensorShape(s, dim), DType.F32);
        backend.Linear(hidden, latentTokens, _projInW!, _projInB);
        LtxVideoDebugDump.Dump("proj_in", hidden);

        (Tensor temb6, Tensor embedded) = TimeEmbed(backend, timestep);
        Tensor encoderProj = CaptionProject(backend, encoder);

        Tensor cur = hidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, temb6, _rope, cos, sin, encoderMask);
            cur.Dispose();
            cur = next;
            LtxVideoDebugDump.Dump($"blocks.{i}", cur);
        }
        cos.Dispose(); sin.Dispose(); temb6.Dispose(); encoderProj.Dispose();

        Tensor outVel = FinalLayer(backend, cur, embedded, s, dim);
        cur.Dispose();
        embedded.Dispose();
        LtxVideoDebugDump.DumpOutput(outVel);
        return outVel;
    }

    /// <summary>AdaLayerNormSingle: <c>embedded = mlp(sinusoidal(t))</c>; <c>temb = linear(silu(embedded))</c> reshaped to <c>[6, dim]</c>.</summary>
    private (Tensor Temb6, Tensor Embedded) TimeEmbed(IBackend backend, float timestep)
    {
        int dim = _config.InnerDim;
        Tensor sinEmb = new Tensor(new TensorShape(1, 256), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmb, timestep, 1, 256, 10000f);
        Tensor e1 = new Tensor(new TensorShape(1, dim), DType.F32);
        backend.Linear(e1, sinEmb, _timeEmb1W!, _timeEmb1B); sinEmb.Dispose();
        Tensor e1a = new Tensor(e1.Shape, DType.F32); backend.Silu(e1a, e1); e1.Dispose();
        Tensor embedded = new Tensor(new TensorShape(dim), DType.F32);
        Tensor embedded2d = new Tensor(new TensorShape(1, dim), DType.F32);
        backend.Linear(embedded2d, e1a, _timeEmb2W!, _timeEmb2B); e1a.Dispose();
        Buffer.MemoryCopy((float*)embedded2d.DataPointer, (float*)embedded.DataPointer, (long)dim * 4, (long)dim * 4);

        // temb = linear(silu(embedded)) → [6*dim] → [6, dim]
        Tensor sil = new Tensor(new TensorShape(1, dim), DType.F32); backend.Silu(sil, embedded2d); embedded2d.Dispose();
        Tensor temb = new Tensor(new TensorShape(6, dim), DType.F32);
        Tensor tembFlat = new Tensor(new TensorShape(1, 6 * dim), DType.F32);
        backend.Linear(tembFlat, sil, _timeLinW!, _timeLinB); sil.Dispose();
        Buffer.MemoryCopy((float*)tembFlat.DataPointer, (float*)temb.DataPointer, (long)6 * dim * 4, (long)6 * dim * 4);
        tembFlat.Dispose();
        return (temb, embedded);
    }

    /// <summary>PixArtAlphaTextProjection: <c>linear_2(gelu_tanh(linear_1(x)))</c>, T5 <c>[L, captionChannels] → [L, dim]</c>.</summary>
    private Tensor CaptionProject(IBackend backend, Tensor encoder)
    {
        int l = (int)encoder.Shape[0];
        int dim = _config.InnerDim;
        Tensor h1 = new Tensor(new TensorShape(l, dim), DType.F32);
        backend.Linear(h1, encoder, _capW1!, _capB1);
        Tensor act = new Tensor(h1.Shape, DType.F32); backend.Gelu(act, h1); h1.Dispose();
        Tensor outT = new Tensor(new TensorShape(l, dim), DType.F32);
        backend.Linear(outT, act, _capW2!, _capB2); act.Dispose();
        return outT;
    }

    private Tensor FinalLayer(IBackend backend, Tensor hidden, Tensor embedded, int s, int dim)
    {
        // shift = ss[0]+embedded; scale = ss[1]+embedded ([dim], broadcast over S).
        float* ss = (float*)_finalScaleShift!.DataPointer;
        float* em = (float*)embedded.DataPointer;
        Tensor shift = new Tensor(new TensorShape(dim), DType.F32);
        Tensor scale = new Tensor(new TensorShape(dim), DType.F32);
        float* shp = (float*)shift.DataPointer; float* scp = (float*)scale.DataPointer;
        for (int d = 0; d < dim; d++) { shp[d] = ss[d] + em[d]; scp[d] = ss[dim + d] + em[d]; }

        Tensor normed = new Tensor(new TensorShape(s, dim), DType.F32);
        DiTUtils.LayerNormNoAffine(normed, hidden, 1, s, dim, 1e-6f);
        float* np = (float*)normed.DataPointer;
        for (int i = 0; i < s; i++)
            for (int d = 0; d < dim; d++)
                np[i * dim + d] = np[i * dim + d] * (1f + scp[d]) + shp[d];
        shift.Dispose(); scale.Dispose();

        Tensor outVel = new Tensor(new TensorShape(s, _config.OutChannels), DType.F32);
        backend.Linear(outVel, normed, _projOutW!, _projOutB);
        normed.Dispose();
        return outVel;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = w[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _projInW = _projInB = _projOutW = _projOutB = _finalScaleShift = null;
            _timeEmb1W = _timeEmb1B = _timeEmb2W = _timeEmb2B = _timeLinW = _timeLinB = null;
            _capW1 = _capB1 = _capW2 = _capB2 = null;
        }
        GC.SuppressFinalize(this);
    }
}
