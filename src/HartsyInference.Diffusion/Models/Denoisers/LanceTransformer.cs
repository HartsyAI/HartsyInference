using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Lance (ByteDance, Apache-2.0) MoT-augmented Qwen2.5-VL backbone for the **generation** path (T2I/T2V velocity prediction). Ported from <c>modeling/lance/{lance.py, qwen2_navit.py}</c> and reconciled against the real <c>Lance_3B</c> checkpoint keys.
///
/// <para>This is the unified-sequence transformer: a packed sequence carries text tokens (und role, including the <c>&lt;|vision_start|&gt;</c>/<c>&lt;|vision_end|&gt;</c> sentinels) and noisy VAE tokens (gen role); each <see cref="LanceMoTBlock"/> routes per role through its parameter set and runs one joint attention. Text tokens are embedded via <c>embed_tokens</c>; VAE tokens via <c>vae2llm</c> (48-dim latent pixel → hidden) plus the checkpoint's frozen <c>latent_pos_embed</c> row (indexed <c>t·64² + h·64 + w</c>) plus the timestep embedding. After 36 blocks, the gen tokens are read out via <c>llm2vae</c> back to latent space.</para>
///
/// <para>Sequence layout (positions, the sparse attention mask) is the pipeline's job — this class takes the role partition + precomputed M-RoPE cos/sin + an optional mask, keeping it reusable.</para></summary>
public sealed unsafe class LanceTransformer : IDisposable
{
    private readonly LanceConfig _config;
    private readonly LanceMoTBlock[] _blocks;
    private readonly Multimodal3DRope _rope;
    private int _disposed;

    private Tensor? _embedTokens;              // [vocab, hidden] (cast to F32 at load)
    private Tensor? _vae2llmW, _vae2llmB;      // Linear(48 → hidden)
    private Tensor? _llm2vaeW, _llm2vaeB;      // Linear(hidden → 48)
    private Tensor? _latentPosEmbed;           // frozen sin-cos table [maxT·64·64, hidden] (cast to F32)
    private Tensor? _timeMlp0W, _timeMlp0B, _timeMlp2W, _timeMlp2B;
    private Tensor? _normUnd, _normGen;        // final RMSNorm per role

    public LanceTransformer(LanceConfig config)
    {
        _config = config;
        _blocks = new LanceMoTBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new LanceMoTBlock(config.HiddenSize, config.NumHeads, config.NumKvHeads,
                config.IntermediateSize, config.RmsNormEps, config.QkNorm);
        _rope = new Multimodal3DRope(config.HeadDim, config.RopeTheta, config.MropeSection);
    }

    public LanceConfig Config => _config;

    /// <summary>Loads weights using the REAL checkpoint key names. Expects the converter to have stripped the <c>language_model.model.</c> prefix off the backbone keys (→ <c>embed_tokens</c>, <c>layers.{i}.*</c>, <c>norm</c>, <c>norm_moe_gen</c>) and passed the top-level Lance heads through unchanged (<c>vae2llm.*</c>, <c>llm2vae.*</c>, <c>latent_pos_embed.pos_embed</c>, <c>time_embedder.mlp.{0,2}.*</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _embedTokens = LoadF32(weights, "embed_tokens.weight");
        _vae2llmW = weights["vae2llm.weight"];
        weights.TryGetValue("vae2llm.bias", out _vae2llmB);
        _llm2vaeW = weights["llm2vae.weight"];
        weights.TryGetValue("llm2vae.bias", out _llm2vaeB);
        _latentPosEmbed = LoadF32(weights, "latent_pos_embed.pos_embed");
        _timeMlp0W = weights["time_embedder.mlp.0.weight"];
        weights.TryGetValue("time_embedder.mlp.0.bias", out _timeMlp0B);
        _timeMlp2W = weights["time_embedder.mlp.2.weight"];
        weights.TryGetValue("time_embedder.mlp.2.bias", out _timeMlp2B);
        _normUnd = LoadF32(weights, "norm.weight");
        _normGen = LoadF32(weights, "norm_moe_gen.weight");
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"layers.{i}");
    }

    /// <summary>Enumerates all weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _embedTokens, _vae2llmW, _vae2llmB, _llm2vaeW, _llm2vaeB,
            _timeMlp0W, _timeMlp0B, _timeMlp2W, _timeMlp2B, _normUnd, _normGen })
            if (t is not null) yield return t;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
    }

    /// <summary>Predicts velocity for the noisy VAE tokens.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="textTokenIds">Text token ids (incl. vision sentinels), placed at <paramref name="undIdx"/> positions in order.</param>
    /// <param name="latentTokens">Noisy VAE tokens <c>[nVae, PatchFeatureDim]</c>, placed at <paramref name="genIdx"/> positions in order.</param>
    /// <param name="latentPosIds">Row index into the frozen <c>latent_pos_embed</c> table per VAE token (<c>t·64² + h·64 + w</c>); length nVae.</param>
    /// <param name="timestep">Flow-matching time in [0,1] (already logit-normal-shifted by the pipeline).</param>
    /// <param name="positionIds">Full M-RoPE positions <c>[seq, 3]</c> from the pipeline's <c>get_rope_index</c> replica.</param>
    /// <param name="undIdx">Sequence positions of und (text) tokens.</param>
    /// <param name="genIdx">Sequence positions of gen (VAE) tokens.</param>
    /// <param name="attentionMask">Optional additive mask <c>[1,1,seq,seq]</c>; null = full attention.</param>
    /// <returns>Velocity <c>[nVae, PatchFeatureDim]</c> in gen-token order.</returns>
    public Tensor Forward(IBackend backend, int[] textTokenIds, Tensor latentTokens, int[] latentPosIds,
        float timestep, Tensor positionIds, int[] undIdx, int[] genIdx, Tensor? attentionMask)
    {
        int seq = undIdx.Length + genIdx.Length;
        int hidden = _config.HiddenSize;
        int nVae = genIdx.Length;
        if (textTokenIds.Length != undIdx.Length)
            throw new ArgumentException($"textTokenIds {textTokenIds.Length} != undIdx {undIdx.Length}.");
        if ((int)latentTokens.Shape[0] != nVae)
            throw new ArgumentException($"latentTokens rows {latentTokens.Shape[0]} != genIdx {nVae}.");
        if (latentPosIds.Length != nVae)
            throw new ArgumentException($"latentPosIds {latentPosIds.Length} != nVae {nVae}.");

        // ── Build packed sequence [seq, hidden] ──
        Tensor h = new Tensor(new TensorShape(seq, hidden), DType.F32);
        new Span<float>((float*)h.DataPointer, checked((int)((long)seq * hidden))).Clear();
        EmbedText(h, textTokenIds, undIdx, hidden);

        // VAE tokens: vae2llm(latent) + latent_pos_embed[posId] + timestep embed
        Tensor vaeHidden = new Tensor(new TensorShape(nVae, hidden), DType.F32);
        backend.Linear(vaeHidden, latentTokens, _vae2llmW!, _vae2llmB);
        Tensor tEmb = ComputeTimestepEmbedding(backend, timestep);
        AddVaeConditioning(vaeHidden, latentPosIds, tEmb, nVae, hidden);
        tEmb.Dispose();
        ScatterRows(vaeHidden, genIdx, h, hidden);
        vaeHidden.Dispose();
        LanceDebugDump.Dump("packed_in", h);

        // ── M-RoPE cos/sin ──
        (Tensor cos, Tensor sin) = _rope.BuildCosSin(positionIds);

        // ── 36 MoT blocks ──
        Tensor cur = h;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, cos, sin, _rope, undIdx, genIdx, attentionMask);
            cur.Dispose();
            cur = next;
            LanceDebugDump.Dump($"layers.{i}", cur);
        }
        cos.Dispose();
        sin.Dispose();

        // ── Final norm (gen role) + llm2vae on gen tokens ──
        Tensor genHidden = GatherRows(cur, genIdx, hidden);
        cur.Dispose();
        Tensor genNormed = new Tensor(genHidden.Shape, DType.F32);
        backend.RmsNorm(genNormed, genHidden, _normGen!, _config.RmsNormEps);
        genHidden.Dispose();
        Tensor velocity = new Tensor(new TensorShape(nVae, _config.PatchFeatureDim), DType.F32);
        backend.Linear(velocity, genNormed, _llm2vaeW!, _llm2vaeB);
        genNormed.Dispose();
        LanceDebugDump.DumpOutput(velocity);
        return velocity;
    }

    private void EmbedText(Tensor h, int[] tokenIds, int[] undIdx, int hidden)
    {
        long vocabRows = _embedTokens!.Shape[0];
        float* hPtr = (float*)h.DataPointer;
        float* eptr = (float*)_embedTokens.DataPointer;
        for (int i = 0; i < tokenIds.Length; i++)
        {
            if ((uint)tokenIds[i] >= (uint)vocabRows)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"token id {tokenIds[i]} out of range for the {vocabRows}-row embedding.");
            long src = (long)tokenIds[i] * hidden;
            long dst = (long)undIdx[i] * hidden;
            Buffer.MemoryCopy(eptr + src, hPtr + dst, (long)hidden * 4, (long)hidden * 4);
        }
    }

    private void AddVaeConditioning(Tensor vaeHidden, int[] latentPosIds, Tensor tEmb, int nVae, int hidden)
    {
        long tableRows = _latentPosEmbed!.Shape[0];
        float* vh = (float*)vaeHidden.DataPointer;
        float* pe = (float*)_latentPosEmbed.DataPointer;
        float* te = (float*)tEmb.DataPointer; // [1, hidden]
        for (int i = 0; i < nVae; i++)
        {
            int row = latentPosIds[i];
            if ((uint)row >= (uint)tableRows)
                throw new ArgumentOutOfRangeException(nameof(latentPosIds),
                    $"latent pos id {row} out of range for the checkpoint's {tableRows}-row latent_pos_embed table (grid too large for this variant).");
            long off = (long)i * hidden;
            long peOff = (long)row * hidden;
            for (int d = 0; d < hidden; d++)
                vh[off + d] += pe[peOff + d] + te[d];
        }
    }

    /// <summary>Timestep embedding: sinusoidal (256, <c>[cos,sin]</c>) → Linear → SiLU → Linear → <c>[1, hidden]</c>. The input is the raw shifted flow time (upstream feeds t∈[0,1] directly, no ×1000).</summary>
    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep)
    {
        int freqDim = _config.TimestepFrequencyDim;
        int hidden = _config.HiddenSize;
        Tensor freq = new Tensor(new TensorShape(1, freqDim), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(freq, timestep, 1, freqDim, 10000f);
        Tensor m0 = new Tensor(new TensorShape(1, hidden), DType.F32);
        backend.Linear(m0, freq, _timeMlp0W!, _timeMlp0B);
        freq.Dispose();
        Tensor act = new Tensor(new TensorShape(1, hidden), DType.F32);
        backend.Silu(act, m0);
        m0.Dispose();
        Tensor outT = new Tensor(new TensorShape(1, hidden), DType.F32);
        backend.Linear(outT, act, _timeMlp2W!, _timeMlp2B);
        act.Dispose();
        return outT;
    }

    private static Tensor GatherRows(Tensor src, int[] idx, int dim)
    {
        Tensor outT = new Tensor(new TensorShape(idx.Length, dim), DType.F32);
        float* s = (float*)src.DataPointer;
        float* d = (float*)outT.DataPointer;
        for (int i = 0; i < idx.Length; i++)
            Buffer.MemoryCopy(s + (long)idx[i] * dim, d + (long)i * dim, (long)dim * 4, (long)dim * 4);
        return outT;
    }

    private static void ScatterRows(Tensor src, int[] idx, Tensor dst, int dim)
    {
        float* s = (float*)src.DataPointer;
        float* d = (float*)dst.DataPointer;
        for (int i = 0; i < idx.Length; i++)
            Buffer.MemoryCopy(s + (long)i * dim, d + (long)idx[i] * dim, (long)dim * 4, (long)dim * 4);
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
            _embedTokens = _vae2llmW = _vae2llmB = _llm2vaeW = _llm2vaeB = null;
            _latentPosEmbed = null;
            _timeMlp0W = _timeMlp0B = _timeMlp2W = _timeMlp2B = null;
            _normUnd = _normGen = null;
        }
        GC.SuppressFinalize(this);
    }
}
