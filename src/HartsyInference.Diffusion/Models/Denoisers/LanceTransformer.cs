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

    // Step-invariant per-prompt state, keyed on the caller's array/tensor references (cond + uncond entries).
    // Text-segment embeds, the gathered latent-pos rows, and the M-RoPE cos/sin tables are all constant across
    // the denoise loop, so each builds once per prompt instead of once per forward. Cleared on Dispose.
    private readonly RefCache _textSegmentCache = new();
    private readonly RefCache _posEmbedCache = new();
    private readonly RefCache _cosSinCache = new();
    private Tensor? _onesRow;                  // [1, hidden] of 1.0f for the broadcast-add of the timestep embed

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
        _embedTokens = TensorCasts.LoadF32(weights, "embed_tokens.weight");
        _vae2llmW = weights["vae2llm.weight"];
        weights.TryGetValue("vae2llm.bias", out _vae2llmB);
        _llm2vaeW = weights["llm2vae.weight"];
        weights.TryGetValue("llm2vae.bias", out _llm2vaeB);
        _latentPosEmbed = TensorCasts.LoadF32(weights, "latent_pos_embed.pos_embed");
        _timeMlp0W = weights["time_embedder.mlp.0.weight"];
        weights.TryGetValue("time_embedder.mlp.0.bias", out _timeMlp0B);
        _timeMlp2W = weights["time_embedder.mlp.2.weight"];
        weights.TryGetValue("time_embedder.mlp.2.bias", out _timeMlp2B);
        _normUnd = TensorCasts.LoadF32(weights, "norm.weight");
        _normGen = TensorCasts.LoadF32(weights, "norm_moe_gen.weight");
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

        // ── Contiguous role layout (BuildGenSequence's invariant): und [0, nPre) + [nPre+nVae, seq), gen the
        // middle run. The GPU-resident block path routes roles via contiguous row slices instead of the old
        // host gather/scatter, so validate the layout once up front. ──
        (int nPre, int nTail) = ValidateContiguousLayout(undIdx, genIdx, seq);

        // ── Build packed sequence [seq, hidden] on the device: cached text segments + per-step VAE tokens ──
        (Tensor txtPre, Tensor txtTail) = GetOrBuildTextSegments(textTokenIds, nPre, nTail, hidden);

        // VAE tokens: vae2llm(latent) + latent_pos_embed[posId] + timestep embed (broadcast over rows).
        Tensor vaeProj = new Tensor(new TensorShape(nVae, hidden), DType.F32);
        backend.Linear(vaeProj, latentTokens, _vae2llmW!, _vae2llmB);
        Tensor posRows = GetOrBuildPosEmbedRows(latentPosIds, nVae, hidden);
        Tensor vaePlusPos = new Tensor(vaeProj.Shape, DType.F32);
        backend.Add(vaePlusPos, vaeProj, posRows);
        vaeProj.Dispose();
        Tensor tEmb = ComputeTimestepEmbedding(backend, timestep);
        _onesRow ??= CreateOnesRow(hidden);
        Tensor vaeHidden = new Tensor(vaePlusPos.Shape, DType.F32);
        backend.AffineBroadcastLastDim(vaeHidden, vaePlusPos, _onesRow, tEmb);
        vaePlusPos.Dispose();
        tEmb.Dispose();

        Tensor h = new Tensor(new TensorShape(seq, hidden), DType.F32);
        if (nTail > 0)
            backend.Concat(h, new[] { txtPre, vaeHidden, txtTail }, 0);
        else
            backend.Concat(h, new[] { txtPre, vaeHidden }, 0);
        vaeHidden.Dispose();
        LanceDebugDump.Dump("packed_in", h);

        // ── M-RoPE cos/sin (cached per prompt: positions are step-invariant) ──
        (Tensor cos, Tensor sin) = GetOrBuildCosSin(positionIds);

        // ── 36 MoT blocks ──
        Tensor cur = h;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, cos, sin, nPre, nVae, attentionMask);
            cur.Dispose();
            cur = next;
            LanceDebugDump.Dump($"layers.{i}", cur);
        }

        // ── Final norm (gen role) + llm2vae on gen tokens (contiguous rows [nPre, nPre+nVae)) ──
        Tensor genHidden = new Tensor(new TensorShape(nVae, hidden), DType.F32);
        backend.SliceRows(genHidden, cur, nPre);
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

    /// <summary>Asserts the packed layout is und-prefix / gen-run / und-tail and returns the segment sizes.</summary>
    private static (int nPre, int nTail) ValidateContiguousLayout(int[] undIdx, int[] genIdx, int seq)
    {
        int nGen = genIdx.Length;
        int nPre = nGen > 0 ? genIdx[0] : undIdx.Length;
        for (int i = 0; i < nGen; i++)
            if (genIdx[i] != nPre + i)
                throw new ArgumentException($"genIdx must be one contiguous run (genIdx[{i}]={genIdx[i]}, expected {nPre + i}).", nameof(genIdx));
        for (int i = 0; i < undIdx.Length; i++)
        {
            int expected = i < nPre ? i : nGen + i;
            if (undIdx[i] != expected)
                throw new ArgumentException($"undIdx must be the prefix+tail around the gen run (undIdx[{i}]={undIdx[i]}, expected {expected}).", nameof(undIdx));
        }
        return (nPre, seq - nPre - nGen);
    }

    /// <summary>Returns the cached <c>(prefix, tail)</c> text-segment embeds for this token array, gathering rows
    /// from <c>embed_tokens</c> on first use (host gather once per prompt; the tensors upload once by reference).</summary>
    private (Tensor pre, Tensor tail) GetOrBuildTextSegments(int[] textTokenIds, int nPre, int nTail, int hidden)
    {
        Tensor[]? cached = _textSegmentCache.Get(textTokenIds);
        if (cached is not null) return (cached[0], cached[1]);

        long vocabRows = _embedTokens!.Shape[0];
        float* eptr = (float*)_embedTokens.DataPointer;
        Tensor pre = new Tensor(new TensorShape(nPre, hidden), DType.F32);
        Tensor tail = new Tensor(new TensorShape(Math.Max(nTail, 1), hidden), DType.F32);
        float* prePtr = (float*)pre.DataPointer;
        float* tailPtr = (float*)tail.DataPointer;
        for (int i = 0; i < textTokenIds.Length; i++)
        {
            if ((uint)textTokenIds[i] >= (uint)vocabRows)
                throw new ArgumentOutOfRangeException(nameof(textTokenIds), $"token id {textTokenIds[i]} out of range for the {vocabRows}-row embedding.");
            float* dst = i < nPre ? prePtr + (long)i * hidden : tailPtr + (long)(i - nPre) * hidden;
            Buffer.MemoryCopy(eptr + (long)textTokenIds[i] * hidden, dst, (long)hidden * 4, (long)hidden * 4);
        }
        _textSegmentCache.Put(textTokenIds, [pre, tail]);
        return (pre, tail);
    }

    /// <summary>Returns the cached <c>[nVae, hidden]</c> gathered <c>latent_pos_embed</c> rows for this pos-id array.</summary>
    private Tensor GetOrBuildPosEmbedRows(int[] latentPosIds, int nVae, int hidden)
    {
        Tensor[]? cached = _posEmbedCache.Get(latentPosIds);
        if (cached is not null) return cached[0];

        long tableRows = _latentPosEmbed!.Shape[0];
        float* pe = (float*)_latentPosEmbed.DataPointer;
        Tensor rows = new Tensor(new TensorShape(nVae, hidden), DType.F32);
        float* dst = (float*)rows.DataPointer;
        for (int i = 0; i < nVae; i++)
        {
            int row = latentPosIds[i];
            if ((uint)row >= (uint)tableRows)
                throw new ArgumentOutOfRangeException(nameof(latentPosIds),
                    $"latent pos id {row} out of range for the checkpoint's {tableRows}-row latent_pos_embed table (grid too large for this variant).");
            Buffer.MemoryCopy(pe + (long)row * hidden, dst + (long)i * hidden, (long)hidden * 4, (long)hidden * 4);
        }
        _posEmbedCache.Put(latentPosIds, [rows]);
        return rows;
    }

    /// <summary>Returns the cached <c>[1, seq, headDim]</c> M-RoPE cos/sin tables for this position tensor.</summary>
    private (Tensor cos, Tensor sin) GetOrBuildCosSin(Tensor positionIds)
    {
        Tensor[]? cached = _cosSinCache.Get(positionIds);
        if (cached is not null) return (cached[0], cached[1]);

        (Tensor cos, Tensor sin) = _rope.BuildCosSin(positionIds);
        // Re-home into owned [1, seq, headDim] tensors (the rank ApplyRopeSingle broadcasts over heads) with
        // stable references — the device cache is keyed by tensor identity, so a per-forward view would re-upload.
        long rows = cos.Shape[0], dim = cos.Shape[1];
        Tensor cos3 = new Tensor(new TensorShape(1, rows, dim), DType.F32);
        Tensor sin3 = new Tensor(new TensorShape(1, rows, dim), DType.F32);
        long bytes = rows * dim * sizeof(float);
        Buffer.MemoryCopy((float*)cos.DataPointer, (float*)cos3.DataPointer, bytes, bytes);
        Buffer.MemoryCopy((float*)sin.DataPointer, (float*)sin3.DataPointer, bytes, bytes);
        cos.Dispose();
        sin.Dispose();
        _cosSinCache.Put(positionIds, [cos3, sin3]);
        return (cos3, sin3);
    }

    private static Tensor CreateOnesRow(int hidden)
    {
        Tensor ones = new Tensor(new TensorShape(1, hidden), DType.F32);
        new Span<float>((float*)ones.DataPointer, hidden).Fill(1.0f);
        return ones;
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

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _textSegmentCache.Clear();
            _posEmbedCache.Clear();
            _cosSinCache.Clear();
            _onesRow?.Dispose();
            _onesRow = null;
            _embedTokens = _vae2llmW = _vae2llmB = _llm2vaeW = _llm2vaeB = null;
            _latentPosEmbed = null;
            _timeMlp0W = _timeMlp0B = _timeMlp2W = _timeMlp2B = null;
            _normUnd = _normGen = null;
        }
        GC.SuppressFinalize(this);
    }

    /// <summary>Two-entry reference-keyed tensor cache (cond + uncond); a third distinct key evicts the oldest.</summary>
    private sealed class RefCache
    {
        private readonly List<(object Key, Tensor[] Vals)> _entries = new(2);

        public Tensor[]? Get(object key)
        {
            for (int i = 0; i < _entries.Count; i++)
                if (ReferenceEquals(_entries[i].Key, key))
                    return _entries[i].Vals;
            return null;
        }

        public void Put(object key, Tensor[] vals)
        {
            if (_entries.Count >= 2)
            {
                foreach (Tensor t in _entries[0].Vals) t.Dispose();
                _entries.RemoveAt(0);
            }
            _entries.Add((key, vals));
        }

        public void Clear()
        {
            for (int i = 0; i < _entries.Count; i++)
                foreach (Tensor t in _entries[i].Vals) t.Dispose();
            _entries.Clear();
        }
    }
}
