using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Ideogram 4 flow-matching transformer (<c>Ideogram4Transformer</c>), ported verbatim from upstream <c>modeling_ideogram4.py</c>. Single-stream, unified-sequence DiT — see <see cref="Ideogram4Config"/> for the architectural overview.
///
/// <para>The pipeline loads TWO instances of this class: a conditional transformer (<c>transformer/</c> weights) and an unconditional one (<c>unconditional_transformer/</c> weights) — same architecture, different weights — for the asymmetric CFG.</para>
///
/// <para>Forward fuses text and image into one length-L sequence by masked addition: image tokens contribute <c>input_proj(x)</c>, text tokens contribute <c>llm_cond_proj(llm_cond_norm(llm_features))</c>, and an <c>image_indicator</c> embedding tags each token. Only positions with <c>indicator == OUTPUT_IMAGE_INDICATOR(2)</c> produce meaningful velocity.</para></summary>
public sealed unsafe class Ideogram4Transformer : IDisposable
{
    private const int LlmTokenIndicator = 3;
    private const int OutputImageIndicator = 2;

    private readonly Ideogram4Config _config;
    private readonly Ideogram4Block[] _blocks;
    private readonly Ideogram4Mrope _rope;
    private int _disposed;

    private Tensor? _inputProjW, _inputProjB;
    private Tensor? _llmCondNormW;
    private Tensor? _llmCondProjW, _llmCondProjB;
    private Tensor? _tMlpInW, _tMlpInB, _tMlpOutW, _tMlpOutB;
    private Tensor? _adalnProjW, _adalnProjB;
    private Tensor? _imageIndicatorW; // Embedding(2, emb)
    private Tensor? _finalLinearW, _finalLinearB;
    private Tensor? _finalAdalnW, _finalAdalnB;

    // Step-invariant caches, keyed on the caller's tensor references (the pipeline keeps them stable across
    // steps and across repeat-prompt generations). The projected text conditioning and the MRoPE cos/sin
    // tables are timestep-independent, so recomputing them per Forward re-uploaded and re-projected ~1 GB of
    // constant data every step. Both are pinned via PreloadWeights so per-step FreeActivations cannot evict
    // their device copies; FreeWeights+Dispose on replacement (the Krea2Transformer _cachedTxt pattern).
    private IBackend? _cacheBackend;
    private Tensor? _cachedLlmProj;
    private object? _cachedLlmProjKey;
    private Tensor? _cachedCos, _cachedSin;
    private object? _cachedRopeKey;

    public Ideogram4Transformer(Ideogram4Config config)
    {
        _config = config;
        _blocks = new Ideogram4Block[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new Ideogram4Block(config.EmbDim, config.NumHeads, config.IntermediateSize, config.NormEps);
        _rope = new Ideogram4Mrope(config.HeadDim, config.RopeTheta, config.MropeSection);
    }

    /// <summary>The configured architecture.</summary>
    public Ideogram4Config Config => _config;

    /// <summary>Loads weights from an upstream-named dict (post-converter, prefix already stripped to bare <c>input_proj.*</c>, <c>layers.{i}.*</c>, etc.).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _inputProjW = weights["input_proj.weight"];
        weights.TryGetValue("input_proj.bias", out _inputProjB);

        _llmCondNormW = TensorCasts.LoadF32(weights, "llm_cond_norm.weight");
        _llmCondProjW = weights["llm_cond_proj.weight"];
        weights.TryGetValue("llm_cond_proj.bias", out _llmCondProjB);

        _tMlpInW = weights["t_embedding.mlp_in.weight"];
        weights.TryGetValue("t_embedding.mlp_in.bias", out _tMlpInB);
        _tMlpOutW = weights["t_embedding.mlp_out.weight"];
        weights.TryGetValue("t_embedding.mlp_out.bias", out _tMlpOutB);

        _adalnProjW = weights["adaln_proj.weight"];
        weights.TryGetValue("adaln_proj.bias", out _adalnProjB);

        _imageIndicatorW = TensorCasts.LoadF32(weights, "embed_image_indicator.weight");

        _finalLinearW = weights["final_layer.linear.weight"];
        weights.TryGetValue("final_layer.linear.bias", out _finalLinearB);
        _finalAdalnW = weights["final_layer.adaln_modulation.weight"];
        weights.TryGetValue("final_layer.adaln_modulation.bias", out _finalAdalnB);

        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"layers.{i}");
    }

    /// <summary>The number of streamable transformer blocks.</summary>
    public int BlockCount => _blocks.Length;

    /// <summary>The streamable block at <paramref name="idx"/>.</summary>
    /// <remarks>Returns the live block instance, so <see cref="IStreamingBlock.EnumerateWeights"/> hands back the same
    /// tensor references every call — the residency tracking in <c>BlockStreamingController</c> is reference-based.</remarks>
    public IStreamingBlock GetBlock(int idx) => _blocks[idx];

    /// <summary>Hook invoked with each block index immediately before that block's forward pass; null = no streaming.</summary>
    /// <remarks>Pipelines plug a <c>BlockStreamingController</c> here so resident VRAM peaks at roughly
    /// (activations + the prefetch window) instead of the whole DiT — which is what lets Ideogram 4's *pair* of 9.3 GB
    /// transformers run on a card that cannot hold them both.
    /// <para><b>If a captured-graph fast path is ever added to this transformer it MUST be disabled whenever this is
    /// non-null</b> — a graph bakes weight device pointers, and streaming re-points them every forward, so replaying a
    /// captured graph over streamed weights reads freed memory (a context-poisoning CUDA 700). See
    /// <c>HunyuanVideoDit</c> / <c>LtxVideo2Transformer</c> for the guard. There is no graph path here today.</para></remarks>
    public Action<int>? BeforeBlockForward { get; set; }

    /// <summary>The non-block weights: embedders, timestep/AdaLN MLPs and the output head.</summary>
    /// <remarks>Touched on every forward regardless of which block is executing, so these stay eagerly resident even on
    /// the streamed path — streaming them would re-upload the same tensors N times per step for no VRAM saving.</remarks>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        if (_inputProjW is not null) yield return _inputProjW;
        if (_inputProjB is not null) yield return _inputProjB;
        if (_llmCondNormW is not null) yield return _llmCondNormW;
        if (_llmCondProjW is not null) yield return _llmCondProjW;
        if (_llmCondProjB is not null) yield return _llmCondProjB;
        if (_tMlpInW is not null) yield return _tMlpInW;
        if (_tMlpInB is not null) yield return _tMlpInB;
        if (_tMlpOutW is not null) yield return _tMlpOutW;
        if (_tMlpOutB is not null) yield return _tMlpOutB;
        if (_adalnProjW is not null) yield return _adalnProjW;
        if (_adalnProjB is not null) yield return _adalnProjB;
        if (_imageIndicatorW is not null) yield return _imageIndicatorW;
        if (_finalLinearW is not null) yield return _finalLinearW;
        if (_finalLinearB is not null) yield return _finalLinearB;
        if (_finalAdalnW is not null) yield return _finalAdalnW;
        if (_finalAdalnB is not null) yield return _finalAdalnB;
    }

    /// <summary>Enumerates every weight tensor for GPU preloading: the shared set plus every block.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in EnumerateSharedWeights()) yield return w;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Velocity prediction over a unified text+image sequence. The text conditioning projection and
    /// the MRoPE tables are step-invariant, so they are computed once and cached keyed on the caller's tensor
    /// references — pass the SAME <paramref name="llmTextFeatures"/> / <paramref name="positionIds"/> instances
    /// every step (and across repeat-prompt generations) to hit the caches.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="llmTextFeatures">Qwen3-VL conditioning for the TEXT rows only, <c>[B, numText, llmFeaturesDim]</c>. Null for a text-free (unconditional) pass — its projection is exactly zero, so the whole text path is skipped.</param>
    /// <param name="imageTokens">Image latent tokens <c>[B, numImg, inChannels]</c> (the sequence tail; text rows are synthesized as zeros).</param>
    /// <param name="timestep">Flow-matching time in [0,1], shared across the batch.</param>
    /// <param name="positionIds">MRoPE positions <c>[B, numText+numImg, 3]</c> F32 (temporal, height, width).</param>
    /// <param name="indicator">Per-token role, length <c>B*L</c> row-major: <c>LLM_TOKEN_INDICATOR(3)</c> first <c>numText</c> rows, then <c>OUTPUT_IMAGE_INDICATOR(2)</c>.</param>
    /// <param name="attentionMask">Optional additive mask <c>[B, 1, L, L]</c> or <c>[B, 1, 1, L]</c>; null = full attention (correct for unpadded single-prompt B=1).</param>
    /// <param name="stepCache">Optional across-step First-Block cache (one instance PER transformer — the
    /// conditional and unconditional models each need their own). Block 0 always runs; on a gate hit blocks
    /// 1..N−1 are reconstructed from the previous step's residual. Null = byte-identical uncached forward.</param>
    /// <returns><c>[B, numImg, inChannels]</c> image-token velocity (F32); text rows are dropped before the final layer.</returns>
    public Tensor Forward(IBackend backend, Tensor? llmTextFeatures, Tensor imageTokens, float timestep,
        Tensor positionIds, int[] indicator, Tensor? attentionMask,
        Utilities.DeviceFeatureCache? stepCache = null)
    {
        int batch = (int)imageTokens.Shape[0];
        int numImg = (int)imageTokens.Shape[1];
        int seqLen = (int)positionIds.Shape[1];
        int numText = seqLen - numImg;
        int emb = _config.EmbDim;
        TensorShape hidShape = new TensorShape(batch, seqLen, emb);

        if (indicator.Length != batch * seqLen)
            throw new ArgumentException($"indicator length {indicator.Length} != B*L {batch * seqLen}.", nameof(indicator));
        if (numText > 0 && (llmTextFeatures is null || (int)llmTextFeatures.Shape[1] != numText))
            throw new ArgumentException($"llmTextFeatures rows {llmTextFeatures?.Shape[1].ToString() ?? "null"} != numText {numText}.", nameof(llmTextFeatures));
        if (numText > 0 && batch != 1)
            throw new ArgumentException("Text+image sequences are single-batch only (row-block scatter/slice).", nameof(imageTokens));
        _cacheBackend = backend;

        // ── Image embedding: input_proj over the image rows only, scattered after a zeroed text head. ──
        // Bit-identical to the old full-sequence masked path: text rows there were Linear-bias rows zeroed by
        // the post-projection MaskRows; here they are never computed.
        Tensor xProjImg = new Tensor(new TensorShape(batch, numImg, emb), DType.F32);
        backend.Linear(xProjImg, imageTokens, _inputProjW!, _inputProjB);
        Tensor h;
        if (numText > 0)
        {
            h = new Tensor(hidShape, DType.F32);
            backend.ScatterRowsAfter(h, xProjImg, numText);
            xProjImg.Dispose();
        }
        else
        {
            h = xProjImg;
        }
        Ideogram4DebugDump.Dump("input_proj", h);

        // ── Text embedding: step-invariant llm_cond_proj(llm_cond_norm(text rows)), cached across steps. ──
        if (numText > 0)
        {
            Tensor llmProj = GetOrBuildLlmProj(backend, llmTextFeatures!, batch, numText, numImg, seqLen, emb);
            Tensor hSum = new Tensor(hidShape, DType.F32);
            backend.Add(hSum, h, llmProj);
            h.Dispose();
            h = hSum;
        }
        Tensor indicatorIdx = BuildIndicatorIndices(indicator, batch, seqLen);
        backend.IndexAddRows(h, _imageIndicatorW!, indicatorIdx);
        indicatorIdx.Dispose();
        Ideogram4DebugDump.Dump("hidden_in", h);

        // ── Shared AdaLN conditioning: adaln_input = silu(adaln_proj(t_embedding(t))) ──
        Tensor adalnInput = ComputeAdalnInput(backend, timestep, batch);
        Ideogram4DebugDump.Dump("adaln_input", adalnInput);

        // ── RoPE (step-invariant, cached keyed on the positionIds reference) ──
        (Tensor cos, Tensor sin) = GetOrBuildRope(backend, positionIds);

        // ── Blocks ──
        // F16 hot path (HARTSY_DIT_F16): cast once at the loop boundary; the blocks follow the input dtype.
        // Masked (regional) passes stay F32 — the additive-mask SDPA path is F32-only.
        bool f16Blocks = DiTBlocks.DitDtype.Act == DType.F16 && attentionMask is null;
        Tensor cur = f16Blocks ? Utilities.DtypeCastHelper.EnsureDtype(backend, h, DType.F16) : h;

        // Across-step First-Block cache (QwenImageTransformer wiring): block 0 always runs and its output is
        // the gate indicator; hit ⇒ blocks 1..N−1 replaced by block0 + previous residual; miss ⇒ the anchor
        // (block-0 output) survives the loop so the fresh residual can be stored. Null stepCache is untouched.
        Tensor? cacheAnchor = null;
        int startBlock = 0;
        if (stepCache is not null && _blocks.Length > 1)
        {
            BeforeBlockForward?.Invoke(0);
            Tensor block0 = _blocks[0].Forward(backend, cur, adalnInput, cos, sin, attentionMask);
            cur.Dispose();
            cur = block0;
            Ideogram4DebugDump.Dump("layers.0", cur);
            startBlock = 1;
            if (!stepCache.ShouldCompute(backend, cur))
            {
                Tensor reconstructed = stepCache.ApplyResidual(backend, cur);
                cur.Dispose();
                cur = reconstructed;
                startBlock = _blocks.Length;
            }
            else
            {
                cacheAnchor = cur;
            }
        }

        for (int i = startBlock; i < _blocks.Length; i++)
        {
            // A step-cache HIT sets startBlock = _blocks.Length, so this loop (and its streaming hook) is skipped
            // entirely — the controller simply keeps whatever it had prefetched, bounded by the prefetch window,
            // and the next miss re-primes from wherever it left off. Correct, because residency is per-block state,
            // not a position in a sequence.
            BeforeBlockForward?.Invoke(i);
            Tensor next = _blocks[i].Forward(backend, cur, adalnInput, cos, sin, attentionMask);
            if (cur != cacheAnchor) cur.Dispose();
            cur = next;
            // No per-block sync: transient allocs/frees go through the stream-ordered async pool
            // (cuMemAllocAsync/FreeAsync on the compute stream), which reuses freed memory in stream
            // order WITHOUT a host sync. A per-block cuStreamSynchronize here both killed host/device
            // overlap AND, with the pool's release-threshold at 0, released the working set to the OS
            // every block (re-reserved next block) — a uniform ~20-200x slowdown across all ops.
            Ideogram4DebugDump.Dump($"layers.{i}", cur);
        }

        if (cacheAnchor is not null)
        {
            stepCache!.StoreResidual(backend, cacheAnchor, cur);
            cacheAnchor.Dispose();
        }

        // ── Final layer over the image rows only (rowwise ops, so slicing first is bit-identical) ──
        cur = Utilities.DtypeCastHelper.EnsureF32(backend, cur);
        Tensor imgHidden;
        if (numText > 0)
        {
            imgHidden = new Tensor(new TensorShape(batch, numImg, emb), DType.F32);
            backend.SliceRows(imgHidden, cur, numText);
            cur.Dispose();
        }
        else
        {
            imgHidden = cur;
        }
        Tensor outVel = ApplyFinalLayer(backend, imgHidden, adalnInput, batch, numImg, emb);
        imgHidden.Dispose();
        adalnInput.Dispose();
        Ideogram4DebugDump.DumpOutput(outVel);
        return outVel;
    }

    /// <summary>Builds (or returns the cached) full-sequence text-conditioning projection <c>[B, L, emb]</c>:
    /// <c>[llm_cond_proj(llm_cond_norm(textRows)) | zeros(numImg)]</c>. Computed over the ~2% of rows that are
    /// text instead of the whole sequence, once per prompt instead of once per forward, and pinned on-device so
    /// per-step FreeActivations cannot evict it.</summary>
    private Tensor GetOrBuildLlmProj(IBackend backend, Tensor llmTextFeatures, int batch, int numText,
        int numImg, int seqLen, int emb)
    {
        if (_cachedLlmProj is not null && ReferenceEquals(_cachedLlmProjKey, llmTextFeatures)
            && (int)_cachedLlmProj.Shape[1] == seqLen)
        {
            return _cachedLlmProj;
        }
        if (_cachedLlmProj is not null)
        {
            backend.FreeWeights(new[] { _cachedLlmProj });
            _cachedLlmProj.Dispose();
            _cachedLlmProj = null;
        }

        Tensor llmNorm = new Tensor(new TensorShape(batch, numText, _config.LlmFeaturesDim), DType.F32);
        backend.RmsNorm(llmNorm, llmTextFeatures, _llmCondNormW!, 1e-6f);
        Tensor projText = new Tensor(new TensorShape(batch, numText, emb), DType.F32);
        backend.Linear(projText, llmNorm, _llmCondProjW!, _llmCondProjB);
        llmNorm.Dispose();

        Tensor zerosTail = new Tensor(new TensorShape(batch, numImg, emb), DType.F32);
        backend.Fill(zerosTail, 0f);
        Tensor full = new Tensor(new TensorShape(batch, seqLen, emb), DType.F32);
        backend.Concat(full, new[] { projText, zerosTail }, dim: 1);
        projText.Dispose();
        zerosTail.Dispose();

        backend.PreloadWeights(new[] { full });
        _cachedLlmProj = full;
        _cachedLlmProjKey = llmTextFeatures;
        return full;
    }

    /// <summary>Builds (or returns the cached) MRoPE cos/sin tables for <paramref name="positionIds"/> — a host
    /// trig loop over B·L·headDim that is position-only, so one build serves every step. Pinned on-device.</summary>
    private (Tensor Cos, Tensor Sin) GetOrBuildRope(IBackend backend, Tensor positionIds)
    {
        if (_cachedCos is not null && _cachedSin is not null && ReferenceEquals(_cachedRopeKey, positionIds))
            return (_cachedCos, _cachedSin);
        if (_cachedCos is not null || _cachedSin is not null)
        {
            backend.FreeWeights(new[] { _cachedCos!, _cachedSin! });
            _cachedCos?.Dispose();
            _cachedSin?.Dispose();
            _cachedCos = _cachedSin = null;
        }

        (Tensor cos, Tensor sin) = _rope.BuildCosSin(positionIds);
        backend.PreloadWeights(new[] { cos, sin });
        _cachedCos = cos;
        _cachedSin = sin;
        _cachedRopeKey = positionIds;
        return (cos, sin);
    }

    /// <summary>Final layer: <c>linear(norm_final(x) * (1 + adaln(silu(c))))</c>. <c>norm_final</c> is non-affine LayerNorm (eps 1e-6); modulation is scale-only.</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor x, Tensor adalnInput, int batch, int seqLen, int emb)
    {
        // scale = 1 + adaln_modulation(silu(adaln_input))  — note the EXTRA silu (adaln_input is already silu'd).
        Tensor silu = new Tensor(adalnInput.Shape, DType.F32);
        backend.Silu(silu, adalnInput);
        Tensor scaleRaw = new Tensor(new TensorShape(batch, emb), DType.F32);
        backend.Linear(scaleRaw, silu, _finalAdalnW!, _finalAdalnB);
        silu.Dispose();

        Tensor scalePlus1 = new Tensor(new TensorShape(batch, emb), DType.F32);
        backend.AddScalar(scalePlus1, scaleRaw, 1.0f);
        scaleRaw.Dispose();

        Tensor normed = new Tensor(new TensorShape(batch, seqLen, emb), DType.F32);
        backend.LayerNormNoAffine(normed, x, 1e-6f);

        Tensor modulated = new Tensor(new TensorShape(batch, seqLen, emb), DType.F32);
        backend.AffineBroadcastLastDim(modulated, normed, scalePlus1, null);
        normed.Dispose();
        scalePlus1.Dispose();

        Tensor outVel = new Tensor(new TensorShape(batch, seqLen, _config.InChannels), DType.F32);
        backend.Linear(outVel, modulated, _finalLinearW!, _finalLinearB);
        modulated.Dispose();
        return outVel;
    }

    /// <summary><c>adaln_input = silu(adaln_proj(t_embedding(t)))</c>, shape <c>[B, adalnDim]</c>.</summary>
    private Tensor ComputeAdalnInput(IBackend backend, float timestep, int batch)
    {
        int emb = _config.EmbDim;
        Tensor sinEmb = new Tensor(new TensorShape(batch, emb), DType.F32);
        BuildSinusoidal(sinEmb, timestep, batch, emb);

        Tensor m1 = new Tensor(new TensorShape(batch, emb), DType.F32);
        backend.Linear(m1, sinEmb, _tMlpInW!, _tMlpInB);
        sinEmb.Dispose();
        Tensor m1Act = new Tensor(new TensorShape(batch, emb), DType.F32);
        backend.Silu(m1Act, m1);
        m1.Dispose();
        Tensor tCond = new Tensor(new TensorShape(batch, emb), DType.F32);
        backend.Linear(tCond, m1Act, _tMlpOutW!, _tMlpOutB);
        m1Act.Dispose();

        Tensor proj = new Tensor(new TensorShape(batch, _config.AdalnDim), DType.F32);
        backend.Linear(proj, tCond, _adalnProjW!, _adalnProjB);
        tCond.Dispose();
        Tensor adaln = new Tensor(new TensorShape(batch, _config.AdalnDim), DType.F32);
        backend.Silu(adaln, proj);
        proj.Dispose();
        return adaln;
    }

    /// <summary>Sinusoidal scalar embedding matching upstream <c>_sinusoidal_embedding</c>: <c>scaled = 1e4·t</c>, <c>freq_k = exp(-k·ln(1e4)/(half-1))</c>, layout <c>[sin(scaled·freq), cos(scaled·freq)]</c>. Same timestep for every batch row.</summary>
    private static void BuildSinusoidal(Tensor output, float timestep, int batch, int dim)
    {
        int half = dim / 2;
        float scaled = 1e4f * timestep;
        float logScale = MathF.Log(1e4f) / (half - 1);
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int baseOff = b * dim;
            for (int k = 0; k < half; k++)
            {
                float freq = MathF.Exp(-k * logScale);
                float angle = scaled * freq;
                outPtr[baseOff + k] = MathF.Sin(angle);
                outPtr[baseOff + half + k] = MathF.Cos(angle);
            }
        }
    }

    /// <summary>Builds per-token I32 indices into <c>embed_image_indicator</c> (row 1 for image tokens, row 0 otherwise),
    /// shape <c>[B*L]</c>, for <see cref="IBackend.IndexAddRows"/>.</summary>
    private static Tensor BuildIndicatorIndices(int[] indicator, int batch, int seqLen)
    {
        Tensor idx = new Tensor(new TensorShape(batch * seqLen), DType.I32);
        int* p = (int*)idx.DataPointer;
        for (int pos = 0; pos < batch * seqLen; pos++)
            p[pos] = indicator[pos] == OutputImageIndicator ? 1 : 0;
        return idx;
    }

    /// <summary>Drops tensor references (underlying weight storage is owned by the mmap loader) and releases the
    /// pinned step-invariant caches.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            try
            {
                if (_cacheBackend is not null)
                {
                    if (_cachedLlmProj is not null) _cacheBackend.FreeWeights(new[] { _cachedLlmProj });
                    if (_cachedCos is not null && _cachedSin is not null) _cacheBackend.FreeWeights(new[] { _cachedCos, _cachedSin });
                }
            }
            catch (Exception ex)
            {
                HartsyInference.Core.Logging.Logs.Error($"Ideogram4Transformer cache release failed: {ex.Message}");
            }
            _cachedLlmProj?.Dispose();
            _cachedCos?.Dispose();
            _cachedSin?.Dispose();
            _cachedLlmProj = _cachedCos = _cachedSin = null;
            _cachedLlmProjKey = _cachedRopeKey = null;
            _cacheBackend = null;
            _inputProjW = _inputProjB = null;
            _llmCondNormW = null;
            _llmCondProjW = _llmCondProjB = null;
            _tMlpInW = _tMlpInB = _tMlpOutW = _tMlpOutB = null;
            _adalnProjW = _adalnProjB = null;
            _imageIndicatorW = null;
            _finalLinearW = _finalLinearB = null;
            _finalAdalnW = _finalAdalnB = null;
        }
        GC.SuppressFinalize(this);
    }
}
