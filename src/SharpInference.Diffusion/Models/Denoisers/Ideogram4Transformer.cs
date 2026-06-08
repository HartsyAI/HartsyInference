using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

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

        _llmCondNormW = LoadAsF32(weights, "llm_cond_norm.weight");
        _llmCondProjW = weights["llm_cond_proj.weight"];
        weights.TryGetValue("llm_cond_proj.bias", out _llmCondProjB);

        _tMlpInW = weights["t_embedding.mlp_in.weight"];
        weights.TryGetValue("t_embedding.mlp_in.bias", out _tMlpInB);
        _tMlpOutW = weights["t_embedding.mlp_out.weight"];
        weights.TryGetValue("t_embedding.mlp_out.bias", out _tMlpOutB);

        _adalnProjW = weights["adaln_proj.weight"];
        weights.TryGetValue("adaln_proj.bias", out _adalnProjB);

        _imageIndicatorW = LoadAsF32(weights, "embed_image_indicator.weight");

        _finalLinearW = weights["final_layer.linear.weight"];
        weights.TryGetValue("final_layer.linear.bias", out _finalLinearB);
        _finalAdalnW = weights["final_layer.adaln_modulation.weight"];
        weights.TryGetValue("final_layer.adaln_modulation.bias", out _finalAdalnB);

        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"layers.{i}");
    }

    /// <summary>Enumerates every weight tensor for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
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
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Velocity prediction over a unified text+image sequence.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="llmFeatures">Qwen3-VL conditioning <c>[B, L, llmFeaturesDim]</c> (placed at text positions; image positions ignored).</param>
    /// <param name="x">Noise tokens <c>[B, L, inChannels]</c> (image positions hold noise; text positions ignored).</param>
    /// <param name="timestep">Flow-matching time in [0,1], shared across the batch.</param>
    /// <param name="positionIds">MRoPE positions <c>[B, L, 3]</c> F32 (temporal, height, width).</param>
    /// <param name="indicator">Per-token role, length <c>B*L</c> row-major: <c>LLM_TOKEN_INDICATOR(3)</c> / <c>OUTPUT_IMAGE_INDICATOR(2)</c> / padding(−1).</param>
    /// <param name="attentionMask">Optional additive mask <c>[B, 1, L, L]</c> or <c>[B, 1, 1, L]</c>; null = full attention (correct for unpadded single-prompt B=1).</param>
    /// <returns><c>[B, L, inChannels]</c> velocity (F32). Only image-token positions are meaningful.</returns>
    public Tensor Forward(IBackend backend, Tensor llmFeatures, Tensor x, float timestep,
        Tensor positionIds, int[] indicator, Tensor? attentionMask)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        int emb = _config.EmbDim;
        TensorShape hidShape = new TensorShape(batch, seqLen, emb);

        if (indicator.Length != batch * seqLen)
            throw new ArgumentException($"indicator length {indicator.Length} != B*L {batch * seqLen}.", nameof(indicator));

        // ── Masked image embedding: x *= image_mask; input_proj(x); *= image_mask ──
        Tensor xMasked = ApplyTokenMask(x, indicator, OutputImageIndicator, batch, seqLen, _config.InChannels);
        Tensor xProjRaw = new Tensor(hidShape, DType.F32);
        backend.Linear(xProjRaw, xMasked, _inputProjW!, _inputProjB);
        xMasked.Dispose();
        Tensor xProj = ApplyTokenMask(xProjRaw, indicator, OutputImageIndicator, batch, seqLen, emb);
        xProjRaw.Dispose();
        Ideogram4DebugDump.Dump("input_proj", xProj);

        // ── Masked text embedding: llm_cond_proj(llm_cond_norm(llm_features)); *= text_mask ──
        Tensor llmMasked = ApplyTokenMask(llmFeatures, indicator, LlmTokenIndicator, batch, seqLen, _config.LlmFeaturesDim);
        Tensor llmNorm = new Tensor(new TensorShape(batch, seqLen, _config.LlmFeaturesDim), DType.F32);
        backend.RmsNorm(llmNorm, llmMasked, _llmCondNormW!, 1e-6f);
        llmMasked.Dispose();
        Tensor llmProjRaw = new Tensor(hidShape, DType.F32);
        backend.Linear(llmProjRaw, llmNorm, _llmCondProjW!, _llmCondProjB);
        llmNorm.Dispose();
        Tensor llmProj = ApplyTokenMask(llmProjRaw, indicator, LlmTokenIndicator, batch, seqLen, emb);
        llmProjRaw.Dispose();

        // ── h = image_emb + text_emb + image_indicator_embedding ──
        Tensor h = new Tensor(hidShape, DType.F32);
        backend.Add(h, xProj, llmProj);
        xProj.Dispose();
        llmProj.Dispose();
        AddImageIndicator(h, indicator, batch, seqLen, emb);
        Ideogram4DebugDump.Dump("hidden_in", h);

        // ── Shared AdaLN conditioning: adaln_input = silu(adaln_proj(t_embedding(t))) ──
        Tensor adalnInput = ComputeAdalnInput(backend, timestep, batch);
        Ideogram4DebugDump.Dump("adaln_input", adalnInput);

        // ── RoPE ──
        (Tensor cos, Tensor sin) = _rope.BuildCosSin(positionIds);

        // ── Blocks ──
        Tensor cur = h;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, adalnInput, cos, sin, _rope, attentionMask);
            cur.Dispose();
            cur = next;
            Ideogram4DebugDump.Dump($"layers.{i}", cur);
        }
        cos.Dispose();
        sin.Dispose();

        // ── Final layer ──
        Tensor outVel = ApplyFinalLayer(backend, cur, adalnInput, batch, seqLen, emb);
        cur.Dispose();
        adalnInput.Dispose();
        Ideogram4DebugDump.DumpOutput(outVel);
        return outVel;
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

        Tensor normed = new Tensor(new TensorShape(batch, seqLen, emb), DType.F32);
        DiTUtils.LayerNormNoAffine(normed, x, batch, seqLen, emb, 1e-6f);

        Tensor modulated = new Tensor(new TensorShape(batch, seqLen, emb), DType.F32);
        float* normPtr = (float*)normed.DataPointer;
        float* scalePtr = (float*)scaleRaw.DataPointer;
        float* outPtr = (float*)modulated.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int condBase = b * emb;
            for (int s = 0; s < seqLen; s++)
            {
                int rowOff = (b * seqLen + s) * emb;
                for (int d = 0; d < emb; d++)
                    outPtr[rowOff + d] = normPtr[rowOff + d] * (1.0f + scalePtr[condBase + d]);
            }
        }
        normed.Dispose();
        scaleRaw.Dispose();

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

    /// <summary>Zeroes channels at positions whose indicator != <paramref name="keepRole"/>; copies through otherwise. Returns a new tensor.</summary>
    private static Tensor ApplyTokenMask(Tensor input, int[] indicator, int keepRole, int batch, int seqLen, int channels)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, channels), input.DType);
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int pos = b * seqLen + s;
                long rowOff = (long)pos * channels;
                if (indicator[pos] == keepRole)
                    Buffer.MemoryCopy(inPtr + rowOff, outPtr + rowOff, (long)channels * sizeof(float), (long)channels * sizeof(float));
                else
                    new Span<float>(outPtr + rowOff, channels).Clear();
            }
        }
        return output;
    }

    /// <summary>Adds the <c>embed_image_indicator</c> row in-place: row 1 for image tokens, row 0 for everything else.</summary>
    private void AddImageIndicator(Tensor h, int[] indicator, int batch, int seqLen, int emb)
    {
        float* hPtr = (float*)h.DataPointer;
        float* wPtr = (float*)_imageIndicatorW!.DataPointer; // [2, emb]
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int pos = b * seqLen + s;
                int row = indicator[pos] == OutputImageIndicator ? 1 : 0;
                float* src = wPtr + (long)row * emb;
                float* dst = hPtr + (long)pos * emb;
                for (int d = 0; d < emb; d++) dst[d] += src[d];
            }
        }
    }

    private static Tensor LoadAsF32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor t = weights[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }

    /// <summary>Drops tensor references. Underlying unmanaged storage is owned by the mmap loader.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
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
