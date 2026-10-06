using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.DiT;
using HartsyInference.Audio.Models.Moonshine;   // RotaryEmbedding (interleaved convention)
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>IndexTTS-2's S2Mel flow-matching DiT — the <see cref="ICfmEstimator"/> velocity network
/// <see cref="HartsyInference.Audio.Models.CosyVoice.ConditionalCfm"/> solves. Real source
/// (<c>diffusion_transformer.py:DiT.forward</c>, confirmed from the real checkpoint's own key names/shapes —
/// see <see cref="IndexTts2DitConfig"/>'s remarks):
/// <code>
/// t1 = t_embedder(t)                                              # [1, hidden]
/// cond = cond_projection(mu)                                      # mu: length-regulated content [1,T,contentDim] -> [1,T,hidden]
/// x_in = cat([x.transpose(1,2), prompt_x.transpose(1,2), cond], dim=-1)   # [1,T, inCh+inCh+hidden]
/// x_in = cat([x_in, style.repeat(1,T,1)], dim=-1)                  # [1,T, +styleDim]
/// x_in = cond_x_merge_linear(x_in)                                 # -> [1,T,hidden]
/// x_res = transformer(x_in, t1)   # 13 blocks, U-ViT skip between the symmetric halves, final AdaptiveLayerNorm
/// x_res = skip_linear(cat([x_res, x.transpose(1,2)], dim=-1))       # long residual
/// h = conv1(x_res)                                                 # Linear hidden->wavenetHidden
/// t2 = t_embedder2(t)
/// h = wavenet(h.transpose(1,2), g=t2) .transpose(1,2) + res_projection(x_res)
/// h = final_layer(h, t1)             # AdaLN-ZERO (SiLU-gated), parameter-free LayerNorm, eps=1e-6
/// out = conv2(h.transpose(1,2))                                    # Conv1d wavenetHidden->inChannels, k=1
/// </code>
/// <see cref="HartsyInference.Audio.Models.CosyVoice.ICfmEstimator"/>'s <c>mu</c> slot carries the
/// length-regulated semantic content (what the real Python calls <c>cond</c> at the DiT level — the naming
/// differs only because <c>BASECFM.solve_euler</c>'s own parameter is named <c>mu</c>); its <c>cond</c> slot
/// carries the zero-padded reference-mel prefix (real <c>prompt_x</c>). <c>cond_embedder</c> (a discrete-code
/// embedding) and <c>content_mask_embedder</c> are confirmed dead for this inference path (the real
/// <c>forward()</c> hardcodes <c>cond_in_module = self.cond_projection</c>, and <c>mask_content</c> is never
/// set True by <c>infer_v2_5.py</c>) — not loaded or wired here, same bar this project applies to every other
/// confirmed-unused checkpoint tensor.</summary>
internal sealed unsafe class IndexTts2Dit : ICfmEstimator, IDisposable
{
    private const int MaxPos = 8_192;   // real config.yaml: s2mel.DiT.block_size

    private readonly IndexTts2DitConfig _cfg;
    private readonly IndexTts2DitBlock[] _blocks;
    private readonly IndexTts2TimestepEmbedder _tEmbedder, _tEmbedder2;
    private readonly AdaptiveLayerNorm _finalTransformerNorm;
    private readonly VitsWaveNet _wavenet;

    private Tensor? _condProjW, _condProjB;
    private Tensor? _condXMergeW, _condXMergeB;
    private Tensor? _skipLinearW, _skipLinearB;
    private Tensor? _conv1W, _conv1B;
    private Tensor? _conv2W, _conv2B;
    private Tensor? _resProjW, _resProjB;
    private Tensor? _finalLayerAdaLnW, _finalLayerAdaLnB;
    private Tensor? _finalLayerLinearW, _finalLayerLinearB;

    private Tensor? _ropeCos, _ropeSin;

    public IndexTts2Dit(IndexTts2DitConfig cfg)
    {
        _cfg = cfg;
        _blocks = new IndexTts2DitBlock[cfg.Depth];
        for (int i = 0; i < cfg.Depth; i++) _blocks[i] = new IndexTts2DitBlock(cfg.HiddenDim, cfg.NumHeads, cfg.FfnDim);
        _tEmbedder = new IndexTts2TimestepEmbedder(cfg.HiddenDim, cfg.FreqEmbedSize);
        _tEmbedder2 = new IndexTts2TimestepEmbedder(cfg.WavenetHiddenDim, cfg.FreqEmbedSize);
        _finalTransformerNorm = new AdaptiveLayerNorm(cfg.HiddenDim);
        _wavenet = new VitsWaveNet(cfg.WavenetHiddenDim, cfg.WavenetKernelSize, cfg.WavenetDilationRate, cfg.WavenetNumLayers);
    }

    /// <param name="prefix">e.g. <c>net.cfm.estimator</c> — the real checkpoint's own module path.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(w, $"{prefix}.transformer.layers.{i}");
        _finalTransformerNorm.LoadWeights(
            WhisperOps.EnsureF32(w[$"{prefix}.transformer.norm.norm.weight"]),
            WhisperOps.EnsureF32(w[$"{prefix}.transformer.norm.project_layer.weight"]),
            WhisperOps.EnsureF32(w[$"{prefix}.transformer.norm.project_layer.bias"]));

        _tEmbedder.LoadWeights(w, $"{prefix}.t_embedder");
        _tEmbedder2.LoadWeights(w, $"{prefix}.t_embedder2");

        _condProjW = WhisperOps.EnsureF32(w[$"{prefix}.cond_projection.weight"]);
        _condProjB = WhisperOps.EnsureF32(w[$"{prefix}.cond_projection.bias"]);
        _condXMergeW = WhisperOps.EnsureF32(w[$"{prefix}.cond_x_merge_linear.weight"]);
        _condXMergeB = WhisperOps.EnsureF32(w[$"{prefix}.cond_x_merge_linear.bias"]);
        _skipLinearW = WhisperOps.EnsureF32(w[$"{prefix}.skip_linear.weight"]);
        _skipLinearB = WhisperOps.EnsureF32(w[$"{prefix}.skip_linear.bias"]);

        _conv1W = WhisperOps.EnsureF32(w[$"{prefix}.conv1.weight"]);
        _conv1B = WhisperOps.EnsureF32(w[$"{prefix}.conv1.bias"]);
        _conv2W = WhisperOps.EnsureF32(w[$"{prefix}.conv2.weight"]);
        _conv2B = WhisperOps.EnsureF32(w[$"{prefix}.conv2.bias"]);
        _resProjW = WhisperOps.EnsureF32(w[$"{prefix}.res_projection.weight"]);
        _resProjB = WhisperOps.EnsureF32(w[$"{prefix}.res_projection.bias"]);

        _finalLayerAdaLnW = WhisperOps.EnsureF32(w[$"{prefix}.final_layer.adaLN_modulation.1.weight"]);
        _finalLayerAdaLnB = WhisperOps.EnsureF32(w[$"{prefix}.final_layer.adaLN_modulation.1.bias"]);
        _finalLayerLinearW = HartsyInference.Audio.Models.Codecs.WeightNormFusion.Compose(w, $"{prefix}.final_layer.linear");
        _finalLayerLinearB = WhisperOps.EnsureF32(w[$"{prefix}.final_layer.linear.bias"]);

        _wavenet.LoadWeights(w, $"{prefix}.wavenet", convKeySuffix: ".conv.conv");

        (float[] cos, float[] sin) = RotaryEmbedding.GetTables(_cfg.HeadDim, _cfg.RopeBase, MaxPos);
        int half = _cfg.HeadDim / 2;
        _ropeCos = new Tensor(new TensorShape(MaxPos, _cfg.HeadDim), DType.F32);
        _ropeSin = new Tensor(new TensorShape(MaxPos, _cfg.HeadDim), DType.F32);
        float* cp = (float*)_ropeCos.DataPointer;
        float* sp = (float*)_ropeSin.DataPointer;
        for (int p = 0; p < MaxPos; p++)
        {
            for (int i = 0; i < half; i++)
            {
                int i0 = 2 * i;
                cp[p * _cfg.HeadDim + i0] = cp[p * _cfg.HeadDim + i0 + 1] = cos[p * half + i];
                sp[p * _cfg.HeadDim + i0] = sp[p * _cfg.HeadDim + i0 + 1] = sin[p * half + i];
            }
        }
    }

    /// <summary><paramref name="x"/>/<paramref name="cond"/> are <c>[1, inChannels, T]</c> (channel-first:
    /// the noisy mel and the zero-padded reference-mel prefix, respectively); <paramref name="mu"/> is
    /// <c>[1, T, contentDim]</c> (channel-last: the length-regulated semantic content); <paramref name="spk"/>
    /// is <c>[1, styleDim]</c>. <paramref name="attnMask"/> is ignored (always full bidirectional attention —
    /// confirmed from the real <c>setup_caches(..., use_kv_cache=False)</c> and non-causal config).</summary>
    public Tensor Estimate(IBackend backend, Tensor x, Tensor mu, float t, Tensor spk, Tensor cond, Tensor? attnMask = null)
    {
        int timeLen = (int)x.Shape[2];
        int hidden = _cfg.HiddenDim, inCh = _cfg.InChannels, contentDim = _cfg.ContentDim, styleDim = _cfg.StyleDim;

        Tensor t1 = _tEmbedder.Forward(backend, t);

        Tensor contentProjected = WhisperOps.ProjectLinear(backend, mu, _condProjW!, _condProjB, 1, timeLen, contentDim, hidden);

        Tensor xChannelLast = TransposeLastTwo(x, inCh, timeLen);
        Tensor promptChannelLast = TransposeLastTwo(cond, inCh, timeLen);

        Tensor xIn = ConcatLastDim3(xChannelLast, promptChannelLast, contentProjected, timeLen, inCh, inCh, hidden);
        contentProjected.Dispose();
        Tensor xInWithStyle = ConcatBroadcastStyle(xIn, spk, timeLen, inCh + inCh + hidden, styleDim);
        xIn.Dispose();

        Tensor current = WhisperOps.ProjectLinear(backend, xInWithStyle, _condXMergeW!, _condXMergeB, 1, timeLen, inCh + inCh + hidden + styleDim, hidden);
        xInWithStyle.Dispose();

        // U-ViT skip bookkeeping (real `Transformer.forward`): the first half of the stack pushes its own
        // output onto a stack; the second half pops (LIFO) and feeds it into `skip_in_linear`. Layer
        // `half` (the dead center, n_layer//2) neither emits nor receives. A pushed tensor is shared between
        // the stack and the loop's own `current` variable for exactly one iteration, so disposal must check
        // whether the PREVIOUS iteration already handed `current` off to the stack before freeing it here.
        List<Tensor> emitStack = [];
        int half = _cfg.Depth / 2;
        for (int i = 0; i < _cfg.Depth; i++)
        {
            Tensor? skipIn = null;
            if (i > half && emitStack.Count > 0)
            {
                skipIn = emitStack[^1];
                emitStack.RemoveAt(emitStack.Count - 1);
            }

            Tensor next = _blocks[i].Forward(backend, current, t1, timeLen, _ropeCos!, _ropeSin!, skipIn);
            skipIn?.Dispose();

            bool currentWasPushed = i > 0 && (i - 1) < half;
            if (!currentWasPushed) current.Dispose();

            current = next;
            if (i < half) emitStack.Add(current);
        }
        foreach (Tensor leftover in emitStack) leftover.Dispose();   // defensive; stack should be empty here.

        Tensor transformerOut = _finalTransformerNorm.Forward(backend, current, t1);
        current.Dispose();

        Tensor longSkipIn = ConcatLastDim2(transformerOut, xChannelLast, timeLen, hidden, inCh);
        transformerOut.Dispose();
        xChannelLast.Dispose();
        promptChannelLast.Dispose();
        Tensor xRes = WhisperOps.ProjectLinear(backend, longSkipIn, _skipLinearW!, _skipLinearB, 1, timeLen, hidden + inCh, hidden);
        longSkipIn.Dispose();

        Tensor hLin = WhisperOps.ProjectLinear(backend, xRes, _conv1W!, _conv1B, 1, timeLen, hidden, _cfg.WavenetHiddenDim);
        Tensor hChannelFirst = TransposeLastTwo(hLin, timeLen, _cfg.WavenetHiddenDim);
        hLin.Dispose();

        Tensor t2 = _tEmbedder2.Forward(backend, t);
        Tensor g = t2.Reshape(new TensorShape(1, _cfg.WavenetHiddenDim, 1));
        Tensor wnOut = _wavenet.Forward(backend, hChannelFirst, timeLen, g);
        hChannelFirst.Dispose();
        t2.Dispose();

        Tensor wnOutChannelLast = TransposeLastTwo(wnOut, _cfg.WavenetHiddenDim, timeLen);
        wnOut.Dispose();
        Tensor resProjected = WhisperOps.ProjectLinear(backend, xRes, _resProjW!, _resProjB, 1, timeLen, hidden, _cfg.WavenetHiddenDim);
        xRes.Dispose();

        Tensor combined = AddLastDim(wnOutChannelLast, resProjected, timeLen, _cfg.WavenetHiddenDim);
        wnOutChannelLast.Dispose();
        resProjected.Dispose();

        Tensor finalOut = FinalLayer(backend, combined, t1, timeLen);
        combined.Dispose();
        t1.Dispose();

        Tensor finalChannelFirst = TransposeLastTwo(finalOut, timeLen, _cfg.WavenetHiddenDim);
        finalOut.Dispose();

        Tensor velocity = new(new TensorShape(1, inCh, timeLen), DType.F32);
        backend.Conv1d(velocity, finalChannelFirst, _conv2W!, _conv2B, 1, 0, 0, 1, 1);
        finalChannelFirst.Dispose();
        return velocity;
    }

    /// <summary>Real <c>FinalLayer.forward</c>: AdaLN-ZERO modulation (SiLU-gated) of a parameter-free
    /// LayerNorm, then one Linear. <paramref name="h"/> is <c>[1, T, wavenetHidden]</c>; <paramref name="t1"/>
    /// is the DiT's own (not <c>t2</c>'s) timestep embedding, confirmed from the real call site
    /// <c>self.final_layer(x, t1)</c>.</summary>
    private Tensor FinalLayer(IBackend backend, Tensor h, Tensor t1, int t)
    {
        int hidden = _cfg.WavenetHiddenDim;
        Tensor siluT1 = new(t1.Shape, DType.F32);
        backend.Silu(siluT1, t1);
        Tensor mod = new(new TensorShape(1, 2 * hidden), DType.F32);
        backend.Linear(mod, siluT1, _finalLayerAdaLnW!, _finalLayerAdaLnB);
        siluT1.Dispose();

        Tensor shift = new(new TensorShape(1, hidden), DType.F32);
        Tensor scale = new(new TensorShape(1, hidden), DType.F32);
        Buffer.MemoryCopy((float*)mod.DataPointer, (float*)shift.DataPointer, (long)hidden * 4, (long)hidden * 4);
        Buffer.MemoryCopy((float*)mod.DataPointer + hidden, (float*)scale.DataPointer, (long)hidden * 4, (long)hidden * 4);
        mod.Dispose();
        backend.AddScalar(scale, scale, 1f);

        Tensor normed = new(h.Shape, DType.F32);
        backend.LayerNormNoAffine(normed, h, 1e-6f);
        Tensor modulated = new(h.Shape, DType.F32);
        backend.AffineBroadcastLastDim(modulated, normed, scale, shift);
        normed.Dispose();
        shift.Dispose();
        scale.Dispose();

        Tensor result = WhisperOps.ProjectLinear(backend, modulated, _finalLayerLinearW!, _finalLayerLinearB, 1, t, hidden, hidden);
        modulated.Dispose();
        return result;
    }

    private static Tensor TransposeLastTwo(Tensor t, int d1, int d2)
    {
        Tensor result = new(new TensorShape(1, d2, d1), DType.F32);
        float* sp = (float*)t.DataPointer;
        float* dp = (float*)result.DataPointer;
        for (int a = 0; a < d1; a++)
            for (int b = 0; b < d2; b++)
                dp[(long)b * d1 + a] = sp[(long)a * d2 + b];
        return result;
    }

    private static Tensor ConcatLastDim2(Tensor a, Tensor b, int t, int aDim, int bDim)
    {
        Tensor result = new(new TensorShape(1, t, aDim + bDim), DType.F32);
        float* ap = (float*)a.DataPointer, bp = (float*)b.DataPointer, rp = (float*)result.DataPointer;
        for (int i = 0; i < t; i++)
        {
            Buffer.MemoryCopy(ap + (long)i * aDim, rp + (long)i * (aDim + bDim), (long)aDim * 4, (long)aDim * 4);
            Buffer.MemoryCopy(bp + (long)i * bDim, rp + (long)i * (aDim + bDim) + aDim, (long)bDim * 4, (long)bDim * 4);
        }
        return result;
    }

    private static Tensor ConcatLastDim3(Tensor a, Tensor b, Tensor c, int t, int aDim, int bDim, int cDim)
    {
        int total = aDim + bDim + cDim;
        Tensor result = new(new TensorShape(1, t, total), DType.F32);
        float* ap = (float*)a.DataPointer, bp = (float*)b.DataPointer, cp = (float*)c.DataPointer, rp = (float*)result.DataPointer;
        for (int i = 0; i < t; i++)
        {
            float* row = rp + (long)i * total;
            Buffer.MemoryCopy(ap + (long)i * aDim, row, (long)aDim * 4, (long)aDim * 4);
            Buffer.MemoryCopy(bp + (long)i * bDim, row + aDim, (long)bDim * 4, (long)bDim * 4);
            Buffer.MemoryCopy(cp + (long)i * cDim, row + aDim + bDim, (long)cDim * 4, (long)cDim * 4);
        }
        return result;
    }

    /// <summary>Concats <paramref name="style"/> <c>[1, styleDim]</c>, broadcast over time, onto
    /// <paramref name="x"/>'s <c>[1, T, xDim]</c> last dim.</summary>
    private static Tensor ConcatBroadcastStyle(Tensor x, Tensor style, int t, int xDim, int styleDim)
    {
        int total = xDim + styleDim;
        Tensor result = new(new TensorShape(1, t, total), DType.F32);
        float* xp = (float*)x.DataPointer, sp = (float*)style.DataPointer, rp = (float*)result.DataPointer;
        for (int i = 0; i < t; i++)
        {
            float* row = rp + (long)i * total;
            Buffer.MemoryCopy(xp + (long)i * xDim, row, (long)xDim * 4, (long)xDim * 4);
            Buffer.MemoryCopy(sp, row + xDim, (long)styleDim * 4, (long)styleDim * 4);
        }
        return result;
    }

    private static Tensor AddLastDim(Tensor a, Tensor b, int t, int dim)
    {
        Tensor result = new(a.Shape, DType.F32);
        float* ap = (float*)a.DataPointer, bp = (float*)b.DataPointer, rp = (float*)result.DataPointer;
        long n = (long)t * dim;
        for (long i = 0; i < n; i++) rp[i] = ap[i] + bp[i];
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (IndexTts2DitBlock block in _blocks) foreach (Tensor t in block.EnumerateWeights()) yield return t;
        foreach (Tensor t in _finalTransformerNorm.EnumerateWeights()) yield return t;
        foreach (Tensor t in _tEmbedder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _tEmbedder2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _wavenet.EnumerateWeights()) yield return t;
        Tensor?[] core =
        [
            _condProjW, _condProjB, _condXMergeW, _condXMergeB, _skipLinearW, _skipLinearB,
            _conv1W, _conv1B, _conv2W, _conv2B, _resProjW, _resProjB,
            _finalLayerAdaLnW, _finalLayerAdaLnB, _finalLayerLinearW, _finalLayerLinearB,
        ];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        if (_ropeCos is not null) yield return _ropeCos;
        if (_ropeSin is not null) yield return _ropeSin;
    }

    private int _disposed;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in EnumerateWeights()) t.Dispose();
        GC.SuppressFinalize(this);
    }
}
