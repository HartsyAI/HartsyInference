using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
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

    // RoPE cos/sin depend only on the latent grid + interp scale, which are fixed for a whole generation — but Forward
    // is called 2×/step (CFG cond+uncond). Cache them so BuildCosSin (a host loop that then uploads [S,dim] cos/sin)
    // runs ONCE per gen instead of ~2·steps times, and the device upload stays cache-hot across steps.
    private Tensor? _cachedCos, _cachedSin;
    private (int F, int H, int W, double T, double IH, double IW) _cosKey = (-1, -1, -1, 0, 0, 0);

    // ── Step-graph state (HARTSY_DIT_GRAPH; see ForwardPaired) ──────────────────────────────────────────
    // The captured body (proj_in → 28 blocks → final norm → proj_out, for the cond and uncond passes) reads ONLY
    // fixed device buffers and writes the velocity buffers. Everything per-step-varying is refreshed into a fixed
    // buffer OUTSIDE the capture: the latent (in-place device CfgEulerStep by the pipeline + PrepareGraphLatent),
    // and the tiny timestep tensors temb6/shift/scaleP1 (the existing host TimeEmbed/final-scale build runs outside
    // capture, then CopyInto). The projected encoders are step-invariant → computed once on call 1 and held.
    private Tensor? _latentFixed;
    private Tensor? _temb6Fixed, _shiftFixed, _scaleP1Fixed;
    private Tensor? _encProjCondFixed, _encProjUncondFixed;
    private Tensor? _graphVelCond, _graphVelUncond;
    private long _graphSig = long.MinValue;   // grid ⊕ interp ⊕ cond/uncond encoder identity the capture is valid for
    private int _graphSigCalls;               // calls at the current sig (capture on the 3rd — caches/promotions warm)
    private int _graphSigFlips;               // sig alternation counter (safety: disable if it never converges)
    private bool _graphDead;                  // permanent per-session fallback to eager
    private const int GraphCaptureCall = 3;

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
        (int Frames, int Height, int Width) grid, (double T, double H, double W) interpScale, Tensor? encoderMask,
        Utilities.DeviceFeatureCache? stepCache = null)
    {
        int s = (int)latentTokens.Shape[0];
        int dim = _config.InnerDim;

        (Tensor cos, Tensor sin) = GetCosSin(grid, interpScale);

        Tensor hidden = new Tensor(new TensorShape(s, dim), DType.F32);
        backend.Linear(hidden, latentTokens, _projInW!, _projInB);
        LtxVideoDebugDump.Dump("proj_in", hidden);

        (Tensor temb6, Tensor embedded) = TimeEmbed(backend, timestep);
        Tensor encoderProj = CaptionProject(backend, encoder);

        Tensor cur = hidden;
        // Across-step First-Block cache (single-stream FBC — WanVideoTransformer holds the reference video
        // wiring): block 0 always runs and gates; hit ⇒ blocks 1..N−1 replaced by the cached residual;
        // miss ⇒ the block-0 output anchors StoreResidual. Null ⇒ byte-identical loop. Callers must force
        // the eager path when a cache is armed (variable per-step topology can't replay a step graph).
        Tensor? cacheAnchor = null;
        int startBlock = 0;
        if (stepCache is not null && _blocks.Length > 1)
        {
            Tensor block0 = _blocks[0].Forward(backend, cur, encoderProj, temb6, _rope, cos, sin, encoderMask);
            cur.Dispose();
            cur = block0;
            LtxVideoDebugDump.Dump("blocks.0", cur);
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
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, temb6, _rope, cos, sin, encoderMask);
            if (!ReferenceEquals(cur, cacheAnchor)) cur.Dispose();
            cur = next;
            LtxVideoDebugDump.Dump($"blocks.{i}", cur);
        }
        if (cacheAnchor is not null)
        {
            stepCache!.StoreResidual(backend, cacheAnchor, cur);
            cacheAnchor.Dispose();
        }
        temb6.Dispose(); encoderProj.Dispose();   // cos/sin are cached (GetCosSin) — not disposed here

        Tensor outVel = FinalLayer(backend, cur, embedded, s, dim);
        cur.Dispose();
        embedded.Dispose();
        LtxVideoDebugDump.DumpOutput(outVel);
        return outVel;
    }

    /// <summary>Copies the working latent into the transformer-owned fixed buffer the step graph captures against,
    /// returning that buffer for the pipeline to denoise in place (device <c>CfgEulerStep</c>). Call once, after the
    /// initial noise is built, ONLY when graph mode is being attempted; the returned tensor is transformer-owned
    /// (do NOT dispose it in the pipeline). Returns the original latent unchanged when graph mode is off, so the
    /// eager path allocates nothing extra.</summary>
    public Tensor PrepareGraphLatent(IBackend backend, Tensor latent)
    {
        if (!(DiTBlocks.DitStepGraph.Enabled && backend.StepGraphSupported) || _graphDead)
            return latent;
        int s = (int)latent.Shape[0], c = (int)latent.Shape[1];
        if (_latentFixed is null || _latentFixed.Shape[0] != s || _latentFixed.Shape[1] != c)
        {
            _latentFixed?.Dispose();
            _latentFixed = new Tensor(new TensorShape(s, c), DType.F32);
        }
        backend.CopyInto(_latentFixed, latent);
        return _latentFixed;
    }

    /// <summary>One denoise step for the CFG pair (cond + optional uncond), sharing the RoPE tables and — with the
    /// step graph active (<see cref="DiTBlocks.DitStepGraph"/>, opt-in via <c>HARTSY_DIT_GRAPH=1</c> and the latent
    /// routed through <see cref="PrepareGraphLatent"/>) — capturing the ENTIRE pair once per generation and replaying
    /// it with a single <c>cuGraphLaunch</c> per step, erasing the host-issue tail over LTX's ~50k ops/gen. The only
    /// per-step-varying content (timestep temb6/shift/scale) enters through fixed buffers refreshed OUTSIDE the
    /// capture; the step-invariant projected encoders are computed once on call 1. Self-disables to eager on capture
    /// failure. <paramref name="callerOwns"/> is true on the eager path (fresh velocities — caller disposes) and false
    /// on the graph path (transformer-owned fixed velocity buffers, rewritten next step).</summary>
    public (Tensor condVel, Tensor? uncondVel, bool callerOwns) ForwardPaired(
        IBackend backend, Tensor latentTokens, Tensor condEncoder, Tensor? uncondEncoder, float timestep,
        (int Frames, int Height, int Width) grid, (double T, double H, double W) interpScale, Tensor? encoderMask,
        Utilities.DeviceFeatureCache? condCache = null, Utilities.DeviceFeatureCache? uncondCache = null)
    {
        // An armed step cache forces eager: cache hits change the per-step block topology, which a captured
        // graph cannot replay (INFERENCE_ACCEL_GRIND §H1.5 — same rule as Chroma's CFG-pair graph).
        bool graphMode = condCache is null && uncondCache is null
            && DiTBlocks.DitStepGraph.Enabled && backend.StepGraphSupported && !_graphDead
            && ReferenceEquals(latentTokens, _latentFixed);
        if (!graphMode)
        {
            // Eager path — byte-identical to two calls of the validated Forward.
            Tensor condV = Forward(backend, latentTokens, condEncoder, timestep, grid, interpScale, encoderMask, condCache);
            Tensor? uncondV = uncondEncoder is not null
                ? Forward(backend, latentTokens, uncondEncoder, timestep, grid, interpScale, encoderMask, uncondCache)
                : null;
            return (condV, uncondV, true);
        }

        // ── Step-graph path ──
        // Refresh the fixed timestep buffers (the ONLY per-step-varying content the captured body reads). The host
        // sinusoid build + final scale/shift host loop run HERE, outside the capture, then land device-side via CopyInto.
        RefreshTimestepFixed(backend, timestep);

        long sig = ((long)grid.Frames * 19349663L) ^ ((long)grid.Height * 83492791L) ^ ((long)grid.Width * 2654435761L)
            ^ (BitConverter.DoubleToInt64Bits(interpScale.T) << 3) ^ BitConverter.DoubleToInt64Bits(interpScale.W)
            ^ ((long)System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(condEncoder) << 17)
            ^ (uncondEncoder is not null ? (long)System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(uncondEncoder) << 31 : 0);
        bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, this);
        if (sig != _graphSig || ownerLost)
        {
            backend.StepGraphReset();
            if (!ownerLost && _graphSig != long.MinValue && ++_graphSigFlips > 8)
            {
                _graphDead = true;
                Logs.Warning("[LTX graph] signature flip storm — step-graph disabled for this session.");
            }
            _graphSig = sig;
            _graphSigCalls = 0;
            backend.StepGraphOwner = this;
        }
        else if (_graphSigCalls > GraphCaptureCall && !backend.StepGraphReady)
        {
            // Backend graph slot reset externally (FreeActivations) while our signature stayed valid — re-warm.
            _graphSigCalls = 0;
        }
        _graphSigCalls++;
        if (_graphSigCalls == 1)
        {
            // Project the step-invariant T5 encoders ONCE and hold them device-resident; pin them + the RoPE tables so
            // the captured body reads fixed addresses (no per-step re-upload from pageable memory would invalidate capture).
            _encProjCondFixed?.Dispose();
            _encProjCondFixed = CaptionProject(backend, condEncoder);
            _encProjUncondFixed?.Dispose();
            _encProjUncondFixed = uncondEncoder is not null ? CaptionProject(backend, uncondEncoder) : null;
            (Tensor cos, Tensor sin) = GetCosSin(grid, interpScale);
            List<Tensor> pins = new List<Tensor> { _encProjCondFixed, cos, sin };
            if (_encProjUncondFixed is not null) pins.Add(_encProjUncondFixed);
            backend.PreloadWeights(pins);
        }
        int s = (int)latentTokens.Shape[0];
        _graphVelCond ??= new Tensor(new TensorShape(s, _config.OutChannels), DType.F32);
        if (uncondEncoder is not null)
            _graphVelUncond ??= new Tensor(new TensorShape(s, _config.OutChannels), DType.F32);

        if (backend.StepGraphReady && _graphSigCalls > GraphCaptureCall)
        {
            backend.StepGraphLaunch();
            return (_graphVelCond!, uncondEncoder is not null ? _graphVelUncond : null, false);
        }

        // Calls 1..GraphCaptureCall-1 run eagerly THROUGH the fixed buffers (warming caches/promotions); call
        // GraphCaptureCall records the graph.
        bool capture = _graphSigCalls == GraphCaptureCall;
        if (capture) backend.StepGraphBegin();
        try
        {
            RunPairIntoFixed(backend, uncondEncoder is not null, grid, interpScale, s);
        }
        catch (Exception ex) when (capture)
        {
            backend.StepGraphReset();
            _graphDead = true;
            Logs.Warning($"[LTX graph] capture invalidated — falling back to eager: {ex.Message}");
            RunPairIntoFixed(backend, uncondEncoder is not null, grid, interpScale, s);
            return (_graphVelCond!, uncondEncoder is not null ? _graphVelUncond : null, false);
        }
        if (capture)
        {
            try
            {
                backend.StepGraphEndAndLaunch();
                Logs.Info("[LTX graph] denoise step pair captured; replaying via cuGraphLaunch.");
            }
            catch (Exception ex)
            {
                backend.StepGraphReset();
                _graphDead = true;
                Logs.Warning($"[LTX graph] capture failed — falling back to eager: {ex.Message}");
                RunPairIntoFixed(backend, uncondEncoder is not null, grid, interpScale, s);
            }
        }
        return (_graphVelCond!, uncondEncoder is not null ? _graphVelUncond : null, false);
    }

    /// <summary>Builds the per-step timestep tensors on the eager path (host sinusoid + final scale/shift build) and
    /// lands them in the fixed device buffers via CopyInto — done OUTSIDE the graph capture so the host excursions
    /// (TimeEmbed's MemoryCopy, the scale/shift DataPointer loop) never run inside the recording.</summary>
    private void RefreshTimestepFixed(IBackend backend, float timestep)
    {
        int dim = _config.InnerDim;
        (Tensor temb6, Tensor embedded) = TimeEmbed(backend, timestep);
        _temb6Fixed ??= new Tensor(new TensorShape(6, dim), DType.F32);
        backend.CopyInto(_temb6Fixed, temb6);

        // shift = ss[0]+embedded ; (1+scale) = 1+ss[1]+embedded  — the exact host build FinalLayer does, hoisted out.
        float* ss = (float*)_finalScaleShift!.DataPointer;
        float* em = (float*)embedded.DataPointer;
        Tensor shift = new Tensor(new TensorShape(dim), DType.F32);
        Tensor scaleP1 = new Tensor(new TensorShape(dim), DType.F32);
        float* shp = (float*)shift.DataPointer; float* scp = (float*)scaleP1.DataPointer;
        for (int d = 0; d < dim; d++) { shp[d] = ss[d] + em[d]; scp[d] = 1f + ss[dim + d] + em[d]; }
        _shiftFixed ??= new Tensor(new TensorShape(dim), DType.F32);
        _scaleP1Fixed ??= new Tensor(new TensorShape(dim), DType.F32);
        backend.CopyInto(_shiftFixed, shift);
        backend.CopyInto(_scaleP1Fixed, scaleP1);

        temb6.Dispose(); embedded.Dispose(); shift.Dispose(); scaleP1.Dispose();
    }

    /// <summary>The captured (or capture-warming) step-pair body: both passes over the fixed latent, reading only the
    /// fixed timestep/encoder buffers + cached RoPE, landing velocities in the fixed pre-capture buffers via CopyInto.</summary>
    private void RunPairIntoFixed(IBackend backend, bool hasUncond, (int Frames, int Height, int Width) grid,
        (double T, double H, double W) interpScale, int s)
    {
        (Tensor cos, Tensor sin) = GetCosSin(grid, interpScale);
        Tensor condV = RunOnePassGraph(backend, _encProjCondFixed!, cos, sin, s);
        backend.CopyInto(_graphVelCond!, condV);
        condV.Dispose();
        if (hasUncond)
        {
            Tensor uncondV = RunOnePassGraph(backend, _encProjUncondFixed!, cos, sin, s);
            backend.CopyInto(_graphVelUncond!, uncondV);
            uncondV.Dispose();
        }
    }

    /// <summary>One conditioning pass over the fixed latent using the fixed timestep/encoder buffers: proj_in → blocks
    /// → device final norm → proj_out. All device ops (capture-safe); no host reads.</summary>
    private Tensor RunOnePassGraph(IBackend backend, Tensor encoderProj, Tensor cos, Tensor sin, int s)
    {
        int dim = _config.InnerDim;
        Tensor hidden = new Tensor(new TensorShape(s, dim), DType.F32);
        backend.Linear(hidden, _latentFixed!, _projInW!, _projInB);
        Tensor cur = hidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, _temb6Fixed!, _rope, cos, sin, null);
            cur.Dispose();
            cur = next;
        }
        Tensor normed = new Tensor(new TensorShape(s, dim), DType.F32);
        backend.LayerNormNoAffine(normed, cur, 1e-6f);
        cur.Dispose();
        Tensor affined = new Tensor(new TensorShape(s, dim), DType.F32);
        backend.AffineBroadcastLastDim(affined, normed, _scaleP1Fixed!, _shiftFixed!);
        normed.Dispose();
        Tensor outVel = new Tensor(new TensorShape(s, _config.OutChannels), DType.F32);
        backend.Linear(outVel, affined, _projOutW!, _projOutB);
        affined.Dispose();
        return outVel;
    }

    /// <summary>Returns the RoPE cos/sin tables for this grid, building them once and caching across the many
    /// per-step (and CFG cond/uncond) Forward calls that share a fixed grid. The transformer owns the cached tensors
    /// (disposed in <see cref="Dispose"/> or when the grid changes).</summary>
    private (Tensor Cos, Tensor Sin) GetCosSin((int Frames, int Height, int Width) grid, (double T, double H, double W) interpScale)
    {
        (int F, int H, int W, double T, double IH, double IW) key = (grid.Frames, grid.Height, grid.Width, interpScale.T, interpScale.H, interpScale.W);
        if (_cachedCos is null || _cachedSin is null || !_cosKey.Equals(key))
        {
            _cachedCos?.Dispose(); _cachedSin?.Dispose();
            (_cachedCos, _cachedSin) = _rope.BuildCosSin(grid.Frames, grid.Height, grid.Width, interpScale);
            _cosKey = key;
        }
        return (_cachedCos, _cachedSin);
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
        // Encoder may arrive rank-3 [1, L, captionChannels] (CfgHelper.SliceBatchElement) — derive the row count
        // from element-count / last-dim, NOT Shape[0] (which is the batch 1). Reading Shape[0] would collapse the
        // caption to a single token and undersize the projection buffer (the Wan TextEmbed rank-3 bug class).
        int l = (int)(encoder.ElementCount / encoder.Shape[encoder.Shape.Rank - 1]);
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
        // shift = ss[0]+embedded; (1+scale) = 1+ss[1]+embedded ([dim], broadcast over S). The [dim] build stays host
        // (tiny), folding the +1 into the scale so the device affine is a single input·scale+shift.
        float* ss = (float*)_finalScaleShift!.DataPointer;
        float* em = (float*)embedded.DataPointer;
        Tensor shift = new Tensor(new TensorShape(dim), DType.F32);
        Tensor scaleP1 = new Tensor(new TensorShape(dim), DType.F32);
        float* shp = (float*)shift.DataPointer; float* scp = (float*)scaleP1.DataPointer;
        for (int d = 0; d < dim; d++) { shp[d] = ss[d] + em[d]; scp[d] = 1f + ss[dim + d] + em[d]; }

        // Device LayerNorm + affine → the [s,dim] output stays GPU-resident through the final Linear. Was a host
        // LayerNorm + a host DataPointer affine loop over [s,dim] per step, which forced a re-upload of `normed`.
        Tensor normed = new Tensor(new TensorShape(s, dim), DType.F32);
        backend.LayerNormNoAffine(normed, hidden, 1e-6f);
        Tensor affined = new Tensor(new TensorShape(s, dim), DType.F32);
        backend.AffineBroadcastLastDim(affined, normed, scaleP1, shift);
        normed.Dispose(); shift.Dispose(); scaleP1.Dispose();

        Tensor outVel = new Tensor(new TensorShape(s, _config.OutChannels), DType.F32);
        backend.Linear(outVel, affined, _projOutW!, _projOutB);
        affined.Dispose();
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
            _cachedCos?.Dispose(); _cachedSin?.Dispose();
            _cachedCos = _cachedSin = null;
            _latentFixed?.Dispose(); _temb6Fixed?.Dispose(); _shiftFixed?.Dispose(); _scaleP1Fixed?.Dispose();
            _encProjCondFixed?.Dispose(); _encProjUncondFixed?.Dispose();
            _graphVelCond?.Dispose(); _graphVelUncond?.Dispose();
            _latentFixed = _temb6Fixed = _shiftFixed = _scaleP1Fixed = null;
            _encProjCondFixed = _encProjUncondFixed = _graphVelCond = _graphVelUncond = null;
        }
        GC.SuppressFinalize(this);
    }
}
