using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Kandinsky 5 text-to-image transformer (<c>Kandinsky5Transformer3DModel</c>, image variant).
/// Mirrors <c>diffusers/models/transformers/transformer_kandinsky.py</c> for the t2i forward path
/// (no video duration, no fractal sparse attention).
///
/// Pipeline:
/// <list type="number">
/// <item>Project the Qwen2.5-VL sequence embeddings <c>[B, S_t, 3584]</c> through <c>Linear → LayerNorm</c>
/// to <c>[B, S_t, model_dim]</c>.</item>
/// <item>Project the CLIP-L pooled embeddings <c>[B, 768]</c> through <c>Linear → LayerNorm</c> to
/// <c>[B, time_dim]</c>; add to the timestep embedding.</item>
/// <item>Patch-embed the latent <c>[B, 1, h, w, in_visual_dim]</c> via reshape + <c>Linear(p_t·p_h·p_w·c → model_dim)</c>.</item>
/// <item>Run <c>num_text_blocks</c> encoder blocks on the text stream with 1D RoPE.</item>
/// <item>Run <c>num_visual_blocks</c> decoder blocks on the visual stream with 3D RoPE for self-attn
/// and cross-attention to the text stream output.</item>
/// <item>Final <c>OutLayer</c>: 2-param modulation (shift, scale) → non-affine LayerNorm → modulate →
/// <c>Linear(model_dim → p_t·p_h·p_w·out_visual_dim)</c> → reshape back to <c>[B, 1, H, W, out_visual_dim]</c>.</item>
/// </list>
///
/// Tensor layout note: diffusers uses NHWC for visual tokens (channels last). Our pipeline produces
/// BCHW latents; the patch-embed step transposes channel-last during reshape and the final
/// <see cref="Unpatchify"/> step converts back so the rest of HartsyInference's VAE/scheduler stack stays
/// in the same NCHW convention as Flux/SD3/Z-Image.</summary>
public sealed unsafe class Kandinsky5Transformer : IDisposable
{
    private readonly Kandinsky5Config _config;
    private readonly Kandinsky5EncoderBlock[] _textBlocks;
    private readonly Kandinsky5DecoderBlock[] _visualBlocks;
    private readonly Kandinsky5Rope _textRope;
    private readonly Kandinsky5Rope _visualRope;
    // Second text-rope instance for the step-graph CFG pair ONLY: cond and uncond text have different seq lengths,
    // so a single instance would rebuild its device tables (a capture-illegal sync alloc) when the paired capture
    // switches passes. Two instances keep both passes' tables pinned simultaneously. Unused on the eager path.
    private readonly Kandinsky5Rope _textRopeUncond;

    private int _disposed;

    // ── Step-graph capture (HARTSY_DIT_GRAPH) — CFG-pair (LTX-0.9 ForwardPaired template) with the host boundary
    // work (PatchEmbed, the timestep sinusoid, Unpatchify) hoisted OUTSIDE the capture. Both passes share the
    // packed latent → shared _visualFixed; temb differs per pass (pooled differs); text proj + pooled proj are
    // step-invariant (computed once, pinned). Restricted to b==1 (RoPE ApplyGpu is a b==1 device path) + resident
    // (Kandinsky always preloads all weights, no streaming). See StepGraphActive. ──
    private Tensor? _visualFixed;                          // [1, numPatches, modelDim] — shared patch-embed, per step
    private Tensor? _textCondFixed, _textUncondFixed;      // [1, sTxt, modelDim]        — projected text, once
    private Tensor? _pooledCondFixed, _pooledUncondFixed;  // [1, timeDim]               — projected pooled, once
    private Tensor? _tembCondFixed, _tembUncondFixed;      // [1, timeDim]               — temb per pass, per step
    private Tensor? _graphProjCond, _graphProjUncond;      // [1, numPatches, patchOutDim] — captured out-layer buffers
    private long _graphSig = long.MinValue;
    private int _graphSigCalls, _graphSigFlips;
    private bool _graphDead;
    private const int GraphCaptureCall = 3;

    /// <summary>True once the CFG-pair step graph is being captured/replayed (from <see cref="GraphCaptureCall"/>
    /// on), not self-disabled. The pipeline suppresses its per-step <c>FreeActivations</c> while this holds — the
    /// captured graph balances its own allocations and holds the fixed buffers at the addresses the capture bakes.
    /// It stays FALSE during the 1..GraphCaptureCall-1 warm passes so those run eagerly WITH the per-step sweep —
    /// otherwise two paired warm forwards' activations pile up beside the graph's own pool and OOM at large token
    /// counts. Re-check AFTER each paired forward so a mid-gen self-disable re-enables the sweep.</summary>
    public bool StepGraphActive => DitStepGraph.Enabled && !_graphDead && _graphSigCalls >= GraphCaptureCall;

    /// <summary>Invalidates the captured step graph — MUST be called before the DiT weights are freed (the graph
    /// bakes weight device pointers; a free + next-gen re-upload would replay against freed memory).</summary>
    public void InvalidateStepGraph(IBackend backend)
    {
        backend.StepGraphReset();
        if (ReferenceEquals(backend.StepGraphOwner, this)) backend.StepGraphOwner = null;
        _graphSig = long.MinValue;
        _graphSigCalls = 0;
    }

    private Tensor? _textProjWeight, _textProjBias;
    private Tensor? _textNormWeight, _textNormBias;

    private Tensor? _pooledProjWeight, _pooledProjBias;
    private Tensor? _pooledNormWeight, _pooledNormBias;

    private Tensor? _timeIn1Weight, _timeIn1Bias;
    private Tensor? _timeIn2Weight, _timeIn2Bias;

    private Tensor? _visualProjWeight, _visualProjBias;

    private Tensor? _outModWeight, _outModBias;
    private Tensor? _outProjWeight, _outProjBias;

    public Kandinsky5Transformer(Kandinsky5Config config)
    {
        _config = config;

        _textBlocks = new Kandinsky5EncoderBlock[config.NumTextBlocks];
        for (int i = 0; i < config.NumTextBlocks; i++)
            _textBlocks[i] = new Kandinsky5EncoderBlock(
                config.ModelDim, config.TimeDim, config.FfDim, config.HeadDim, config.QkNormEps);

        _visualBlocks = new Kandinsky5DecoderBlock[config.NumVisualBlocks];
        for (int i = 0; i < config.NumVisualBlocks; i++)
            _visualBlocks[i] = new Kandinsky5DecoderBlock(
                config.ModelDim, config.TimeDim, config.FfDim, config.HeadDim, config.QkNormEps);

        _textRope = new Kandinsky5Rope(config.HeadDim, config.RopeMaxPeriod);
        _visualRope = new Kandinsky5Rope(config.HeadDim, config.RopeMaxPeriod);
        _textRopeUncond = new Kandinsky5Rope(config.HeadDim, config.RopeMaxPeriod);
    }

    /// <summary>Loads weights using the diffusers state-dict naming. See
    /// <see cref="Kandinsky5CheckpointConverter"/> for the canonical key set.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _textProjWeight = weights["text_embeddings.in_layer.weight"];
        _textProjBias   = weights["text_embeddings.in_layer.bias"];
        _textNormWeight = weights["text_embeddings.norm.weight"];
        _textNormBias   = weights["text_embeddings.norm.bias"];

        _pooledProjWeight = weights["pooled_text_embeddings.in_layer.weight"];
        _pooledProjBias   = weights["pooled_text_embeddings.in_layer.bias"];
        _pooledNormWeight = weights["pooled_text_embeddings.norm.weight"];
        _pooledNormBias   = weights["pooled_text_embeddings.norm.bias"];

        _timeIn1Weight = weights["time_embeddings.in_layer.weight"];
        _timeIn1Bias   = weights["time_embeddings.in_layer.bias"];
        _timeIn2Weight = weights["time_embeddings.out_layer.weight"];
        _timeIn2Bias   = weights["time_embeddings.out_layer.bias"];

        _visualProjWeight = weights["visual_embeddings.in_layer.weight"];
        _visualProjBias   = weights["visual_embeddings.in_layer.bias"];

        for (int i = 0; i < _config.NumTextBlocks; i++)
            _textBlocks[i].LoadWeights(weights, $"text_transformer_blocks.{i}");

        for (int i = 0; i < _config.NumVisualBlocks; i++)
            _visualBlocks[i].LoadWeights(weights, $"visual_transformer_blocks.{i}");

        _outModWeight = weights["out_layer.modulation.out_layer.weight"];
        _outModBias   = weights["out_layer.modulation.out_layer.bias"];
        _outProjWeight = weights["out_layer.out_layer.weight"];
        _outProjBias   = weights["out_layer.out_layer.bias"];
    }

    /// <summary>Yields every weight tensor for GPU preloading via <c>backend.PreloadWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head =
        [
            _textProjWeight, _textProjBias, _textNormWeight, _textNormBias,
            _pooledProjWeight, _pooledProjBias, _pooledNormWeight, _pooledNormBias,
            _timeIn1Weight, _timeIn1Bias, _timeIn2Weight, _timeIn2Bias,
            _visualProjWeight, _visualProjBias,
        ];
        for (int i = 0; i < head.Length; i++)
            if (head[i] is Tensor t) yield return t;

        for (int i = 0; i < _textBlocks.Length; i++)
            foreach (Tensor w in _textBlocks[i].EnumerateWeights()) yield return w;

        for (int i = 0; i < _visualBlocks.Length; i++)
            foreach (Tensor w in _visualBlocks[i].EnumerateWeights()) yield return w;

        Tensor?[] tail = [_outModWeight, _outModBias, _outProjWeight, _outProjBias];
        for (int i = 0; i < tail.Length; i++)
            if (tail[i] is Tensor t) yield return t;
    }

    /// <summary>Forward pass (image variant): predicts velocity for one denoising step. Inputs in BCHW
    /// layout (matches the rest of the HartsyInference diffusion stack); the channel-last conversion
    /// happens internally. Thin wrapper over the shared image/video core with duration 1.</summary>
    /// <param name="latent">Noisy latent <c>[B, in_visual_dim, H_lat, W_lat]</c>.</param>
    /// <param name="timestep">Scaled timestep value (sigma * 1000).</param>
    /// <param name="textEmbeds">Qwen2.5-VL sequence embeddings <c>[B, S_t, in_text_dim]</c>.</param>
    /// <param name="pooledEmbeds">CLIP-L pooled embeddings <c>[B, in_text_dim2]</c>.</param>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds, Tensor pooledEmbeds)
    {
        ThrowIfDisposed();
        if (latent.Shape.Rank != 4)
            throw new ArgumentException($"Image forward expects a 4-D BCHW latent, got {latent.Shape}.", nameof(latent));

        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int latH = (int)latent.Shape[2];
        int latW = (int)latent.Shape[3];

        // BCHW and BC1HW share the same memory layout, so the video core runs unchanged with T=1.
        Tensor latent5d = latent.Reshape(new TensorShape([(long)batch, channels, 1, latH, latW]));
        return ForwardCore(backend, latent5d, timestep, textEmbeds, pooledEmbeds,
            1.0f, 1.0f, 1.0f, new TensorShape(batch, _config.OutVisualDim, latH, latW));
    }

    /// <summary>Forward pass (video variant): predicts velocity over a clip. Latent is BCTHW with
    /// <c>C = config.VisualEmbedDim</c> (33 = noisy 16 + cond 16 + mask 1 when <c>visual_cond</c>);
    /// output is <c>[B, out_visual_dim, T, H, W]</c>. RoPE scale factors divide the rope args per
    /// diffusers (<c>(1,2,2)</c> at 480p-class resolutions, <c>(1,3.16,3.16)</c> otherwise).</summary>
    public Tensor ForwardVideo(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds,
        Tensor pooledEmbeds, float ropeScaleT = 1.0f, float ropeScaleH = 1.0f, float ropeScaleW = 1.0f)
    {
        ThrowIfDisposed();
        if (latent.Shape.Rank != 5)
            throw new ArgumentException($"Video forward expects a 5-D BCTHW latent, got {latent.Shape}.", nameof(latent));
        if ((int)latent.Shape[1] != _config.VisualEmbedDim)
            throw new ArgumentException(
                $"Video latent channels {latent.Shape[1]} != config.VisualEmbedDim {_config.VisualEmbedDim} " +
                $"(visual_cond={_config.VisualCond} expects noisy+cond+mask packing).", nameof(latent));

        int batch = (int)latent.Shape[0];
        int latT = (int)latent.Shape[2];
        int latH = (int)latent.Shape[3];
        int latW = (int)latent.Shape[4];
        TensorShape outShape = new TensorShape([(long)batch, _config.OutVisualDim, latT, latH, latW]);
        return ForwardCore(backend, latent, timestep, textEmbeds, pooledEmbeds,
            ropeScaleT, ropeScaleH, ropeScaleW, outShape);
    }

    /// <summary>CFG-pair forward: cond + uncond velocities for one denoising step. With HARTSY_DIT_GRAPH the whole
    /// pair (both passes sharing the packed latent's patch-embed) is captured once and replayed via a single
    /// cuGraphLaunch; the host boundary work (PatchEmbed, the timestep sinusoid, Unpatchify) runs OUTSIDE the
    /// capture and refreshes the fixed buffers via CopyInto. Falls back to two eager <see cref="ForwardVideo"/>
    /// calls when graph mode is off or self-disabled. b==1 only; returns fresh caller-owned velocities.</summary>
    public (Tensor condVelocity, Tensor uncondVelocity) ForwardVideoPaired(IBackend backend, Tensor packedLatent,
        float timestep, Tensor qwenCond, Tensor pooledCond, Tensor qwenUncond, Tensor pooledUncond,
        float ropeScaleT, float ropeScaleH, float ropeScaleW)
    {
        ThrowIfDisposed();
        int batch = (int)packedLatent.Shape[0];
        int channels = (int)packedLatent.Shape[1];
        int latT = (int)packedLatent.Shape[2];
        int latH = (int)packedLatent.Shape[3];
        int latW = (int)packedLatent.Shape[4];
        TensorShape outShape = new TensorShape([(long)batch, _config.OutVisualDim, latT, latH, latW]);

        bool graphMode = DitStepGraph.Enabled && backend.StepGraphSupported && !_graphDead && batch == 1;
        if (!graphMode)
        {
            Tensor c = ForwardVideo(backend, packedLatent, timestep, qwenCond, pooledCond, ropeScaleT, ropeScaleH, ropeScaleW);
            Tensor u = ForwardVideo(backend, packedLatent, timestep, qwenUncond, pooledUncond, ropeScaleT, ropeScaleH, ropeScaleW);
            return (c, u);
        }

        (int pT, int pH, int pW) = _config.PatchSize;
        int gridT = latT, gridH = latH / pH, gridW = latW / pW;
        int numPatches = gridT * gridH * gridW;
        int sTxtCond = (int)qwenCond.Shape[1];
        int sTxtUncond = (int)qwenUncond.Shape[1];

        // ── Refresh the fixed inputs OUTSIDE the capture (the per-step host excursions live here) ──
        // Shared patch-embed (both passes use the same packed latent) → _visualFixed.
        Tensor vis = PatchEmbed(backend, packedLatent, batch, channels, gridT, latH, latW, gridH, gridW);
        _visualFixed ??= new Tensor(vis.Shape, DType.F32);
        backend.CopyInto(_visualFixed, vis); vis.Dispose();

        // RoPE: host precompute is a no-op after the first call for fixed geometry; device tables are built by
        // ApplyGpu on the warm calls and stay pinned. Separate text-rope instances keep both passes' tables live.
        int[] posCond = new int[sTxtCond]; for (int i = 0; i < sTxtCond; i++) posCond[i] = i;
        int[] posUncond = new int[sTxtUncond]; for (int i = 0; i < sTxtUncond; i++) posUncond[i] = i;
        _textRope.Precompute1D(posCond);
        _textRopeUncond.Precompute1D(posUncond);
        _visualRope.Precompute3D(_config.AxesDims, gridT, gridH, gridW, 0, ropeScaleT, ropeScaleH, ropeScaleW);

        Tensor tstep = ComputeTimestepEmbedding(backend, timestep, batch);   // [1, timeDim], per-step

        // ── Signature / owner management (mirror the LTX-0.9 / Krea2 CFG-pair template) ──
        long sig = ((long)gridT * 73856093L) ^ ((long)gridH * 19349663L) ^ ((long)gridW * 83492791L)
            ^ ((long)sTxtCond * 2654435761L) ^ ((long)sTxtUncond * 40503L);
        bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, this);
        if (sig != _graphSig || ownerLost)
        {
            backend.StepGraphReset();
            if (!ownerLost && _graphSig != long.MinValue && ++_graphSigFlips > 8)
            {
                _graphDead = true;
                HartsyInference.Core.Logging.Logs.Warning("[Kandinsky5 graph] signature flip storm — step-graph disabled for this session.");
            }
            _graphSig = sig;
            _graphSigCalls = 0;
            backend.StepGraphOwner = this;
        }
        _graphSigCalls++;

        if (_graphSigCalls == 1)
        {
            // Step-invariant projections computed ONCE and held. _text*Fixed are read inside the capture → pin them
            // so a pageable re-upload can't invalidate the recording; _pooled*Fixed feed the outside-capture temb Add.
            _textCondFixed?.Dispose(); _textCondFixed = ProjectText(backend, qwenCond);
            _textUncondFixed?.Dispose(); _textUncondFixed = ProjectText(backend, qwenUncond);
            _pooledCondFixed?.Dispose(); _pooledCondFixed = ProjectPooled(backend, pooledCond);
            _pooledUncondFixed?.Dispose(); _pooledUncondFixed = ProjectPooled(backend, pooledUncond);
            // Pin ALL step-invariant buffers into the weight cache: they are computed only on this call and read
            // every subsequent step, so the warm-phase FreeActivations sweeps (StepGraphActive is still false until
            // the capture call) must not reclaim them. _text* are also read inside the capture.
            backend.PreloadWeights([_textCondFixed, _textUncondFixed, _pooledCondFixed, _pooledUncondFixed]);
        }

        // temb per pass = timestep sinusoid MLP + projected pooled (device Add), refreshed each step OUTSIDE capture.
        _tembCondFixed ??= new Tensor(tstep.Shape, DType.F32);
        { Tensor s = new Tensor(tstep.Shape, DType.F32); backend.Add(s, tstep, _pooledCondFixed!); backend.CopyInto(_tembCondFixed, s); s.Dispose(); }
        _tembUncondFixed ??= new Tensor(tstep.Shape, DType.F32);
        { Tensor s = new Tensor(tstep.Shape, DType.F32); backend.Add(s, tstep, _pooledUncondFixed!); backend.CopyInto(_tembUncondFixed, s); s.Dispose(); }
        tstep.Dispose();

        int patchOutDim = pH * pW * _config.OutVisualDim;
        _graphProjCond ??= new Tensor(new TensorShape(1, numPatches, patchOutDim), DType.F32);
        _graphProjUncond ??= new Tensor(new TensorShape(1, numPatches, patchOutDim), DType.F32);

        if (backend.StepGraphReady && _graphSigCalls > GraphCaptureCall)
        {
            backend.StepGraphLaunch();
            return (SnapshotUnpatchify(backend, _graphProjCond!, gridT, gridH, gridW, latH, latW, outShape),
                    SnapshotUnpatchify(backend, _graphProjUncond!, gridT, gridH, gridW, latH, latW, outShape));
        }

        // Calls 1..GraphCaptureCall-1 run the pair eagerly THROUGH the fixed buffers (warm rope-table builds +
        // weight promotions); call GraphCaptureCall records the same op sequence.
        bool capture = _graphSigCalls == GraphCaptureCall;
        if (capture)
        {
            // Return the default pool's retained reservation (the warm passes ran trimPool:false) to the driver so
            // the graph's OWN stream-ordered memory pool has headroom — otherwise both are held at once.
            backend.TrimMemoryPool();
            // VRAM guard: the captured CFG pair's footprint is dominated by the O(numPatches²) dense self-attention
            // buffers, which grow quadratically with the token count. At high token counts (e.g. 7168 at 512²/25f)
            // it exceeds a 24 GB card, and a graph-pool OOM is an UNCATCHABLE hard fault — so estimate the footprint
            // (~1.5 KB per token², calibrated: 3072 tokens captured cleanly with 17.6 GB free) and fall back to eager
            // when free VRAM is short. A bigger GPU clears the bar and captures. Below the bar the graph engages.
            long estBytes = (long)numPatches * numPatches * 1500L;
            long freeBytes = backend.FreeMemoryBytes();
            if (freeBytes < estBytes)
            {
                _graphDead = true;
                HartsyInference.Core.Logging.Logs.Info(
                    $"[Kandinsky5 graph] {numPatches} tokens need ~{estBytes >> 20} MB but only {freeBytes >> 20} MB free — running eager (no capture).");
                RunPairIntoFixed(backend, numPatches);
                return (SnapshotUnpatchify(backend, _graphProjCond!, gridT, gridH, gridW, latH, latW, outShape),
                        SnapshotUnpatchify(backend, _graphProjUncond!, gridT, gridH, gridW, latH, latW, outShape));
            }
            backend.StepGraphBegin();
        }
        try
        {
            RunPairIntoFixed(backend, numPatches);
        }
        catch (Exception ex) when (capture)
        {
            backend.StepGraphReset();
            _graphDead = true;
            HartsyInference.Core.Logging.Logs.Warning($"[Kandinsky5 graph] capture invalidated — falling back to eager: {ex.Message}");
            RunPairIntoFixed(backend, numPatches);
            return (SnapshotUnpatchify(backend, _graphProjCond!, gridT, gridH, gridW, latH, latW, outShape),
                    SnapshotUnpatchify(backend, _graphProjUncond!, gridT, gridH, gridW, latH, latW, outShape));
        }
        if (capture)
        {
            try
            {
                backend.StepGraphEndAndLaunch();
                HartsyInference.Core.Logging.Logs.Info("[Kandinsky5 graph] CFG-pair denoise step captured; replaying via cuGraphLaunch.");
            }
            catch (Exception ex)
            {
                backend.StepGraphReset();
                _graphDead = true;
                HartsyInference.Core.Logging.Logs.Warning($"[Kandinsky5 graph] capture failed — falling back to eager: {ex.Message}");
                RunPairIntoFixed(backend, numPatches);
            }
        }
        return (SnapshotUnpatchify(backend, _graphProjCond!, gridT, gridH, gridW, latH, latW, outShape),
                SnapshotUnpatchify(backend, _graphProjUncond!, gridT, gridH, gridW, latH, latW, outShape));
    }

    /// <summary>The captured (or capture-warming) CFG-pair body: cond then uncond, both over the shared fixed
    /// visual tokens, each landing its out-layer projection in its fixed buffer. Both passes captured in ONE graph
    /// so their differing text seq lengths are baked (the LTX-0.9 RunPairIntoFixed property).</summary>
    private void RunPairIntoFixed(IBackend backend, int numPatches)
    {
        RunOnePassGraph(backend, _textCondFixed!, _tembCondFixed!, _textRope, _graphProjCond!, numPatches);
        RunOnePassGraph(backend, _textUncondFixed!, _tembUncondFixed!, _textRopeUncond, _graphProjUncond!, numPatches);
    }

    /// <summary>One conditioning pass over the shared fixed visual tokens with a pass-specific fixed text/temb and
    /// text-rope: text encoder blocks → visual decoder blocks (cross-attend to the text output) → device out-layer,
    /// landing in <paramref name="graphProjDst"/>. All device ops — no host reads (b==1 RoPE is the GPU path).</summary>
    private void RunOnePassGraph(IBackend backend, Tensor textFixed, Tensor tembFixed, Kandinsky5Rope textRope,
        Tensor graphProjDst, int numPatches)
    {
        Tensor curText = new Tensor(textFixed.Shape, DType.F32);
        backend.CopyInto(curText, textFixed);
        for (int i = 0; i < _textBlocks.Length; i++)
        {
            Tensor next = _textBlocks[i].Forward(backend, curText, tembFixed, textRope);
            curText.Dispose(); curText = next;
        }

        Tensor curVisual = new Tensor(_visualFixed!.Shape, DType.F32);
        backend.CopyInto(curVisual, _visualFixed!);
        for (int i = 0; i < _visualBlocks.Length; i++)
        {
            Tensor next = _visualBlocks[i].Forward(backend, curVisual, curText, tembFixed, _visualRope);
            curVisual.Dispose(); curVisual = next;
        }
        curText.Dispose();

        Tensor finalProj = ApplyOutLayer(backend, curVisual, tembFixed, 1, numPatches);
        curVisual.Dispose();
        backend.CopyInto(graphProjDst, finalProj);
        finalProj.Dispose();
    }

    /// <summary>Device-copies a fixed out-layer buffer into a fresh disposable tensor and unpatchifies THAT into a
    /// caller-owned velocity. Reading the fixed buffer's DataPointer directly would D2H-AND-FREE the graph-owned
    /// buffer the captured graph writes (the Krea2 SnapshotGraphLatent rule), corrupting later replays.</summary>
    private Tensor SnapshotUnpatchify(IBackend backend, Tensor graphProj, int gridT, int gridH, int gridW,
        int latH, int latW, TensorShape outShape)
    {
        Tensor snap = new Tensor(graphProj.Shape, DType.F32);
        backend.CopyInto(snap, graphProj);
        Tensor velocity = Unpatchify(snap, 1, gridT, gridH, gridW, latH, latW, outShape);
        snap.Dispose();
        return velocity;
    }

    /// <summary>Shared image/video forward over a BCTHW latent view. <paramref name="outShape"/> selects
    /// the rank of the returned tensor (4-D for image, 5-D for video; identical memory layout).</summary>
    private Tensor ForwardCore(IBackend backend, Tensor latent, float timestep, Tensor textEmbeds,
        Tensor pooledEmbeds, float ropeScaleT, float ropeScaleH, float ropeScaleW, TensorShape outShape)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int latT = (int)latent.Shape[2];
        int latH = (int)latent.Shape[3];
        int latW = (int)latent.Shape[4];

        (int pT, int pH, int pW) = _config.PatchSize;
        if (pT != 1)
            throw new InvalidOperationException($"Kandinsky5Transformer requires patch_size[0]=1; got {pT}");
        if (latH % pH != 0 || latW % pW != 0)
            throw new InvalidOperationException(
                $"Latent size {latH}x{latW} must be divisible by patch size {pH}x{pW}");

        int gridT = latT, gridH = latH / pH, gridW = latW / pW;
        int numPatches = gridT * gridH * gridW;

        // ── 1. Project text embeddings ──
        Tensor textHidden = ProjectText(backend, textEmbeds);
        Kandinsky5DebugDump.Dump("text_proj", textHidden);

        // ── 2. Project pooled embeddings + timestep MLP, sum ──
        Tensor pooledHidden = ProjectPooled(backend, pooledEmbeds);
        Kandinsky5DebugDump.Dump("pooled_proj", pooledHidden);

        // GPU add — the previous host AddInPlace read temb.DataPointer, which EVICTED temb from the GPU
        // activation cache; every block's modulation Silu then re-uploaded it via a synchronous pageable
        // H2D that drained the whole queued stream (a full pipeline stall per block — measured at ~90% of
        // the Kandinsky-5 step time).
        Tensor tembSum = ComputeTimestepEmbedding(backend, timestep, batch);
        Tensor temb = new Tensor(tembSum.Shape, DType.F32);
        backend.Add(temb, tembSum, pooledHidden);
        tembSum.Dispose();
        pooledHidden.Dispose();
        Kandinsky5DebugDump.Dump("time_embed", temb);

        // ── 3. Patch embed ──
        Tensor visualTokens = PatchEmbed(backend, latent, batch, channels, gridT, latH, latW, gridH, gridW);
        Kandinsky5DebugDump.Dump("patch_embed", visualTokens);

        // ── 4. Build RoPEs ──
        int sT = (int)textHidden.Shape[1];
        int[] textPosArr = new int[sT];
        for (int i = 0; i < sT; i++) textPosArr[i] = i;
        _textRope.Precompute1D(textPosArr);
        _visualRope.Precompute3D(_config.AxesDims, gridT, gridH, gridW, 0, ropeScaleT, ropeScaleH, ropeScaleW);

        // ── 5. Text encoder blocks ──
        Tensor curText = textHidden;
        for (int i = 0; i < _textBlocks.Length; i++)
        {
            Tensor next = _textBlocks[i].Forward(backend, curText, temb, _textRope);
            curText.Dispose();
            curText = next;
            Kandinsky5DebugDump.Dump($"text_block_{i}", curText);
        }

        // ── 6. Visual decoder blocks ──
        Tensor curVisual = visualTokens;
        for (int i = 0; i < _visualBlocks.Length; i++)
        {
            Tensor next = _visualBlocks[i].Forward(backend, curVisual, curText, temb, _visualRope);
            curVisual.Dispose();
            curVisual = next;
            if (Kandinsky5DebugDump.Enabled)
                Kandinsky5DebugDump.Dump($"visual_block_{i}", curVisual);
        }
        curText.Dispose();

        // ── 7. Final out layer: 2-param modulation + LayerNorm + Linear ──
        Tensor finalProj = ApplyOutLayer(backend, curVisual, temb, batch, numPatches);
        curVisual.Dispose();
        temb.Dispose();
        Kandinsky5DebugDump.Dump("out_layer", finalProj);

        // ── 8. Unpatchify back to BC(T)HW ──
        Tensor result = Unpatchify(finalProj, batch, gridT, gridH, gridW, latH, latW, outShape);
        finalProj.Dispose();
        Kandinsky5DebugDump.DumpOutput(result);

        return result;
    }

    /// <summary>Pipeline-level debug hook: dumps the post-denoise, pre-VAE latent.</summary>
    public static void DumpFinalLatent(Tensor latent) => Kandinsky5DebugDump.Dump("final_latent", latent);

    private Tensor ProjectText(IBackend backend, Tensor textEmbeds)
    {
        int batch = (int)textEmbeds.Shape[0];
        int seqLen = (int)textEmbeds.Shape[1];
        int dim = _config.ModelDim;

        Tensor projected = new Tensor(new TensorShape(batch, seqLen, dim), DType.F32);
        backend.Linear(projected, textEmbeds, _textProjWeight!, _textProjBias);

        Tensor normed = new Tensor(new TensorShape(batch, seqLen, dim), DType.F32);
        backend.LayerNorm(normed, projected, _textNormWeight!, _textNormBias!, 1e-5f);
        projected.Dispose();
        return normed;
    }

    private Tensor ProjectPooled(IBackend backend, Tensor pooled)
    {
        int batch = (int)pooled.Shape[0];
        int outDim = _config.TimeDim;

        Tensor projected = new Tensor(new TensorShape(batch, outDim), DType.F32);
        backend.Linear(projected, pooled, _pooledProjWeight!, _pooledProjBias);

        Tensor normed = new Tensor(new TensorShape(batch, outDim), DType.F32);
        backend.LayerNorm(normed, projected, _pooledNormWeight!, _pooledNormBias!, 1e-5f);
        projected.Dispose();
        return normed;
    }

    /// <summary>Computes the time embedding per <c>Kandinsky5TimeEmbeddings</c>:
    /// <c>args = t ⊗ freqs(model_dim/2)</c> → <c>cat(cos, sin)</c> → <c>Linear(model_dim → time_dim) → SiLU →
    /// Linear(time_dim → time_dim)</c>. Note this differs from the AuraFlow timestep MLP: the sinusoidal
    /// dim is <c>model_dim</c> (not 256) and the activation is between the two linears, not after.</summary>
    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep, int batch)
    {
        int modelDim = _config.ModelDim;
        int timeDim = _config.TimeDim;
        int half = modelDim / 2;

        Tensor sin = new Tensor(new TensorShape(batch, modelDim), DType.F32);
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int baseOff = b * modelDim;
            for (int k = 0; k < half; k++)
            {
                double freq = 1.0 / Math.Pow(_config.RopeMaxPeriod, (double)(2 * k) / modelDim);
                double angle = timestep * freq;
                sinPtr[baseOff + k] = (float)Math.Cos(angle);
                sinPtr[baseOff + half + k] = (float)Math.Sin(angle);
            }
        }

        Tensor mid = new Tensor(new TensorShape(batch, timeDim), DType.F32);
        backend.Linear(mid, sin, _timeIn1Weight!, _timeIn1Bias);
        sin.Dispose();

        Tensor activated = new Tensor(new TensorShape(batch, timeDim), DType.F32);
        backend.Silu(activated, mid);
        mid.Dispose();

        Tensor temb = new Tensor(new TensorShape(batch, timeDim), DType.F32);
        backend.Linear(temb, activated, _timeIn2Weight!, _timeIn2Bias);
        activated.Dispose();
        return temb;
    }

    /// <summary>Patch embed over a BCTHW latent. Mirrors diffusers' <c>Kandinsky5VisualEmbeddings</c>:
    /// <c>[B, T, H, W, C] → view(B, T/pT, pT, H/pH, pH, W/pW, pW, C) → permute(0,1,3,5,2,4,6,7) →
    /// flatten(4,7)</c>, i.e. token order <c>(t, gy, gx)</c> and inner patch order <c>(pt, py, px, c)</c>
    /// with C innermost. With BCTHW input we walk (c, py, px) and write to (py, px, c) per patch
    /// (pT is always 1), then project via <c>visual_embeddings.in_layer</c>.</summary>
    private Tensor PatchEmbed(IBackend backend, Tensor latent, int batch, int channels, int gridT,
        int latH, int latW, int gridH, int gridW)
    {
        (int _, int pH, int pW) = _config.PatchSize;
        int patchInDim = pH * pW * channels;
        int numPatches = gridT * gridH * gridW;

        Tensor flat = new Tensor(new TensorShape(batch, numPatches, patchInDim), DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)flat.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < gridT; t++)
            {
                for (int gy = 0; gy < gridH; gy++)
                {
                    for (int gx = 0; gx < gridW; gx++)
                    {
                        int patchIdx = (t * gridH + gy) * gridW + gx;
                        long dstBase = ((long)b * numPatches + patchIdx) * patchInDim;
                        for (int py = 0; py < pH; py++)
                        {
                            for (int px = 0; px < pW; px++)
                            {
                                int srcY = gy * pH + py;
                                int srcX = gx * pW + px;
                                for (int c = 0; c < channels; c++)
                                {
                                    long srcIdx = ((((long)b * channels + c) * gridT + t) * latH + srcY) * latW + srcX;
                                    long dstIdx = dstBase + (py * pW + px) * channels + c;
                                    outPtr[dstIdx] = inPtr[srcIdx];
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor projected = new Tensor(new TensorShape(batch, numPatches, _config.ModelDim), DType.F32);
        backend.Linear(projected, flat, _visualProjWeight!, _visualProjBias);
        flat.Dispose();
        return projected;
    }

    /// <summary>Final output layer per <c>Kandinsky5OutLayer.forward</c>: produces 2 modulation params
    /// (shift, scale), applies non-affine LayerNorm + modulate, then projects to
    /// <c>p_h * p_w * out_visual_dim</c> per token.</summary>
    private Tensor ApplyOutLayer(IBackend backend, Tensor visual, Tensor temb, int batch, int numPatches)
    {
        int dim = _config.ModelDim;
        (int _, int pH, int pW) = _config.PatchSize;
        int patchOutDim = pH * pW * _config.OutVisualDim;

        Tensor activated = new Tensor(new TensorShape(batch, _config.TimeDim), DType.F32);
        backend.Silu(activated, temb);

        Tensor mod = new Tensor(new TensorShape(batch, 2 * dim), DType.F32);
        backend.Linear(mod, activated, _outModWeight!, _outModBias);
        activated.Dispose();

        // GPU-resident modulation: out = norm * (1 + scale) + shift.
        // mod chunk order is (shift, scale) per `torch.chunk(self.modulation(...), 2, dim=-1)`.
        Tensor shift = new Tensor(new TensorShape(batch, dim), DType.F32);
        backend.SliceLastDim(shift, mod, 0);
        Tensor scale = new Tensor(new TensorShape(batch, dim), DType.F32);
        backend.SliceLastDim(scale, mod, dim);
        mod.Dispose();

        Tensor normed = new Tensor(new TensorShape(batch, numPatches, dim), DType.F32);
        backend.LayerNormNoAffine(normed, visual, 1e-6f);

        Tensor scalePlus1 = new Tensor(new TensorShape(batch, dim), DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        scale.Dispose();

        Tensor modulated = new Tensor(new TensorShape(batch, numPatches, dim), DType.F32);
        backend.AffineBroadcastLastDim(modulated, normed, scalePlus1, shift);
        normed.Dispose();
        scalePlus1.Dispose();
        shift.Dispose();

        Tensor projected = new Tensor(new TensorShape(batch, numPatches, patchOutDim), DType.F32);
        backend.Linear(projected, modulated, _outProjWeight!, _outProjBias);
        modulated.Dispose();
        return projected;
    }

    /// <summary>Reverses the out-layer projection reshape: <c>[B, num_patches, p_h·p_w·C_out]</c> →
    /// BC(T)HW per <paramref name="outShape"/>. Per diffusers <c>Kandinsky5OutLayer</c>, the projection
    /// vector splits as <c>view(..., -1, p_t, p_h, p_w)</c>, i.e. <b>channel-major</b> inner order
    /// <c>(c, pt, py, px)</c> — note this differs from the patch-embed input order (which is C-minor).</summary>
    private Tensor Unpatchify(Tensor patched, int batch, int gridT, int gridH, int gridW, int latH, int latW,
        TensorShape outShape)
    {
        (int _, int pH, int pW) = _config.PatchSize;
        int outChannels = _config.OutVisualDim;
        int patchOutDim = pH * pW * outChannels;
        int numPatches = gridT * gridH * gridW;

        Tensor result = new Tensor(outShape, DType.F32);
        float* srcPtr = (float*)patched.DataPointer;
        float* dstPtr = (float*)result.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < gridT; t++)
            {
                for (int gy = 0; gy < gridH; gy++)
                {
                    for (int gx = 0; gx < gridW; gx++)
                    {
                        int patchIdx = (t * gridH + gy) * gridW + gx;
                        long srcBase = ((long)b * numPatches + patchIdx) * patchOutDim;
                        for (int c = 0; c < outChannels; c++)
                        {
                            for (int py = 0; py < pH; py++)
                            {
                                for (int px = 0; px < pW; px++)
                                {
                                    int dstY = gy * pH + py;
                                    int dstX = gx * pW + px;
                                    long srcIdx = srcBase + (c * pH + py) * pW + px;
                                    long dstIdx = ((((long)b * outChannels + c) * gridT + t) * latH + dstY) * latW + dstX;
                                    dstPtr[dstIdx] = srcPtr[srcIdx];
                                }
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Drops tensor references without disposing them — the underlying weight storage is owned by the mmap loader.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _textProjWeight = null; _textProjBias = null;
            _textNormWeight = null; _textNormBias = null;
            _pooledProjWeight = null; _pooledProjBias = null;
            _pooledNormWeight = null; _pooledNormBias = null;
            _timeIn1Weight = null; _timeIn1Bias = null;
            _timeIn2Weight = null; _timeIn2Bias = null;
            _visualProjWeight = null; _visualProjBias = null;
            _outModWeight = null; _outModBias = null;
            _outProjWeight = null; _outProjBias = null;
        }
        GC.SuppressFinalize(this);
    }

    /// <summary>Converts a channel-last (BHWC) tensor to BCHW. No call sites in this codebase today — kept
    /// for advanced callers that already have channel-last latents.</summary>
    public static Tensor ToBchw(Tensor bhwc)
    {
        int b = (int)bhwc.Shape[0];
        int h = (int)bhwc.Shape[1];
        int w = (int)bhwc.Shape[2];
        int c = (int)bhwc.Shape[3];

        Tensor result = new Tensor(new TensorShape(b, c, h, w), DType.F32);
        float* src = (float*)bhwc.DataPointer;
        float* dst = (float*)result.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int hi = 0; hi < h; hi++)
                    for (int wi = 0; wi < w; wi++)
                        dst[((bi * c + ci) * h + hi) * w + wi] = src[((bi * h + hi) * w + wi) * c + ci];
        return result;
    }
}
