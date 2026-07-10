using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Chroma diffusion transformer (<c>lodestones/Chroma</c>, ~8.9B params, 19 double + 38 single blocks).
///
/// Key architectural differences from <see cref="FluxTransformer"/>:
/// <list type="bullet">
///   <item><b>Distilled-guidance approximator</b> replaces every per-block <c>norm.linear</c>: a single shared
///         5-layer SiLU+RMSNorm-residual MLP takes <c>concat([Timesteps(t,16), Timesteps(0,16), mod_proj(arange,32)])</c>
///         and produces a <c>[B, mod_index_length, hidden]</c> table that is sliced per block.</item>
///   <item><b>T5-only conditioning</b> (no CLIP pooled, no guidance scalar). T5-XXL hidden dim 4096.</item>
///   <item>Pruned <c>ChromaAdaLayerNormZero{Pruned,SinglePruned}</c> blocks consume modulation rows directly.</item>
///   <item>Optional <b>per-token attention mask</b> propagated through every block (Flux ignores masks).</item>
///   <item>Final norm <c>ChromaAdaLayerNormContinuousPruned</c> takes <c>[B, 2, hidden]</c> from the last two rows of
///         the modulation table — chunk order is <c>[scale, shift]</c> (row 0 scale, row 1 shift). No checkpoint linear.</item>
/// </list>
///
/// Reference: <c>diffusers/models/transformers/transformer_chroma.py:423-624</c>.</summary>
public sealed unsafe class ChromaTransformer : IDisposable
{
    private readonly ChromaConfig _config;
    private readonly ChromaCombinedTimestepEmbeddings _timestepEmbed;
    private readonly ChromaApproximator _approximator;
    private readonly ChromaDoubleStreamBlock[] _doubleBlocks;
    private readonly ChromaSingleStreamBlock[] _singleBlocks;

    // Per-(txtSeqLen, grid) FluxRope cache, two slots for the alternating cond/uncond text lengths of
    // dual-pass CFG. The old single instance re-ran Precompute (a ~300k-trig host loop) AND re-expanded +
    // re-uploaded the GPU cos/sin tables EVERY forward (~40×/gen) even though positions only change with
    // resolution or prompt length. Same axes/theta as Flux — Chroma reuses FluxPosEmbed verbatim.
    private readonly FluxRope?[] _ropeSlots = new FluxRope?[2];
    private readonly long[] _ropeSig = new long[2];
    private int _ropeNextSlot;

    // x_embedder: Linear(in_channels=64, hidden_size=3072, bias=True)
    private Tensor? _xEmbedWeight, _xEmbedBias;

    // context_embedder: Linear(joint_attention_dim=4096, hidden_size=3072, bias=True)
    private Tensor? _contextEmbedWeight, _contextEmbedBias;

    // proj_out: Linear(hidden_size, patch_size² * out_channels=64, bias=True)
    private Tensor? _projOutWeight, _projOutBias;

    /// <summary>True when this instance runs the audited F16 block loop (classic Chroma + HARTSY_DIT_F16):
    /// token streams are cast to F16 around the block loop and the residual stream rides at
    /// <see cref="ChromaF16.ResidualDamp"/> scale (exact — see <see cref="ChromaF16"/>). Radiance instances
    /// stay F32/undamped (their NeRF head is un-audited).</summary>
    private bool _f16Mode;

    private int _disposed;

    /// <summary>Creates a Chroma transformer from configuration.</summary>
    public ChromaTransformer(ChromaConfig config)
    {
        _config = config;

        _timestepEmbed = new ChromaCombinedTimestepEmbeddings(
            numChannels: config.ApproximatorNumChannels / 4,
            modIndexLength: config.ModIndexLength);

        _approximator = new ChromaApproximator(
            inDim: config.ApproximatorNumChannels,
            outDim: config.HiddenSize,
            hiddenDim: config.ApproximatorHiddenDim,
            numLayers: config.ApproximatorLayers);

        _doubleBlocks = new ChromaDoubleStreamBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++)
        {
            _doubleBlocks[i] = new ChromaDoubleStreamBlock(
                config.HiddenSize, config.NumHeads, config.HeadDim);
        }

        _singleBlocks = new ChromaSingleStreamBlock[config.DepthSingleBlocks];
        for (int i = 0; i < config.DepthSingleBlocks; i++)
        {
            _singleBlocks[i] = new ChromaSingleStreamBlock(
                config.HiddenSize, config.NumHeads, config.HeadDim);
        }
    }

    /// <summary>Loads all transformer weights from named tensors using diffusers naming (post-conversion).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
        => LoadWeightsInternal(weights, requireImageProjections: true);

    /// <summary>Weight-loading core shared with <see cref="ChromaRadianceTransformer"/>. Radiance checkpoints have
    /// no <c>x_embedder.*</c> / <c>proj_out.*</c> (replaced by conv patchify + NeRF head), so those lookups become
    /// optional when <paramref name="requireImageProjections"/> is false — and the F16 block path (with its exact
    /// residual damping) stays classic-Chroma-only until the Radiance head is audited.</summary>
    internal void LoadWeightsInternal(IReadOnlyDictionary<string, Tensor> weights, bool requireImageProjections)
    {
        _f16Mode = requireImageProjections && DitDtype.Act == DType.F16;
        float branchDamp = _f16Mode ? ChromaF16.ResidualDamp : 1.0f;

        if (requireImageProjections)
        {
            _xEmbedWeight = weights["x_embedder.weight"];
            _xEmbedBias = weights["x_embedder.bias"];
            _projOutWeight = weights["proj_out.weight"];
            _projOutBias = weights["proj_out.bias"];
        }
        else
        {
            weights.TryGetValue("x_embedder.weight", out _xEmbedWeight);
            weights.TryGetValue("x_embedder.bias", out _xEmbedBias);
            weights.TryGetValue("proj_out.weight", out _projOutWeight);
            weights.TryGetValue("proj_out.bias", out _projOutBias);
        }

        _contextEmbedWeight = weights["context_embedder.weight"];
        _contextEmbedBias = weights["context_embedder.bias"];

        if (_f16Mode)
        {
            // Enter the damped-residual regime at the embedders: both token streams start at damp scale, so the
            // whole residual stream (which the block-output damping keeps at that scale) fits F16 in the late
            // blocks. The final no-affine LayerNorm cancels the factor exactly, so proj_out is untouched.
            _xEmbedWeight!.Fp8ScaleFactor *= ChromaF16.ResidualDamp;
            _xEmbedBias = ChromaF16.DampBias(_xEmbedBias!, ChromaF16.ResidualDamp);
            _contextEmbedWeight.Fp8ScaleFactor *= ChromaF16.ResidualDamp;
            _contextEmbedBias = ChromaF16.DampBias(_contextEmbedBias!, ChromaF16.ResidualDamp);
            Logs.Info($"[Chroma] F16 block loop active (residual damp 1/{1.0f / ChromaF16.ResidualDamp:F0})");
        }

        _approximator.LoadWeights(weights);

        for (int i = 0; i < _config.Depth; i++)
            _doubleBlocks[i].LoadWeights(weights, $"transformer_blocks.{i}", branchDamp);

        for (int i = 0; i < _config.DepthSingleBlocks; i++)
            _singleBlocks[i].LoadWeights(weights, $"single_transformer_blocks.{i}", branchDamp);
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_xEmbedWeight is not null) yield return _xEmbedWeight;
        if (_xEmbedBias is not null) yield return _xEmbedBias;
        if (_contextEmbedWeight is not null) yield return _contextEmbedWeight;
        if (_contextEmbedBias is not null) yield return _contextEmbedBias;
        foreach (Tensor w in _approximator.EnumerateWeights()) yield return w;
        for (int i = 0; i < _doubleBlocks.Length; i++)
            foreach (Tensor w in _doubleBlocks[i].EnumerateWeights()) yield return w;
        for (int i = 0; i < _singleBlocks.Length; i++)
            foreach (Tensor w in _singleBlocks[i].EnumerateWeights()) yield return w;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="packedLatent">Packed latent tokens [B, imgSeqLen, 64].</param>
    /// <param name="encoderHidden">T5 text embeddings [B, txtSeqLen, 4096].</param>
    /// <param name="timestep">Timestep in [0, 1] (caller should pass <c>t/1000</c> to match diffusers).</param>
    /// <param name="txtSeqLen">Number of text tokens (used to build position ids and to split the joint sequence).</param>
    /// <param name="hPacked">Packed image height (latent_h / 2).</param>
    /// <param name="wPacked">Packed image width (latent_w / 2).</param>
    /// <param name="attentionMask">Optional [B, txtSeqLen] T5 attention mask. The transformer extends it with all-ones for image tokens to form the [B, txtSeqLen+imgSeqLen] mask passed to every block.</param>
    public Tensor Forward(
        IBackend backend,
        Tensor packedLatent,
        Tensor encoderHidden,
        float timestep,
        int txtSeqLen,
        int hPacked,
        int wPacked,
        Tensor? attentionMask)
    {
        Tensor modTable = BuildModTable(backend, timestep, (int)packedLatent.Shape[0]);
        Tensor velocity = ForwardOnePass(backend, packedLatent, encoderHidden, modTable,
            txtSeqLen, hPacked, wPacked, attentionMask);
        modTable.Dispose();
        return velocity;
    }

    /// <summary>Runs one full denoise step's transformer work: the cond pass and (for CFG) the uncond pass,
    /// sharing ONE approximator/modulation-table computation — the two passes always see the same timestep, so
    /// the old per-<see cref="Forward"/> rebuild ran the whole approximator MLP twice per step for nothing.
    /// <para>With the step graph active (<see cref="DitStepGraph"/> — default-ON for Chroma) and the latent
    /// routed through <see cref="PrepareGraphLatent"/>, the ENTIRE pair (approximator → cond forward → uncond
    /// forward) is CUDA-graph-captured once per generation and replayed with a single launch per step, erasing
    /// the host-issue tail over Chroma's ~60k ops/gen. The per-step-varying timestep enters through a fixed
    /// device buffer refreshed OUTSIDE the capture. Self-disables on capture failure (the Z-Image/Krea2 recipe).
    /// <c>callerOwns</c> tells the pipeline whether to dispose the returned velocities: true on the eager path,
    /// false on the graph path (transformer-owned fixed buffers, rewritten by the next step) — the mode can
    /// flip mid-generation when a capture failure falls back to eager.</para></summary>
    public (Tensor condVelocity, Tensor? uncondVelocity, bool callerOwns) ForwardPaired(
        IBackend backend,
        Tensor packedLatent,
        Tensor condContext,
        Tensor? uncondContext,
        float timestep,
        int hPacked,
        int wPacked,
        Tensor? condMask,
        Tensor? uncondMask)
    {
        int batch = (int)packedLatent.Shape[0];
        int condTxtLen = (int)condContext.Shape[1];
        int uncondTxtLen = uncondContext is not null ? (int)uncondContext.Shape[1] : 0;

        bool graphMode = DitStepGraph.EnabledDefaultOn && backend.StepGraphSupported && !_graphDead
            && batch == 1 && ReferenceEquals(packedLatent, _latentFixed);
        if (!graphMode)
        {
            Tensor modTable = BuildModTable(backend, timestep, batch);
            Tensor condV = ForwardOnePass(backend, packedLatent, condContext, modTable,
                condTxtLen, hPacked, wPacked, condMask);
            Tensor? uncondV = null;
            if (uncondContext is not null)
            {
                uncondV = ForwardOnePass(backend, packedLatent, uncondContext, modTable,
                    uncondTxtLen, hPacked, wPacked, uncondMask);
            }
            modTable.Dispose();
            return (condV, uncondV, true);
        }

        // ── Step-graph path ──
        // Refresh the fixed approximator-input buffer — the ONLY per-step-varying content the captured
        // graph reads (the [1, 344, 64] host build is trivial; the H2D lands outside the capture).
        Tensor modInput = _timestepEmbed.Forward(timestep * 1000.0f, 1);
        _modInputFixed ??= new Tensor(modInput.Shape, DType.F32);
        backend.CopyInto(_modInputFixed, modInput);
        modInput.Dispose();

        long sig = ((long)hPacked * 19349663L) ^ ((long)wPacked * 83492791L)
            ^ ((long)condTxtLen * 2654435761L) ^ ((long)uncondTxtLen * 73856093L)
            ^ ((long)System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(condContext) << 17)
            ^ (uncondContext is not null ? (long)System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(uncondContext) << 31 : 0);
        // The backend graph slot is SHARED across models (KEEP_MODELS alternation) — owner mismatch forces a
        // re-warm + re-capture and never counts toward the flip-storm heuristic.
        bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, this);
        if (sig != _graphSig || ownerLost)
        {
            backend.StepGraphReset();
            if (!ownerLost && _graphSig != long.MinValue && ++_graphSigFlips > 8)
            {
                _graphDead = true;
                Logs.Warning("[Chroma graph] signature flip storm — step-graph disabled for this session.");
            }
            _graphSig = sig;
            _graphSigCalls = 0;
            backend.StepGraphOwner = this;
        }
        else if (_graphSigCalls > GraphCaptureCall && !backend.StepGraphReady)
        {
            // The backend graph slot was reset externally (FreeActivations — e.g. a VAE full-res fallback)
            // while our signature stayed valid: the fixed buffers were freed, so re-warm and re-capture.
            _graphSigCalls = 0;
        }
        _graphSigCalls++;
        if (_graphSigCalls == 1)
        {
            // Pin the step-invariant conditioning as device-resident weights: the pipeline host-materializes
            // these caches, so without the pin every forward re-uploads them from PAGEABLE memory — an
            // internally-syncing copy that would invalidate capture (and a hidden per-step serializer in eager
            // mode). The derived SDPA masks are built (or cache-hit) here so their upload also precedes capture.
            List<Tensor> pins = new List<Tensor>(4) { condContext };
            if (uncondContext is not null) pins.Add(uncondContext);
            int imgSeqLen = (int)packedLatent.Shape[1];
            if (condMask is not null)
                pins.Add(GetOrBuildSdpaMask(backend, condMask, 1, condTxtLen, imgSeqLen, condTxtLen + imgSeqLen));
            if (uncondMask is not null)
                pins.Add(GetOrBuildSdpaMask(backend, uncondMask, 1, uncondTxtLen, imgSeqLen, uncondTxtLen + imgSeqLen));
            backend.PreloadWeights(pins);
        }
        int outDim = _projOutWeight is not null ? (int)_projOutWeight.Shape[0] : 64;
        _graphVelCond ??= new Tensor(new TensorShape(1, (long)packedLatent.Shape[1], outDim), DType.F32);
        if (uncondContext is not null)
            _graphVelUncond ??= new Tensor(new TensorShape(1, (long)packedLatent.Shape[1], outDim), DType.F32);

        if (backend.StepGraphReady && _graphSigCalls > GraphCaptureCall)
        {
            backend.StepGraphLaunch();
            return (_graphVelCond!, uncondContext is not null ? _graphVelUncond : null, false);
        }

        // Calls 1..GraphCaptureCall-1 run eagerly THROUGH the fixed buffers (warming the pins + rope/mask
        // uploads so nothing inside the capture does a synchronous alloc); call GraphCaptureCall records.
        bool capture = _graphSigCalls == GraphCaptureCall;
        if (capture) backend.StepGraphBegin();
        try
        {
            RunPairIntoFixed(backend, packedLatent, condContext, uncondContext,
                condTxtLen, uncondTxtLen, hPacked, wPacked, condMask, uncondMask);
        }
        catch (Exception ex) when (capture)
        {
            // A capture-illegal op invalidated the recording (nothing executed). Abort, disable graph mode
            // for the session, and re-run this step eagerly so the generation stays correct.
            backend.StepGraphReset();
            _graphDead = true;
            Logs.Warning($"[Chroma graph] capture invalidated — falling back to eager: {ex}");
            RunPairIntoFixed(backend, packedLatent, condContext, uncondContext,
                condTxtLen, uncondTxtLen, hPacked, wPacked, condMask, uncondMask);
            return (_graphVelCond!, uncondContext is not null ? _graphVelUncond : null, false);
        }
        if (capture)
        {
            try
            {
                backend.StepGraphEndAndLaunch();   // capture records without executing — this runs the step
                Logs.Info("[Chroma graph] denoise step pair captured; replaying via cuGraphLaunch.");
            }
            catch (Exception ex)
            {
                backend.StepGraphReset();
                _graphDead = true;
                Logs.Warning($"[Chroma graph] capture failed — falling back to eager: {ex.Message}");
                RunPairIntoFixed(backend, packedLatent, condContext, uncondContext,
                    condTxtLen, uncondTxtLen, hPacked, wPacked, condMask, uncondMask);
            }
        }
        return (_graphVelCond!, uncondContext is not null ? _graphVelUncond : null, false);
    }

    /// <summary>The captured (or capture-warming) step-pair body: approximator from the fixed timestep buffer,
    /// then both passes, landing the velocities in the fixed pre-capture buffers via <c>CopyInto</c>.</summary>
    private void RunPairIntoFixed(IBackend backend, Tensor packedLatent,
        Tensor condContext, Tensor? uncondContext,
        int condTxtLen, int uncondTxtLen, int hPacked, int wPacked,
        Tensor? condMask, Tensor? uncondMask)
    {
        Tensor modTable = _approximator.Forward(backend, _modInputFixed!);
        Tensor condV = ForwardOnePass(backend, packedLatent, condContext, modTable,
            condTxtLen, hPacked, wPacked, condMask);
        backend.CopyInto(_graphVelCond!, condV);
        condV.Dispose();
        if (uncondContext is not null)
        {
            Tensor uncondV = ForwardOnePass(backend, packedLatent, uncondContext, modTable,
                uncondTxtLen, hPacked, wPacked, uncondMask);
            backend.CopyInto(_graphVelUncond!, uncondV);
            uncondV.Dispose();
        }
        modTable.Dispose();
    }

    /// <summary>Builds the [B, modIndexLength, hidden] modulation table for one timestep: the host-side
    /// combined timestep/slot embedding through the approximator MLP.</summary>
    internal Tensor BuildModTable(IBackend backend, float timestep, int batch)
    {
        // Timestep × 1000: diffusers does this on the input side; the embedding sees [0, 1000].
        Tensor modInput = _timestepEmbed.Forward(timestep * 1000.0f, batch);
        ChromaDebugDump.Dump("mod_input", modInput);
        Tensor modTable = _approximator.Forward(backend, modInput);
        modInput.Dispose();
        ChromaDebugDump.Dump("mod_table", modTable);
        return modTable;
    }

    /// <summary>One conditioning pass: img_in → backbone core → final norm → proj_out. Does NOT dispose
    /// <paramref name="modTable"/> (shared by the CFG pair).</summary>
    private Tensor ForwardOnePass(
        IBackend backend,
        Tensor packedLatent,
        Tensor encoderHidden,
        Tensor modTable,
        int txtSeqLen,
        int hPacked,
        int wPacked,
        Tensor? attentionMask)
    {
        int batch = (int)packedLatent.Shape[0];
        int imgSeqLen = (int)packedLatent.Shape[1];
        int hidden = _config.HiddenSize;

        // ── img_in: [B, imgSeqLen, 64] → [B, imgSeqLen, hidden] ──
        TensorShape imgTokShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor img = new Tensor(imgTokShape, DType.F32);
        backend.Linear(img, packedLatent, _xEmbedWeight!, _xEmbedBias);
        ChromaDebugDump.Dump("img_in", img);

        Tensor imgOut = ForwardCore(backend, img, encoderHidden, modTable,
            txtSeqLen, hPacked, wPacked, attentionMask);

        // ── Final norm (ChromaAdaLayerNormContinuousPruned) ──
        // temb_final = modTable[:, -2:, :] → row 0 (=-2) = SHIFT, row 1 (=-1) = SCALE (ComfyUI Chroma
        // get_modulations("final") = (mod[-2], mod[-1]) → LastLayer `shift, scale = vec`). ApplyContinuousNorm consumes it in that order.
        Tensor finalScaleShift = batch == 1
            ? SliceModSlabDevice(backend, modTable, _config.ModIndexLength - 2, rowCount: 2, hidden)
            : SliceModSlab(modTable, batch, _config.ModIndexLength - 2, rowCount: 2, hidden);

        Tensor normedOut = batch == 1
            ? ApplyContinuousNormDevice(backend, imgOut, finalScaleShift, imgSeqLen, hidden)
            : ApplyContinuousNorm(imgOut, finalScaleShift, batch, imgSeqLen, hidden);
        finalScaleShift.Dispose();
        imgOut.Dispose();
        ChromaDebugDump.Dump("img_post_norm_out", normedOut);

        // ── proj_out: [B, imgSeqLen, hidden] → [B, imgSeqLen, patch_size² * out_channels=64] ──
        int outDim = (int)_projOutWeight!.Shape[0];
        TensorShape projShape = new TensorShape(batch, imgSeqLen, outDim);
        Tensor velocity = new Tensor(projShape, DType.F32);
        backend.Linear(velocity, normedOut, _projOutWeight!, _projOutBias);
        normedOut.Dispose();

        ChromaDebugDump.DumpOutput(velocity);
        return velocity;
    }

    /// <summary>Backbone forward shared with <see cref="ChromaRadianceTransformer"/>: double/single blocks from
    /// already-embedded image tokens to pre-final-norm image tokens (always F32 — the F16 block loop casts in
    /// and out internally). Consumes (disposes) <paramref name="img"/>; does NOT dispose
    /// <paramref name="modTable"/> (caller-owned — build via <see cref="BuildModTable"/>).</summary>
    internal Tensor ForwardCore(
        IBackend backend,
        Tensor img,
        Tensor encoderHidden,
        Tensor modTable,
        int txtSeqLen,
        int hPacked,
        int wPacked,
        Tensor? attentionMask)
    {
        int batch = (int)img.Shape[0];
        int imgSeqLen = (int)img.Shape[1];
        int hidden = _config.HiddenSize;
        int totalSeqLen = txtSeqLen + imgSeqLen;
        int numDoubles = _config.Depth;
        int numSingles = _config.DepthSingleBlocks;

        // ── context_embedder: [B, txtSeqLen, 4096] → [B, txtSeqLen, hidden] ──
        TensorShape txtTokShape = new TensorShape(batch, txtSeqLen, hidden);
        Tensor txt = new Tensor(txtTokShape, DType.F32);
        backend.Linear(txt, encoderHidden, _contextEmbedWeight!, _contextEmbedBias);
        ChromaDebugDump.Dump("txt_in", txt);

        // ── F16 block loop (classic Chroma + HARTSY_DIT_F16, B=1): one cast into F16 per stream before the
        //    loop — every block activation then follows (half the HBM traffic of the bandwidth-bound glue
        //    kernels, and all 57 SDPAs ride the zero-cast F16 cuDNN engine). The streams are already at
        //    ResidualDamp scale from the damped embedders (see ChromaF16), so late-block residual growth
        //    stays inside F16 range. Cast back to F32 after the loop for the final norm + Euler step. ──
        bool f16Loop = _f16Mode && batch == 1;
        if (f16Loop)
        {
            Tensor imgF16 = new Tensor(img.Shape, DType.F16);
            backend.CastToF16(imgF16, img);
            img.Dispose();
            img = imgF16;
            Tensor txtF16 = new Tensor(txt.Shape, DType.F16);
            backend.CastToF16(txtF16, txt);
            txt.Dispose();
            txt = txtF16;
        }
        DType act = img.DType;

        // ── RoPE for joint sequence (text first, then image tokens) — slot-cached per (txtSeqLen, grid) ──
        FluxRope rope = GetRope(txtSeqLen, hPacked, wPacked);

        // ── Extend attention mask to cover image tokens (all-ones), then expand to the [B,1,S,S] additive
        //    SDPA mask — CACHED per (attentionMask ref, seqLen): the mask is step-invariant, and rebuilding
        //    it was a ~21M-iteration host loop + an ~85 MB H2D upload EVERY forward (~40×/gen with dual-pass
        //    CFG — the profiled top H2D_MISS_BIG source). Two slots because the CFG loop alternates the
        //    cond/uncond mask tensors every step. The cache owns the tensors; the forward must not dispose. ──
        Tensor? sdpaMask = attentionMask is not null
            ? GetOrBuildSdpaMask(backend, attentionMask, batch, txtSeqLen, imgSeqLen, totalSeqLen)
            : null;

        // ── Double-stream blocks ──
        // Modulation table layout (matches transformer_chroma.py:546-557):
        //   rows [0,           3 * numSingles)            → single block mods (3 each, used later)
        //   rows [3*numSingles, 3*numSingles + 6*numDoubles) → double-block IMG mods (6 each)
        //   rows [3*numSingles + 6*numDoubles, 3*numSingles + 12*numDoubles) → double-block TXT mods
        //   rows [..., -2]                                 → final norm (scale, shift)
        int imgOffset = 3 * numSingles;
        int txtOffset = imgOffset + 6 * numDoubles;

        for (int i = 0; i < numDoubles; i++)
        {
            int imgRow = imgOffset + 6 * i;
            int txtRow = txtOffset + 6 * i;
            Tensor doubleTemb = batch == 1
                ? BuildDoubleBlockTembDevice(backend, modTable, imgRow, txtRow, hidden)
                : BuildDoubleBlockTemb(modTable, batch, imgRow, txtRow, hidden);
            ChromaDebugDump.Dump($"double_{i}_temb", doubleTemb);

            (Tensor newImg, Tensor newTxt) = _doubleBlocks[i].Forward(backend, img, txt, doubleTemb, rope, sdpaMask);
            doubleTemb.Dispose();

            img.Dispose();
            txt.Dispose();
            img = newImg;
            txt = newTxt;

            ChromaDebugDump.Dump($"double_{i}_img_out", img);
            ChromaDebugDump.Dump($"double_{i}_txt_out", txt);
        }

        // ── Concatenate [txt, img] for single-stream processing. B=1: device row-concat (the old host
        //    Buffer.MemoryCopy read both streams' DataPointer — a full pipeline drain per forward). ──
        TensorShape concatShape = new TensorShape(batch, totalSeqLen, hidden);
        Tensor combined = new Tensor(concatShape, act);
        if (batch == 1)
        {
            backend.Concat(combined, new Tensor[] { txt, img }, dim: 1);
        }
        else
        {
            ConcatTextImage(combined, txt, img, batch, txtSeqLen, imgSeqLen, hidden);
        }
        img.Dispose();
        txt.Dispose();

        // ── Single-stream blocks ──
        for (int i = 0; i < numSingles; i++)
        {
            int singleRow = 3 * i;
            Tensor singleTemb = batch == 1
                ? SliceModSlabDevice(backend, modTable, singleRow, rowCount: 3, hidden)
                : SliceModSlab(modTable, batch, singleRow, rowCount: 3, hidden);
            ChromaDebugDump.Dump($"single_{i}_temb", singleTemb);

            Tensor newCombined = _singleBlocks[i].Forward(backend, combined, singleTemb, rope, sdpaMask);
            singleTemb.Dispose();
            combined.Dispose();
            combined = newCombined;

            ChromaDebugDump.Dump($"single_{i}_out", combined);
        }

        // sdpaMask is owned by the per-generation cache — NOT disposed here.

        // ── Strip text prefix → image tail. B=1: device row slice. ──
        TensorShape imgOutShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgOut = new Tensor(imgOutShape, act);
        if (batch == 1)
        {
            backend.SliceRows(imgOut, combined, txtSeqLen);
        }
        else
        {
            ExtractImageTokens(imgOut, combined, batch, txtSeqLen, imgSeqLen, hidden);
        }
        combined.Dispose();

        if (imgOut.DType == DType.F16)
        {
            // Back to F32 for the final norm + proj_out (velocity precision matters across Euler steps).
            Tensor imgOutF32 = new Tensor(imgOutShape, DType.F32);
            backend.CastToF32(imgOutF32, imgOut);
            imgOut.Dispose();
            imgOut = imgOutF32;
        }
        ChromaDebugDump.Dump("img_pre_norm_out", imgOut);

        return imgOut;
    }

    /// <summary>Convenience pass-through that hooks the static debug dumper for the final latent.</summary>
    public static void DumpFinalLatent(Tensor latent) => ChromaDebugDump.Dump("final_latent", latent);

    /// <summary>Returns the slot-cached <see cref="FluxRope"/> precomputed for this (txtSeqLen, grid), building
    /// it on first use. Two slots cover dual-pass CFG's alternating cond/uncond text lengths.</summary>
    private FluxRope GetRope(int txtSeqLen, int hPacked, int wPacked)
    {
        long sig = ((long)txtSeqLen << 40) ^ ((long)hPacked << 20) ^ (long)wPacked;
        for (int s = 0; s < 2; s++)
        {
            if (_ropeSlots[s] is not null && _ropeSig[s] == sig)
                return _ropeSlots[s]!;
        }
        FluxRope rope = new FluxRope([16, 56, 56], 10000);
        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        rope.Precompute(posIds);
        posIds.Dispose();
        int slot = _ropeNextSlot;
        _ropeNextSlot = 1 - _ropeNextSlot;
        _ropeSlots[slot] = rope;
        _ropeSig[slot] = sig;
        return rope;
    }

    // ── Step-graph state (see ForwardPaired). The fixed boundary buffers live in the activation pool, so the
    // graph is per-generation: the pipeline MUST call InvalidateStepGraph before any FreeActivations /
    // weight-free — replaying a graph whose baked addresses were freed is a context-poisoning CUDA 700. ──
    private const int GraphCaptureCall = 3;
    private Tensor? _latentFixed;
    private Tensor? _modInputFixed;
    private Tensor? _graphVelCond;
    private Tensor? _graphVelUncond;
    private long _graphSig = long.MinValue;
    private int _graphSigCalls;
    private int _graphSigFlips;
    private bool _graphDead;

    /// <summary>Routes a fresh packed latent into the step-graph's FIXED latent buffer (the address the captured
    /// graph reads and the pipeline's in-place <c>CfgEulerStep</c> updates). Returns the fixed tensor —
    /// transformer-owned; the pipeline must not dispose it or read its DataPointer (snapshot via
    /// <see cref="SnapshotGraphLatent"/>). A shape change resets the graph.</summary>
    public Tensor PrepareGraphLatent(IBackend backend, Tensor freshPackedLatent)
    {
        if (_latentFixed is not null && _latentFixed.Shape != freshPackedLatent.Shape)
        {
            InvalidateStepGraph(backend);
            _latentFixed.Dispose();
            _latentFixed = null;
            _graphVelCond?.Dispose();
            _graphVelCond = null;
            _graphVelUncond?.Dispose();
            _graphVelUncond = null;
        }
        _latentFixed ??= new Tensor(freshPackedLatent.Shape, DType.F32);
        backend.CopyInto(_latentFixed, freshPackedLatent);
        return _latentFixed;
    }

    /// <summary>Device-copies the fixed latent into a fresh tensor the caller may freely read/dispose (reading
    /// the fixed tensor's DataPointer directly would D2H-and-evict the buffer the captured graph points at).</summary>
    public Tensor SnapshotGraphLatent(IBackend backend)
    {
        Tensor snap = new Tensor(_latentFixed!.Shape, DType.F32);
        backend.CopyInto(snap, _latentFixed);
        return snap;
    }

    /// <summary>Invalidates the captured step graph. MUST be called at the end of every generation (before
    /// activation reclaims or weight frees): the captured graph bakes weight AND activation-pool device
    /// pointers, so replaying it after either is freed is a CUDA 700 that poisons the whole context (the
    /// Krea2 fleet lesson). The next generation re-warms and re-captures.</summary>
    public void InvalidateStepGraph(IBackend backend)
    {
        backend.StepGraphReset();
        if (ReferenceEquals(backend.StepGraphOwner, this))
            backend.StepGraphOwner = null;
        _graphSig = long.MinValue;   // MinValue = "no sig": the next call resets WITHOUT counting a CFG flip
        _graphSigCalls = 0;
    }

    // Step-invariant SDPA-mask cache (see the build site). Two slots for the alternating cond/uncond
    // mask tensors; keyed on the source tensor reference + joint sequence length. Nulled (not disposed)
    // in Dispose — the tensors may be auto-promoted and their callbacks touch the CudaContext, which
    // callers routinely tear down first.
    private readonly Tensor?[] _sdpaMaskCache = new Tensor?[2];
    private readonly Tensor?[] _sdpaMaskSourceRef = new Tensor?[2];
    private readonly long[] _sdpaMaskSeqLen = new long[2];
    private int _sdpaMaskNextSlot;

    /// <summary>Returns the cached [B,1,S,S] additive SDPA mask for this conditioning tensor, building it on first use.</summary>
    private Tensor GetOrBuildSdpaMask(IBackend backend, Tensor attentionMask, int batch, int txtSeqLen, int imgSeqLen, int totalSeqLen)
    {
        for (int s = 0; s < 2; s++)
        {
            if (_sdpaMaskCache[s] is not null && ReferenceEquals(_sdpaMaskSourceRef[s], attentionMask)
                && _sdpaMaskSeqLen[s] == totalSeqLen)
                return _sdpaMaskCache[s]!;
        }
        Tensor joinedMask = ExtendMaskWithImageOnes(attentionMask, batch, txtSeqLen, imgSeqLen);
        Tensor built = BuildSdpaMask(joinedMask, batch, totalSeqLen);
        joinedMask.Dispose();
        int slot = _sdpaMaskNextSlot;
        _sdpaMaskNextSlot = 1 - _sdpaMaskNextSlot;
        if (_sdpaMaskCache[slot] is not null)
        {
            // The graph warm-up pins cached masks device-resident — un-pin before disposing (no-op otherwise).
            backend.FreeWeights(new[] { _sdpaMaskCache[slot]! });
            _sdpaMaskCache[slot]!.Dispose();
        }
        _sdpaMaskCache[slot] = built;
        _sdpaMaskSourceRef[slot] = attentionMask;
        _sdpaMaskSeqLen[slot] = totalSeqLen;
        return built;
    }

    /// <summary>Expands a [B, S] keep-mask into the additive [B, 1, S, S] SDPA mask (0 where the q·k pair is
    /// kept, -1e30 where masked; broadcast over heads). Built ONCE per forward and shared by every block —
    /// identical to the former per-block <c>BuildSdpaMask</c> (which the blocks now no longer call).</summary>
    private static unsafe Tensor BuildSdpaMask(Tensor mask, int batch, int seqLen)
    {
        Tensor outMask = new Tensor(new TensorShape(batch, 1, seqLen, seqLen), DType.F32);
        float* mPtr = (float*)mask.DataPointer;
        float* outPtr = (float*)outMask.DataPointer;
        const float NegInf = -1.0e30f;
        for (int b = 0; b < batch; b++)
        {
            int maskOffset = b * seqLen;
            int outOffset = b * seqLen * seqLen;
            for (int q = 0; q < seqLen; q++)
            {
                float qKeep = mPtr[maskOffset + q];
                for (int k = 0; k < seqLen; k++)
                {
                    float kKeep = mPtr[maskOffset + k];
                    outPtr[outOffset + q * seqLen + k] = (qKeep * kKeep) > 0.5f ? 0.0f : NegInf;
                }
            }
        }
        return outMask;
    }

    /// <summary>Extends a [B, txtSeqLen] mask with all-ones for image tokens, returning [B, txtSeqLen+imgSeqLen].</summary>
    private static Tensor ExtendMaskWithImageOnes(Tensor txtMask, int batch, int txtSeqLen, int imgSeqLen)
    {
        int total = txtSeqLen + imgSeqLen;
        TensorShape shape = new TensorShape(batch, total);
        Tensor output = new Tensor(shape, DType.F32);

        float* srcPtr = (float*)txtMask.DataPointer;
        float* dstPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < txtSeqLen; s++)
                dstPtr[b * total + s] = srcPtr[b * txtSeqLen + s];
            for (int s = 0; s < imgSeqLen; s++)
                dstPtr[b * total + txtSeqLen + s] = 1.0f;
        }
        return output;
    }

    /// <summary>Device twin of <see cref="BuildDoubleBlockTemb"/> (B=1): two contiguous row-block slices out of
    /// the DEVICE-resident modulation table, concatenated on the GPU. The host version read
    /// <c>modTable.DataPointer</c> per block — a full pipeline drain of the freshly-computed table, twice per
    /// double block per forward (the Kandinsky temb stream-stall pattern).</summary>
    private static Tensor BuildDoubleBlockTembDevice(IBackend backend, Tensor modTable, int imgRow, int txtRow, int hidden)
    {
        Tensor imgRows = new Tensor(new TensorShape(1, 6, hidden), DType.F32);
        backend.SliceRows(imgRows, modTable, imgRow);
        Tensor txtRows = new Tensor(new TensorShape(1, 6, hidden), DType.F32);
        backend.SliceRows(txtRows, modTable, txtRow);
        Tensor output = new Tensor(new TensorShape(1, 12, hidden), DType.F32);
        backend.Concat(output, new Tensor[] { imgRows, txtRows }, dim: 1);
        imgRows.Dispose();
        txtRows.Dispose();
        return output;
    }

    /// <summary>Device twin of <see cref="SliceModSlab"/> (B=1): one contiguous row-block slice on the GPU.</summary>
    private static Tensor SliceModSlabDevice(IBackend backend, Tensor modTable, int rowStart, int rowCount, int hidden)
    {
        Tensor output = new Tensor(new TensorShape(1, rowCount, hidden), DType.F32);
        backend.SliceRows(output, modTable, rowStart);
        return output;
    }

    /// <summary>Builds the [B, 12, hidden] modulation slab consumed by a double block by stacking the 6 IMG rows
    /// followed by the 6 TXT rows. Mirrors <c>torch.cat((pooled[:, img:img+6], pooled[:, txt:txt+6]), dim=1)</c>.</summary>
    private static Tensor BuildDoubleBlockTemb(Tensor modTable, int batch, int imgRow, int txtRow, int hidden)
    {
        int totalRows = (int)modTable.Shape[1];
        TensorShape shape = new TensorShape(batch, 12, hidden);
        Tensor output = new Tensor(shape, DType.F32);

        float* srcPtr = (float*)modTable.DataPointer;
        float* dstPtr = (float*)output.DataPointer;

        long rowBytes = (long)hidden * sizeof(float);
        for (int b = 0; b < batch; b++)
        {
            for (int r = 0; r < 6; r++)
            {
                long src = ((long)b * totalRows + (imgRow + r)) * hidden;
                long dst = ((long)b * 12 + r) * hidden;
                Buffer.MemoryCopy(srcPtr + src, dstPtr + dst, rowBytes, rowBytes);
            }
            for (int r = 0; r < 6; r++)
            {
                long src = ((long)b * totalRows + (txtRow + r)) * hidden;
                long dst = ((long)b * 12 + (6 + r)) * hidden;
                Buffer.MemoryCopy(srcPtr + src, dstPtr + dst, rowBytes, rowBytes);
            }
        }
        return output;
    }

    /// <summary>Slices a contiguous block of <paramref name="rowCount"/> rows out of the modulation table.</summary>
    private static Tensor SliceModSlab(Tensor modTable, int batch, int rowStart, int rowCount, int hidden)
    {
        int totalRows = (int)modTable.Shape[1];
        TensorShape shape = new TensorShape(batch, rowCount, hidden);
        Tensor output = new Tensor(shape, DType.F32);

        float* srcPtr = (float*)modTable.DataPointer;
        float* dstPtr = (float*)output.DataPointer;
        long rowBytes = (long)hidden * sizeof(float);

        for (int b = 0; b < batch; b++)
        {
            long src = ((long)b * totalRows + rowStart) * hidden;
            long dst = (long)b * rowCount * hidden;
            long copyBytes = (long)rowCount * rowBytes;
            Buffer.MemoryCopy(srcPtr + src, dstPtr + dst, copyBytes, copyBytes);
        }
        return output;
    }

    /// <summary>Device twin of <see cref="ApplyContinuousNorm"/> (B=1): LayerNorm + scale/shift modulation as
    /// backend ops, keeping the block-loop output GPU-resident into proj_out. temb rows: 0 = SHIFT, 1 = SCALE.</summary>
    private static Tensor ApplyContinuousNormDevice(IBackend backend, Tensor x, Tensor temb, int seqLen, int hidden)
    {
        TensorShape outShape = new TensorShape(1, seqLen, hidden);
        Tensor normed = new Tensor(outShape, DType.F32);
        backend.LayerNormNoAffine(normed, x, 1e-6f);

        Tensor shift = new Tensor(new TensorShape(1, hidden), DType.F32);
        backend.SliceRows(shift, temb, 0);
        Tensor scale = new Tensor(new TensorShape(1, hidden), DType.F32);
        backend.SliceRows(scale, temb, 1);
        Tensor scalePlus1 = new Tensor(new TensorShape(1, hidden), DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        scale.Dispose();

        Tensor output = new Tensor(outShape, DType.F32);
        backend.AffineBroadcastLastDim(output, normed, scalePlus1, shift);
        normed.Dispose();
        scalePlus1.Dispose();
        shift.Dispose();
        return output;
    }

    /// <summary>Applies <c>ChromaAdaLayerNormContinuousPruned</c>: <c>output = layernorm(x) * (1 + scale) + shift</c>
    /// where <c>(scale, shift)</c> come from <c>temb.flatten(1, 2).chunk(2, dim=1)</c>: row 0 of <paramref name="temb"/>
    /// is scale and row 1 is shift (each [B, hidden]). LayerNorm has no affine parameters.</summary>
    private static Tensor ApplyContinuousNorm(Tensor x, Tensor temb, int batch, int seqLen, int hidden)
    {
        TensorShape outShape = new TensorShape(batch, seqLen, hidden);
        Tensor output = new Tensor(outShape, DType.F32);

        // First normalize x in place (no affine).
        Tensor normed = new Tensor(outShape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, x, batch, seqLen, hidden);

        float* nPtr = (float*)normed.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* tembPtr = (float*)temb.DataPointer;
        // temb is [B, 2, hidden] in (SHIFT, SCALE) order. Per ComfyUI's
        // `comfy/ldm/chroma/layers.py:LastLayer.forward`:
        //   shift, scale = vec  # vec = (mod_vectors[:, -2:-1, :], mod_vectors[:, -1:, :])
        //   x = addcmul(shift, 1 + scale, norm_final(x))   == norm_final(x) * (1 + scale) + shift
        // Row -2 of the 344-row modulation table is SHIFT, row -1 is SCALE. Our previous code had this
        // swapped (treating row 0 of the 2-row slice as scale, row 1 as shift), which scrambled the
        // final-layer modulation and added a fixed-per-image noise overlay even when the rest of the
        // pipeline was correct.
        for (int b = 0; b < batch; b++)
        {
            int shiftBase = b * 2 * hidden + 0 * hidden;
            int scaleBase = b * 2 * hidden + 1 * hidden;
            for (int s = 0; s < seqLen; s++)
            {
                int seqBase = (b * seqLen + s) * hidden;
                for (int d = 0; d < hidden; d++)
                {
                    float scale = tembPtr[scaleBase + d];
                    float shift = tembPtr[shiftBase + d];
                    outPtr[seqBase + d] = nPtr[seqBase + d] * (1.0f + scale) + shift;
                }
            }
        }

        normed.Dispose();
        return output;
    }

    /// <summary>Concatenates [B, txtSeqLen, D] and [B, imgSeqLen, D] along the sequence dim → [B, txt+img, D].
    /// Host fallback (B&gt;1 only — the pipeline path concats on-device).</summary>
    private static void ConcatTextImage(Tensor output, Tensor txt, Tensor img,
        int batch, int txtSeqLen, int imgSeqLen, int hidden)
    {
        float* txtPtr = (float*)txt.DataPointer;
        float* imgPtr = (float*)img.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int totalSeqLen = txtSeqLen + imgSeqLen;

        for (int b = 0; b < batch; b++)
        {
            long txtBytes = (long)txtSeqLen * hidden * sizeof(float);
            long imgBytes = (long)imgSeqLen * hidden * sizeof(float);

            Buffer.MemoryCopy(
                txtPtr + (long)b * txtSeqLen * hidden,
                outPtr + (long)b * totalSeqLen * hidden,
                txtBytes, txtBytes);

            Buffer.MemoryCopy(
                imgPtr + (long)b * imgSeqLen * hidden,
                outPtr + (long)b * totalSeqLen * hidden + (long)txtSeqLen * hidden,
                imgBytes, imgBytes);
        }
    }

    /// <summary>Extracts image tokens (the tail) from concatenated [B, txt+img, D] → [B, img, D].
    /// Host fallback (B&gt;1 only — the pipeline path slices on-device).</summary>
    private static void ExtractImageTokens(Tensor output, Tensor combined, int batch, int txtSeqLen, int imgSeqLen, int hidden)
    {
        float* inPtr = (float*)combined.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int totalSeqLen = txtSeqLen + imgSeqLen;

        for (int b = 0; b < batch; b++)
        {
            long imgBytes = (long)imgSeqLen * hidden * sizeof(float);
            Buffer.MemoryCopy(
                inPtr + (long)b * totalSeqLen * hidden + (long)txtSeqLen * hidden,
                outPtr + (long)b * imgSeqLen * hidden,
                imgBytes, imgBytes);
        }
    }

    /// <summary>Releases all tensor references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            // Nulled WITHOUT disposing (the weight-field pattern): the cached masks / fixed graph buffers may
            // be auto-promoted and their dispose callbacks touch the CudaContext, which callers tear down first.
            for (int s = 0; s < 2; s++)
            {
                _sdpaMaskCache[s] = null;
                _sdpaMaskSourceRef[s] = null;
                _ropeSlots[s] = null;
            }
            _latentFixed = null;
            _modInputFixed = null;
            _graphVelCond = null;
            _graphVelUncond = null;
            _xEmbedWeight = null;
            _xEmbedBias = null;
            _contextEmbedWeight = null;
            _contextEmbedBias = null;
            _projOutWeight = null;
            _projOutBias = null;
            _approximator.Dispose();
        }
        GC.SuppressFinalize(this);
    }
}
