using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Krea 2 diffusion transformer (<c>Krea2Transformer2DModel</c>). Single-stream MMDiT flow-matching backbone:
/// a text-fusion stage collapses the tapped Qwen3-VL-4B hidden states, the result is concatenated with patchified
/// image latents into one <c>[text, image]</c> sequence, 28 sigmoid-gate GQA blocks (modulated by a shared timestep
/// vector + per-block tables) run over it, and the image tail is projected back to a velocity. See
/// <c>docs/Research/KREA2.md</c>.
///
/// <para>Reuses <see cref="FluxRope"/> (3-axis [32,48,48] θ=1000, text rows at position 0, image rows on the latent
/// grid — byte-compatible with Krea 2's FluxPosEmbed convention) and <see cref="DiTUtils"/>. The only Krea 2-specific
/// pieces are the sigmoid output-gate attention, the 6-way <c>scale_shift_table</c> modulation, and the text-fusion
/// stage (see <see cref="Krea2Block"/> / <see cref="Krea2TextFusion"/>).</para></summary>
public sealed unsafe class Krea2Transformer : IDisposable
{
    private readonly Krea2Config _config;
    private readonly FluxRope _rope;
    private readonly Krea2TextFusion _textFusion;
    private readonly Krea2Block[] _blocks;

    private Tensor? _imgInW, _imgInB;
    private Tensor? _time1W, _time1B, _time2W, _time2B;     // time_embed MLP
    private Tensor? _timeModW, _timeModB;                   // time_mod_proj
    private Tensor? _txtNormW, _txt1W, _txt1B, _txt2W, _txt2B; // txt_in (Krea2TextProjection)
    private Tensor? _finalTable, _finalNormW, _finalLinW, _finalLinB;

    // Per-generation caches (the text projection and the RoPE tables are identical across all denoise steps for a
    // given prompt+resolution — recomputing them every step was ~95 op launches + 4 text-fusion SDPAs + a RoPE
    // rebuild per step of pure waste). Keyed on the encoderHidden reference / a (txtSeq,hPacked,wPacked) signature
    // so CFG's alternating cond/uncond streams each recompute correctly; a new prompt (new encoderHidden) evicts.
    private Tensor? _cachedTxt;
    private object? _cachedTxtKey;
    private long _ropeSig = long.MinValue;

    // ── Step-graph state (HARTSY_DIT_GRAPH; see ForwardPatched) ──────────────────────────────────────────
    // The captured graph bakes device addresses, so every per-step-varying boundary lives in a FIXED buffer
    // owned here: the patchified latent (_latentFixed, updated in-place by the pipeline's CfgEulerStep and
    // refreshed per gen via PrepareGraphLatent), the timestep modulation (_tembFixed/_tembModFixed, refreshed
    // per step via CopyInto), and the velocity output (_graphVelocity, written by a captured CopyInto as the
    // graph's last op — a pre-capture NORMAL buffer, so it is safely disposable, unlike graph-owned memory).
    private Tensor? _latentFixed;
    private Tensor? _tembFixed, _tembModFixed;
    private Tensor? _graphVelocity;
    private long _graphSig = long.MinValue;   // rope signature ⊕ txt identity the captured graph is valid for
    private int _graphSigCalls;               // calls at the current sig (capture on the 3rd — caches/promotions warm)
    private int _graphSigFlips;               // sig alternation counter (CFG cond/uncond → graph unusable)
    private bool _graphDead;                  // permanent per-session fallback to eager
    private const int GraphCaptureCall = 3;

    private int _disposed;

    public Krea2Transformer(Krea2Config config)
    {
        _config = config;
        if (config.AxesDimRope[0] + config.AxesDimRope[1] + config.AxesDimRope[2] != config.AttentionHeadDim)
            throw new ArgumentException("sum(axes_dim_rope) must equal attention_head_dim.");
        _rope = new FluxRope(config.AxesDimRope, config.RopeTheta);
        _textFusion = new Krea2TextFusion(config.NumTextLayers, config.TextHiddenDim, config.TextNumHeads,
            config.TextNumKvHeads, config.TextIntermediateSize, config.NumLayerwiseTextBlocks,
            config.NumRefinerTextBlocks, config.NormEps);
        _blocks = new Krea2Block[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new Krea2Block(config.HiddenSize, config.IntermediateSize, config.NumAttentionHeads,
                config.NumKvHeads, config.NormEps);
    }

    public Krea2Config Config => _config;

    /// <summary>Loads weights from a diffusers-style key dict (bare keys).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _imgInW = w["img_in.weight"]; _imgInB = w["img_in.bias"];
        _time1W = w["time_embed.linear_1.weight"]; _time1B = w["time_embed.linear_1.bias"];
        _time2W = w["time_embed.linear_2.weight"]; _time2B = w["time_embed.linear_2.bias"];
        _timeModW = w["time_mod_proj.weight"]; _timeModB = w["time_mod_proj.bias"];

        _txtNormW = Krea2Norm.LoadZeroCentered(w["txt_in.norm.weight"]);
        _txt1W = w["txt_in.linear_1.weight"]; _txt1B = w["txt_in.linear_1.bias"];
        _txt2W = w["txt_in.linear_2.weight"]; _txt2B = w["txt_in.linear_2.bias"];

        _finalTable = F32(w["final_layer.scale_shift_table"]);
        _finalNormW = Krea2Norm.LoadZeroCentered(w["final_layer.norm.weight"]);
        _finalLinW = w["final_layer.linear.weight"]; _finalLinB = w["final_layer.linear.bias"];

        _textFusion.LoadWeights(w, "text_fusion");
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"transformer_blocks.{i}");
    }

    /// <summary>The number of streamable transformer blocks (28).</summary>
    public int BlockCount => _blocks.Length;

    /// <summary>The streamable block at <paramref name="idx"/>.</summary>
    /// <remarks>Returns the live block instance so <see cref="IStreamingBlock.EnumerateWeights"/> hands back the same
    /// tensor references every call — <c>BlockStreamingController</c> tracks residency by reference.</remarks>
    public IStreamingBlock GetBlock(int idx) => _blocks[idx];

    /// <summary>Hook invoked with each block index immediately before that block's forward pass; null = no streaming.</summary>
    /// <remarks>Pipelines plug a <c>BlockStreamingController</c> here so resident VRAM peaks at roughly
    /// (activations + the prefetch window) instead of the whole 13 GB fp8 DiT.
    /// <para><b>The captured step graph is disabled while this is non-null</b> (see <see cref="ForwardPatched"/>): a
    /// graph bakes weight device pointers and streaming re-points them every forward, so replay would read freed
    /// memory — a context-poisoning CUDA 700. Same guard as <c>HunyuanVideoDit</c> / <c>LtxVideo2Transformer</c>.</para></remarks>
    public Action<int>? BeforeBlockForward { get; set; }

    /// <summary>The non-block weights: <c>img_in</c>, the timestep MLP + modulation projection, the text-fusion stage,
    /// the text projection, and the final layer.</summary>
    /// <remarks>Touched on every forward regardless of which block runs, so they stay eagerly resident even when
    /// streaming. Unlike Qwen-Image's (which bracket its blocks), these are all a PREFIX of
    /// <see cref="EnumerateWeights"/> — the blocks are its tail — but the split is still a genuine partition.</remarks>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        Tensor?[] top =
        [
            _imgInW, _imgInB, _time1W, _time1B, _time2W, _time2B, _timeModW, _timeModB,
            _txtNormW, _txt1W, _txt1B, _txt2W, _txt2B, _finalTable, _finalNormW, _finalLinW, _finalLinB,
        ];
        foreach (Tensor? t in top) if (t is not null) yield return t;
        foreach (Tensor t in _textFusion.EnumerateWeights()) yield return t;
    }

    /// <summary>Enumerates every weight tensor for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in EnumerateSharedWeights()) yield return t;
        foreach (Krea2Block b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Predicts the flow-match velocity for one step.</summary>
    /// <param name="latent">Noisy latent <c>[1, 16, H, W]</c>.</param>
    /// <param name="timestep">Flow-match time <c>t</c> in <c>[0, 1]</c> (1 = noise, 0 = data).</param>
    /// <param name="encoderHidden">Tapped text hidden states <c>[1, txt_seq, numTextLayers · textHiddenDim]</c> (layer-major).</param>
    /// <returns>Velocity <c>[1, 16, H, W]</c>.</returns>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor encoderHidden)
    {
        ThrowIfDisposed();
        if (latent.Shape.Rank != 4 || latent.Shape[0] != 1)
            throw new ArgumentException($"latent must be [1, 16, H, W], got {latent.Shape}.", nameof(latent));

        int channels = (int)latent.Shape[1];
        int patch = _config.PatchSize;
        int hPacked = (int)latent.Shape[2] / patch;
        int wPacked = (int)latent.Shape[3] / patch;

        Tensor patchLatent = PatchifyLatent(latent);
        Tensor projected = ForwardPatched(backend, patchLatent, timestep, encoderHidden, hPacked, wPacked);
        patchLatent.Dispose();
        Tensor velocity = UnpatchifyChannelOuter(projected, 1, channels, hPacked, wPacked, patch);
        projected.Dispose();
        return velocity;
    }

    /// <summary>Patchifies a pixel latent <c>[1, C, H, W]</c> → the channel-outer token grid <c>[1, imgSeq, C·p²]</c>
    /// that the loop operates on. Host op; call once per generation.</summary>
    public Tensor PatchifyLatent(Tensor latent)
    {
        ThrowIfDisposed();
        int channels = (int)latent.Shape[1];
        int patch = _config.PatchSize;
        return PatchifyChannelOuter(latent, 1, channels, (int)latent.Shape[2], (int)latent.Shape[3], patch);
    }

    /// <summary>Unpatchifies the loop's token grid <c>[1, imgSeq, C·p²]</c> back to a pixel latent <c>[1, C, H, W]</c>.
    /// Host op; call once per generation.</summary>
    public Tensor UnpatchifyLatent(Tensor patchTokens, int hPacked, int wPacked)
    {
        ThrowIfDisposed();
        return UnpatchifyChannelOuter(patchTokens, 1, _config.VaeChannels, hPacked, wPacked, _config.PatchSize);
    }

    /// <summary>Denoise-step core operating entirely in patchified token space: consumes the patchified latent
    /// <c>[1, imgSeq, C·p²]</c> and returns the patchified flow-match velocity of the same shape — no per-step
    /// patchify/unpatchify. Keeping the latent in this form across the whole sampling loop (patchify once before,
    /// unpatchify once after) lets the scheduler step run on-device (Scale+Add), so a denoise step never reads any
    /// tensor's <c>DataPointer</c> and the host can queue all steps without the per-step D2H pipeline drain that
    /// otherwise serialized host dispatch against GPU execution.</summary>
    public Tensor ForwardPatched(IBackend backend, Tensor patchLatent, float timestep, Tensor encoderHidden,
        int hPacked, int wPacked, Utilities.DeviceFeatureCache? stepCache = null)
    {
        ThrowIfDisposed();
        const int batch = 1;
        int hidden = _config.HiddenSize;
        int imgSeq = hPacked * wPacked;
        int txtSeq = (int)encoderHidden.Shape[1];

        // ── timestep embedding + shared modulation ──
        Tensor temb = ComputeTimeEmbedding(backend, timestep, batch, hidden);          // [B, hidden]
        Tensor tembGelu = new Tensor(temb.Shape, DType.F32);
        backend.Gelu(tembGelu, temb);
        Tensor tembMod = new Tensor(new TensorShape(batch, 6 * hidden), DType.F32);
        backend.Linear(tembMod, tembGelu, _timeModW!, _timeModB);                       // [B, 6·hidden]
        tembGelu.Dispose();

        // ── text: fusion → projection (cached across steps — depends only on the prompt, not the timestep) ──
        bool txtCached = ReferenceEquals(_cachedTxtKey, encoderHidden) && _cachedTxt is not null;
        Tensor txt;
        if (txtCached)
        {
            txt = _cachedTxt!;
        }
        else
        {
            Tensor fused = _textFusion.Forward(backend, encoderHidden, batch, txtSeq);  // [B, S, textDim]
            Tensor computedTxt = ApplyTxtIn(backend, fused, batch, txtSeq, hidden);      // [B, S, hidden]
            fused.Dispose();
            _ = computedTxt.DataPointer;   // materialize to host once so it survives (and is re-uploaded) across steps
            if (_cachedTxt is not null)
            {
                backend.FreeWeights(new[] { _cachedTxt });   // no-op unless graph mode pinned it (see below)
                _cachedTxt.Dispose();
            }
            _cachedTxt = computedTxt;
            _cachedTxtKey = encoderHidden;
            txt = computedTxt;
        }

        // ── RoPE tables (host build, cached across steps) — hoisted ABOVE the capturable region ──
        long ropeSig = ((long)txtSeq * 73856093L) ^ ((long)hPacked * 19349663L) ^ ((long)wPacked * 83492791L);
        if (_ropeSig != ropeSig)
        {
            Tensor posIds = FluxRope.BuildPositionIds(txtSeq, hPacked, wPacked);
            _rope.Precompute(posIds);
            posIds.Dispose();
            _ropeSig = ropeSig;
        }

        // ── Step-graph mode (HARTSY_DIT_GRAPH): capture the fixed per-step region (img_in → blocks → final
        // layer) once and replay it with a single graph launch. Only the fast t2i path qualifies (the pipeline
        // routes the latent through PrepareGraphLatent → patchLatent IS _latentFixed); everything per-step-varying
        // is refreshed into fixed device buffers before the launch. Self-disables on capture failure or a
        // CFG-style signature flip storm.
        // An armed step cache is per-step-variable topology (hit vs miss) — a captured graph cannot replay it.
        // FULLY RESIDENT only (BeforeBlockForward null): block streaming re-points every block's weights each
        // forward, so a graph that baked their device pointers would replay against freed memory — a CUDA 700 that
        // poisons the whole context. Same guard as HunyuanVideoDit / LtxVideo2Transformer.
        bool graphMode = stepCache is null && BeforeBlockForward is null
            && DiTBlocks.DitStepGraph.Enabled && backend.StepGraphSupported
            && !_graphDead && ReferenceEquals(patchLatent, _latentFixed);
        if (!graphMode)
        {
            Tensor eager = ForwardCore(backend, patchLatent, txt, temb, tembMod, batch, imgSeq, txtSeq, hidden, stepCache);
            tembMod.Dispose();
            temb.Dispose();
            return eager;
        }

        // Refresh the fixed timestep buffers (the ONLY per-step-varying content the captured graph reads).
        _tembFixed ??= new Tensor(temb.Shape, DType.F32);
        backend.CopyInto(_tembFixed, temb);
        temb.Dispose();
        _tembModFixed ??= new Tensor(tembMod.Shape, DType.F32);
        backend.CopyInto(_tembModFixed, tembMod);
        tembMod.Dispose();

        long sig = ropeSig ^ ((long)System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(txt) << 17);
        // The backend graph slot is SHARED across models — if another transformer captured since our last
        // launch (models alternating under KEEP_MODELS), a "ready" graph is THEIRS: replaying it would run the
        // wrong model's step. Owner mismatch forces re-warm + re-capture and never counts as a CFG flip.
        bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, this);
        if (sig != _graphSig || ownerLost)
        {
            backend.StepGraphReset();
            if (!ownerLost && _graphSig != long.MinValue && ++_graphSigFlips > 8)
            {
                _graphDead = true;   // alternating signatures (CFG cond/uncond) — capture can't converge
                HartsyInference.Core.Logging.Logs.Warning("[Krea2 graph] signature flip storm — step-graph disabled for this session.");
            }
            _graphSig = sig;
            _graphSigCalls = 0;
            backend.StepGraphOwner = this;
        }
        _graphSigCalls++;
        if (_graphSigCalls == 1)
        {
            // Pin txt as a device-resident weight: the host-materialized txt cache otherwise re-uploads from
            // PAGEABLE memory every step — an internally-synchronizing copy that is capture-ILLEGAL (this was
            // the CUDA_ERROR_STREAM_CAPTURE_INVALIDATED at 99% VRAM, where auto-promotion's headroom gate
            // blocks the pin). PreloadWeights is a sync alloc, so it must happen here, OUTSIDE the capture.
            backend.PreloadWeights(new[] { txt });
        }
        _graphVelocity ??= new Tensor(new TensorShape(batch, imgSeq, _config.InChannels), DType.F32);

        if (backend.StepGraphReady && _graphSigCalls > GraphCaptureCall)
        {
            backend.StepGraphLaunch();
            return _graphVelocity;
        }

        // Calls 1..2 at this sig run eagerly THROUGH THE FIXED BUFFERS (warms txt auto-promotion + rope-table
        // uploads so nothing inside the capture does a synchronous alloc); call 3 records the same sequence.
        bool capture = _graphSigCalls == GraphCaptureCall;
        if (capture) backend.StepGraphBegin();
        try
        {
            Tensor projected = ForwardCore(backend, patchLatent, txt, _tembFixed, _tembModFixed, batch, imgSeq, txtSeq, hidden);
            backend.CopyInto(_graphVelocity, projected);   // last captured op: land the velocity at a fixed, normal buffer
            projected.Dispose();
        }
        catch (Exception ex) when (capture)
        {
            // A capture-illegal op invalidated the recording (nothing executed). Abort the capture, disable
            // graph mode for the session, and re-run this step eagerly so the generation stays correct.
            backend.StepGraphReset();
            _graphDead = true;
            HartsyInference.Core.Logging.Logs.Warning($"[Krea2 graph] capture invalidated — falling back to eager: {ex}");
            Tensor projected = ForwardCore(backend, patchLatent, txt, _tembFixed, _tembModFixed, batch, imgSeq, txtSeq, hidden);
            backend.CopyInto(_graphVelocity, projected);
            projected.Dispose();
            return _graphVelocity;
        }
        if (capture)
        {
            try
            {
                backend.StepGraphEndAndLaunch();   // capture records without executing — this runs the step
                HartsyInference.Core.Logging.Logs.Info("[Krea2 graph] denoise step captured; replaying via cuGraphLaunch.");
            }
            catch (Exception ex)
            {
                // Instantiation failed (some op wasn't capturable): the recorded work never executed. Fall back
                // permanently and re-run this step eagerly so the generation stays correct.
                backend.StepGraphReset();
                _graphDead = true;
                HartsyInference.Core.Logging.Logs.Warning($"[Krea2 graph] capture failed — falling back to eager: {ex.Message}");
                Tensor projected = ForwardCore(backend, patchLatent, txt, _tembFixed, _tembModFixed, batch, imgSeq, txtSeq, hidden);
                backend.CopyInto(_graphVelocity, projected);
                projected.Dispose();
            }
        }
        return _graphVelocity;
    }

    /// <summary>Invalidates the captured step graph. MUST be called whenever the transformer's weights are freed
    /// (the pipeline's post-loop <c>FreeWeights</c> when models aren't kept resident): the captured graph bakes
    /// the WEIGHT device pointers, so a free + next-gen re-upload leaves it pointing at freed memory — replaying
    /// it then is a CUDA 700 illegal-address that poisons the whole context (found by the fleet benchmark, where
    /// models rotate and weights evict between gens). The next generation re-warms and re-captures.</summary>
    public void InvalidateStepGraph(IBackend backend)
    {
        backend.StepGraphReset();
        if (ReferenceEquals(backend.StepGraphOwner, this))
            backend.StepGraphOwner = null;
        _graphSig = long.MinValue;   // MinValue = "no sig": the next call resets WITHOUT counting a CFG flip
        _graphSigCalls = 0;
    }

    /// <summary>Routes a fresh patchified latent into the step-graph's FIXED latent buffer (the address the
    /// captured graph reads and the pipeline's in-place Euler updates). Returns the fixed tensor — owned by the
    /// transformer; the pipeline must not dispose it or read its DataPointer (snapshot via
    /// <see cref="SnapshotGraphLatent"/> instead). A resolution change resets the graph.</summary>
    public Tensor PrepareGraphLatent(IBackend backend, Tensor freshPatchLatent)
    {
        ThrowIfDisposed();
        if (_latentFixed is not null && _latentFixed.Shape != freshPatchLatent.Shape)
        {
            backend.StepGraphReset();
            _graphSig = long.MinValue;
            _latentFixed.Dispose();
            _latentFixed = null;
            _graphVelocity?.Dispose();
            _graphVelocity = null;
        }
        _latentFixed ??= new Tensor(freshPatchLatent.Shape, DType.F32);
        backend.CopyInto(_latentFixed, freshPatchLatent);
        return _latentFixed;
    }

    /// <summary>Device-copies the fixed latent into a fresh tensor the caller may freely read/dispose (reading
    /// the fixed tensor's DataPointer directly would D2H-and-FREE the buffer the captured graph points at).</summary>
    public Tensor SnapshotGraphLatent(IBackend backend)
    {
        ThrowIfDisposed();
        Tensor snap = new Tensor(_latentFixed!.Shape, DType.F32);
        backend.CopyInto(snap, _latentFixed);
        return snap;
    }

    /// <summary>The fixed per-step region: img_in → concat[text,image] → (F16 cast) → 28 blocks → tail slice →
    /// (F32 cast) → final layer. Identical op sequence every step for a given (txt, resolution) — the property
    /// that makes it CUDA-graph-capturable. Caller owns temb/tembMod.</summary>
    private Tensor ForwardCore(IBackend backend, Tensor patchLatent, Tensor txt, Tensor temb, Tensor tembMod,
        int batch, int imgSeq, int txtSeq, int hidden, Utilities.DeviceFeatureCache? stepCache = null)
    {
        // ── image: img_in (patchLatent is already the [1, imgSeq, C·p²] token grid) ──
        Tensor img = new Tensor(new TensorShape(batch, imgSeq, hidden), DType.F32);
        backend.Linear(img, patchLatent, _imgInW!, _imgInB);

        // ── concat [text, image] (device concat — see the GPU-residency notes in git history) ──
        int jointSeq = txtSeq + imgSeq;
        Tensor joint = new Tensor(new TensorShape(batch, jointSeq, hidden), DType.F32);
        backend.Concat(joint, new[] { txt, img }, dim: 1);
        img.Dispose();

        // F16 hot path (HARTSY_DIT_F16): one cast into F16 before the 28-block loop — the blocks and attention
        // then run entirely in F16 (half the HBM traffic of the bandwidth-bound glue kernels). The once-per-forward
        // text/image/timestep paths stay F32; the tail is cast back after the loop.
        if (DiTBlocks.DitDtype.Act == DType.F16)
        {
            Tensor jointF16 = new Tensor(joint.Shape, DType.F16);
            backend.CastToF16(jointF16, joint);
            joint.Dispose();
            joint = jointF16;
        }

        // Across-step First-Block cache (QwenImageTransformer wiring; see DeviceFeatureCache): block 0 always
        // runs as the gate indicator; hit ⇒ blocks 1..N−1 replaced by block0 + previous residual; miss ⇒ the
        // anchor survives the loop for the fresh residual. Null stepCache = byte-identical original loop.
        Tensor? cacheAnchor = null;
        int startBlock = 0;
        if (stepCache is not null && _blocks.Length > 1)
        {
            BeforeBlockForward?.Invoke(0);
            Tensor block0 = _blocks[0].Forward(backend, joint, tembMod, _rope, batch, jointSeq);
            joint.Dispose();
            joint = block0;
            startBlock = 1;
            if (!stepCache.ShouldCompute(backend, joint))
            {
                Tensor reconstructed = stepCache.ApplyResidual(backend, joint);
                joint.Dispose();
                joint = reconstructed;
                startBlock = _blocks.Length;
            }
            else
            {
                cacheAnchor = joint;
            }
        }

        for (int i = startBlock; i < _blocks.Length; i++)
        {
            // A step-cache HIT sets startBlock = _blocks.Length, skipping this loop and its streaming hook entirely.
            // The controller simply keeps whatever it had prefetched (bounded by the prefetch window) and the next
            // miss resumes from there — residency is per-block state, not a position in a sequence.
            BeforeBlockForward?.Invoke(i);
            Tensor next = _blocks[i].Forward(backend, joint, tembMod, _rope, batch, jointSeq);
            if (joint != cacheAnchor) joint.Dispose();
            joint = next;
        }

        if (cacheAnchor is not null)
        {
            stepCache!.StoreResidual(backend, cacheAnchor, joint);
            cacheAnchor.Dispose();
        }

        // ── strip text prefix, final layer (device-resident; returns the patchified velocity) ──
        Tensor imgTail = SliceTail(backend, joint, txtSeq, imgSeq, hidden);
        joint.Dispose();
        if (imgTail.DType == DType.F16)
        {
            // Back to F32 for the final layer + the on-device Euler step (velocity precision matters across steps).
            Tensor tailF32 = new Tensor(imgTail.Shape, DType.F32);
            backend.CastToF32(tailF32, imgTail);
            imgTail.Dispose();
            imgTail = tailF32;
        }
        Tensor projected = ApplyFinalLayer(backend, imgTail, temb, batch, imgSeq, hidden);
        imgTail.Dispose();
        return projected;
    }

    /// <summary>Sinusoidal(t·1000, cos-first, dim 256) → Linear → gelu_tanh → Linear. Returns <c>[B, hidden]</c>.</summary>
    private Tensor ComputeTimeEmbedding(IBackend backend, float timestep, int batch, int hidden)
    {
        int freqDim = _config.TimestepEmbedDim;
        Tensor sin = new Tensor(new TensorShape(batch, freqDim), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sin, timestep * 1000.0f, batch, freqDim);
        Tensor h1 = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.Linear(h1, sin, _time1W!, _time1B);
        sin.Dispose();
        Tensor act = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.Gelu(act, h1);
        h1.Dispose();
        Tensor outp = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.Linear(outp, act, _time2W!, _time2B);
        act.Dispose();
        return outp;
    }

    /// <summary>txt_in: zero-center RMSNorm(textDim) → Linear(textDim→hidden) → gelu_tanh → Linear(hidden→hidden).</summary>
    private Tensor ApplyTxtIn(IBackend backend, Tensor fused, int batch, int seqLen, int hidden)
    {
        int textDim = _config.TextHiddenDim;
        Tensor normed = new Tensor(new TensorShape(batch, seqLen, textDim), DType.F32);
        backend.RmsNorm(normed, fused, _txtNormW!, _config.NormEps);
        Tensor h1 = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.Linear(h1, normed, _txt1W!, _txt1B);
        normed.Dispose();
        Tensor act = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.Gelu(act, h1);
        h1.Dispose();
        Tensor outp = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.Linear(outp, act, _txt2W!, _txt2B);
        act.Dispose();
        return outp;
    }

    /// <summary>final_layer: <c>scale = temb + table[0]</c>, <c>shift = temb + table[1]</c>;
    /// <c>(1+scale)·RMSNorm(h) + shift</c> → <c>Linear(hidden → p²·channels)</c>.</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor h, Tensor temb, int batch, int seqLen, int hidden)
    {
        Tensor normed = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.RmsNorm(normed, h, _finalNormW!, _config.NormEps);

        // scale = temb + table[0], shift = temb + table[1] (both [B, hidden]); (1+scale)·normed + shift — all on
        // device (was a per-step D2H drain + a seqLen·hidden host loop reading normed/temb.DataPointer). B=1 path;
        // B>1 (unused by pipelines) keeps the host loop.
        if (batch == 1)
        {
            Tensor tabScale = new Tensor(new TensorShape(1, hidden), DType.F32);
            backend.SliceRows(tabScale, _finalTable!, 0);
            Tensor tabShift = new Tensor(new TensorShape(1, hidden), DType.F32);
            backend.SliceRows(tabShift, _finalTable!, 1);
            Tensor scale = new Tensor(new TensorShape(1, hidden), DType.F32);
            backend.Add(scale, temb, tabScale);
            Tensor shift = new Tensor(new TensorShape(1, hidden), DType.F32);
            backend.Add(shift, temb, tabShift);
            tabScale.Dispose(); tabShift.Dispose();

            Tensor modulatedDev = DiTUtils.Modulate(backend, normed, shift, scale, new TensorShape(batch, seqLen, hidden));
            normed.Dispose(); scale.Dispose(); shift.Dispose();

            Tensor projectedDev = new Tensor(new TensorShape(batch, seqLen, _config.InChannels), DType.F32);
            backend.Linear(projectedDev, modulatedDev, _finalLinW!, _finalLinB);
            modulatedDev.Dispose();
            return projectedDev;
        }

        Tensor modulated = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        float* np = (float*)normed.DataPointer, mp = (float*)modulated.DataPointer;
        float* tp = (float*)temb.DataPointer, tab = (float*)_finalTable!.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long tb = (long)b * hidden;
            for (int s = 0; s < seqLen; s++)
            {
                long vb = ((long)b * seqLen + s) * hidden;
                for (int d = 0; d < hidden; d++)
                {
                    float scale = tp[tb + d] + tab[d];
                    float shift = tp[tb + d] + tab[hidden + d];
                    mp[vb + d] = (1.0f + scale) * np[vb + d] + shift;
                }
            }
        }
        normed.Dispose();

        int outDim = _config.InChannels; // p²·channels = 64
        Tensor projected = new Tensor(new TensorShape(batch, seqLen, outDim), DType.F32);
        backend.Linear(projected, modulated, _finalLinW!, _finalLinB);
        modulated.Dispose();
        return projected;
    }

    /// <summary>Patchifies <c>[B, C, H, W]</c> → <c>[B, (H/p)(W/p), C·p²]</c> with <b>channel-outer</b> feature order
    /// <c>(c, ph, pw)</c> (diffusers <c>view → permute(0,2,4,1,3,5) → reshape</c>).</summary>
    private static Tensor PatchifyChannelOuter(Tensor latent, int batch, int channels, int height, int width, int patch)
    {
        int hPacked = height / patch, wPacked = width / patch;
        int imgSeq = hPacked * wPacked, patchVol = channels * patch * patch;
        Tensor result = new Tensor(new TensorShape(batch, imgSeq, patchVol), DType.F32);
        float* src = (float*)latent.DataPointer, dst = (float*)result.DataPointer;
        long chw = (long)channels * height * width, hw = (long)height * width;
        for (int b = 0; b < batch; b++)
            for (int hp = 0; hp < hPacked; hp++)
                for (int wp = 0; wp < wPacked; wp++)
                {
                    float* tok = dst + (((long)b * imgSeq) + ((long)hp * wPacked + wp)) * patchVol;
                    int idx = 0;
                    for (int c = 0; c < channels; c++)
                        for (int ph = 0; ph < patch; ph++)
                            for (int pw = 0; pw < patch; pw++)
                                tok[idx++] = src[b * chw + c * hw + (long)(hp * patch + ph) * width + (wp * patch + pw)];
                }
        return result;
    }

    /// <summary>Inverse of <see cref="PatchifyChannelOuter"/>: <c>[B, (H/p)(W/p), C·p²]</c> → <c>[B, C, H, W]</c>.</summary>
    private static Tensor UnpatchifyChannelOuter(Tensor tokens, int batch, int channels, int hPacked, int wPacked, int patch)
    {
        int height = hPacked * patch, width = wPacked * patch;
        int imgSeq = hPacked * wPacked, patchVol = channels * patch * patch;
        Tensor result = new Tensor(new TensorShape(batch, channels, height, width), DType.F32);
        float* src = (float*)tokens.DataPointer, dst = (float*)result.DataPointer;
        long chw = (long)channels * height * width, hw = (long)height * width;
        for (int b = 0; b < batch; b++)
            for (int hp = 0; hp < hPacked; hp++)
                for (int wp = 0; wp < wPacked; wp++)
                {
                    float* tok = src + (((long)b * imgSeq) + ((long)hp * wPacked + wp)) * patchVol;
                    int idx = 0;
                    for (int c = 0; c < channels; c++)
                        for (int ph = 0; ph < patch; ph++)
                            for (int pw = 0; pw < patch; pw++)
                                dst[b * chw + c * hw + (long)(hp * patch + ph) * width + (wp * patch + pw)] = tok[idx++];
                }
        return result;
    }

    // Device-resident tail slice: copy the image rows [start, start+tailLen) of the joint sequence via the backend's
    // SliceRows (a contiguous row-block copy) so the last block's output never leaves the GPU (was a joint.DataPointer
    // D2H drain + host memcpy at the end of every step). Follows the joint's dtype (F16 on the HARTSY_DIT_F16 path).
    private static Tensor SliceTail(IBackend backend, Tensor joint, int start, int tailLen, int hidden)
    {
        Tensor output = new Tensor(new TensorShape(1, tailLen, hidden), joint.DType);
        backend.SliceRows(output, joint, start);
        return output;
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _cachedTxt?.Dispose();
            _cachedTxt = null;
            _cachedTxtKey = null;
            _latentFixed?.Dispose();
            _latentFixed = null;
            _tembFixed?.Dispose();
            _tembFixed = null;
            _tembModFixed?.Dispose();
            _tembModFixed = null;
            _graphVelocity?.Dispose();
            _graphVelocity = null;
            _imgInW = _imgInB = _time1W = _time1B = _time2W = _time2B = _timeModW = _timeModB = null;
            _txtNormW = _txt1W = _txt1B = _txt2W = _txt2B = null;
            _finalTable = _finalNormW = _finalLinW = _finalLinB = null;
        }
        GC.SuppressFinalize(this);
    }
}
