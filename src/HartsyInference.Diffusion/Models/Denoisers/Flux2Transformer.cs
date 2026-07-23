using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Flux.2 Diffusion Transformer (Klein 4B / Klein 9B / Dev). Distinct from <see cref="FluxTransformer"/>: LayerNorm for stream norms, top-level shared modulation projections (one Linear per stream type, output reused across all blocks), 4-axis RoPE with theta=2000, parallel single-stream block (fused QKV+MLP), SwiGLU MLP in both block types. Reference: <c>diffusers.transformer_flux2.Flux2Transformer2DModel</c>.</summary>
public sealed unsafe class Flux2Transformer : IDisposable
{
    private readonly Flux2Config _config;
    private readonly Flux2DoubleBlock[] _doubleBlocks;
    private readonly Flux2SingleBlock[] _singleBlocks;
    private readonly FluxRope _rope;

    // Shared modulation projections (top-level — one set per stream type, reused across all blocks)
    private readonly AdaLNModulation _doubleModImg;   // 6 params: shift/scale/gate × {msa, mlp}
    private readonly AdaLNModulation _doubleModTxt;   // 6 params
    private readonly AdaLNModulation _singleMod;      // 3 params: shift, scale, gate

    // Input projections (no bias for Flux.2)
    private Tensor? _xEmbedWeight;
    private Tensor? _contextEmbedWeight;

    // Time-only MLP (Klein) or time + guidance MLPs (Dev)
    private Tensor? _timestepLinear1Weight;
    private Tensor? _timestepLinear2Weight;
    private Tensor? _guidanceLinear1Weight;
    private Tensor? _guidanceLinear2Weight;

    // Final layer: AdaLN-Continuous (shift, scale only — no gate) + proj_out
    private Tensor? _normOutLinearWeight;
    private Tensor? _projOutWeight;

    /// <summary>True when this instance runs the audited F16 block loop (HARTSY_DIT_F16) with the exact
    /// <see cref="ChromaF16.ResidualDamp"/> residual damp — every branch input passes a no-affine LayerNorm
    /// and the final AdaLN-continuous norm cancels the factor before proj_out (the Chroma/Flux.1 recipe).</summary>
    private bool _f16Mode;

    // True when any loaded weight is block-quantized (GGUF Q4_K/... or nvfp4) — the transient-dequant regime
    // that decides the step-graph default (see StepGraphEnabled).
    private bool _hasQuantizedWeights;

    // Rope-table signature: Precompute rebuilds host trig tables AND re-uploads the GPU cos/sin tables —
    // positions only change with resolution / text length.
    private long _ropeSig = long.MinValue;

    private int _disposed;

    public Flux2Transformer(Flux2Config config)
    {
        _config = config;

        int mlpInner = (int)(config.HiddenSize * config.MlpRatio);

        _doubleBlocks = new Flux2DoubleBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++)
        {
            _doubleBlocks[i] = new Flux2DoubleBlock(
                config.HiddenSize, config.NumHeads, mlpInner,
                config.QkvBias, config.QkNormEps, config.LayerNormEps);
        }

        _singleBlocks = new Flux2SingleBlock[config.DepthSingleBlocks];
        for (int i = 0; i < config.DepthSingleBlocks; i++)
        {
            _singleBlocks[i] = new Flux2SingleBlock(
                config.HiddenSize, config.NumHeads, mlpInner,
                config.QkvBias, config.QkNormEps, config.LayerNormEps);
        }

        _rope = new FluxRope(config.AxesDim, config.Theta);

        _doubleModImg = new AdaLNModulation(config.HiddenSize, 6);
        _doubleModTxt = new AdaLNModulation(config.HiddenSize, 6);
        _singleMod = new AdaLNModulation(config.HiddenSize, 3);
    }

    /// <summary>Loads weights using the canonical naming emitted by <c>Flux2CheckpointConverter</c>. Follows the diffusers Flux2 module hierarchy except where the converter has split fused weights (see <see cref="Flux2DoubleBlock"/> and <see cref="Flux2SingleBlock"/>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _f16Mode = DitDtype.Act == DType.F16;
        float branchDamp = _f16Mode ? ChromaF16.ResidualDamp : 1.0f;

        _xEmbedWeight = weights["x_embedder.weight"];
        _contextEmbedWeight = weights["context_embedder.weight"];
        if (_f16Mode)
        {
            // Enter the damped-residual regime at the (bias-less) embedders; block-output damping keeps the
            // stream there; the final no-affine LayerNorm cancels the factor exactly.
            _xEmbedWeight.Fp8ScaleFactor *= ChromaF16.ResidualDamp;
            _contextEmbedWeight.Fp8ScaleFactor *= ChromaF16.ResidualDamp;
            Logs.Info($"[Flux2] F16 block loop active (residual damp 1/{1.0f / ChromaF16.ResidualDamp:F0})");
        }

        _timestepLinear1Weight = weights["time_guidance_embed.timestep_embedder.linear_1.weight"];
        _timestepLinear2Weight = weights["time_guidance_embed.timestep_embedder.linear_2.weight"];

        if (_config.GuidanceEmbed)
        {
            _guidanceLinear1Weight = weights["time_guidance_embed.guidance_embedder.linear_1.weight"];
            _guidanceLinear2Weight = weights["time_guidance_embed.guidance_embedder.linear_2.weight"];
        }

        // Shared modulation projections — produce output reused across all blocks of the same type
        _doubleModImg.LoadWeights(weights["double_stream_modulation_img.linear.weight"], null);
        _doubleModTxt.LoadWeights(weights["double_stream_modulation_txt.linear.weight"], null);
        _singleMod.LoadWeights(weights["single_stream_modulation.linear.weight"], null);

        for (int i = 0; i < _config.Depth; i++)
            _doubleBlocks[i].LoadWeights(weights, $"transformer_blocks.{i}", branchDamp);

        for (int i = 0; i < _config.DepthSingleBlocks; i++)
            _singleBlocks[i].LoadWeights(weights, $"single_transformer_blocks.{i}", branchDamp);

        _normOutLinearWeight = weights["norm_out.linear.weight"];
        _projOutWeight = weights["proj_out.weight"];

        _hasQuantizedWeights = false;
        foreach (Tensor w in EnumerateWeights())
        {
            if (w.DType.IsQuantized)
            {
                _hasQuantizedWeights = true;
                break;
            }
        }
        if (_hasQuantizedWeights && !DitStepGraph.Enabled)
            Logs.Info("[Flux2] quantized DiT (transient per-GEMM dequant) — persistent step graph is opt-in (HARTSY_DIT_GRAPH=1); eager loop by default.");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_xEmbedWeight is not null) yield return _xEmbedWeight;
        if (_contextEmbedWeight is not null) yield return _contextEmbedWeight;
        if (_timestepLinear1Weight is not null) yield return _timestepLinear1Weight;
        if (_timestepLinear2Weight is not null) yield return _timestepLinear2Weight;
        if (_guidanceLinear1Weight is not null) yield return _guidanceLinear1Weight;
        if (_guidanceLinear2Weight is not null) yield return _guidanceLinear2Weight;
        if (_normOutLinearWeight is not null) yield return _normOutLinearWeight;
        if (_projOutWeight is not null) yield return _projOutWeight;
        foreach (Tensor w in _doubleModImg.EnumerateWeights()) yield return w;
        foreach (Tensor w in _doubleModTxt.EnumerateWeights()) yield return w;
        foreach (Tensor w in _singleMod.EnumerateWeights()) yield return w;
        for (int i = 0; i < _doubleBlocks.Length; i++)
            foreach (Tensor w in _doubleBlocks[i].EnumerateWeights()) yield return w;
        for (int i = 0; i < _singleBlocks.Length; i++)
            foreach (Tensor w in _singleBlocks[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="packedLatent">Packed latent tokens <c>[B, imgSeqLen, in_channels=128]</c>.</param>
    /// <param name="textEmbeddings">Text embeddings <c>[B, txtSeqLen, joint_attention_dim=7680]</c> (Qwen3 multi-layer concat for Klein).</param>
    /// <param name="sigma">Current sigma (noise level, 0-1 range). Pipeline passes <c>timestep / 1000</c> here so we re-scale by 1000 internally to match diffusers.</param>
    /// <param name="guidanceScale">Guidance scale (Dev only, embedded via MLP). Ignored when <see cref="Flux2Config.GuidanceEmbed"/> is false.</param>
    /// <param name="hPacked">Patchified latent height (= image_height / 16).</param>
    /// <param name="wPacked">Patchified latent width (= image_width / 16).</param>
    /// <returns>Predicted velocity <c>[B, imgSeqLen, out_channels=128]</c>.</returns>
    public Tensor Forward(IBackend backend, Tensor packedLatent, Tensor textEmbeddings,
        float sigma, float guidanceScale, int hPacked, int wPacked,
        Utilities.DeviceFeatureCache? stepCache = null)
    {
        Tensor tembOuter = ComputeTimestepEmbedding(backend, sigma, guidanceScale, (int)packedLatent.Shape[0]);
        Tensor velocity = ForwardWithTemb(backend, packedLatent, textEmbeddings, tembOuter, hPacked, wPacked, stepCache);
        tembOuter.Dispose();
        return velocity;
    }

    /// <summary>Forward body with a caller-owned temb (shared by the eager and step-graph paths — the graph
    /// path computes temb inside the capture from a fixed device sin buffer). Does NOT dispose temb.
    /// <para><paramref name="stepCache"/> = the original FBCache target shape: double block 0 always runs and
    /// its IMG stream gates; on a hit the remaining doubles + concat + all single blocks are replaced by
    /// block0Img + the previous step's img-portion residual (stored against the pre-final-layer img hidden,
    /// which has block0Img's exact shape/dtype). Null = byte-identical uncached forward.</para></summary>
    private Tensor ForwardWithTemb(IBackend backend, Tensor packedLatent, Tensor textEmbeddings, Tensor temb,
        int hPacked, int wPacked, Utilities.DeviceFeatureCache? stepCache = null)
    {
        int batch = (int)packedLatent.Shape[0];
        int imgSeqLen = (int)packedLatent.Shape[1];
        int txtSeqLen = (int)textEmbeddings.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;
        int hidden = _config.HiddenSize;

        // ── 1. Project image and text tokens into hidden dim ──
        TensorShape imgTokShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgTokens = new Tensor(imgTokShape, DType.F32);
        backend.Linear(imgTokens, packedLatent, _xEmbedWeight!, null);

        TensorShape txtTokShape = new TensorShape(batch, txtSeqLen, hidden);
        Tensor txtTokens = new Tensor(txtTokShape, DType.F32);
        backend.Linear(txtTokens, textEmbeddings, _contextEmbedWeight!, null);

        // ── 2. F16 block loop (HARTSY_DIT_F16, B=1): one cast per stream before the loop; every block
        //       activation follows; streams already ride at ResidualDamp scale from the damped embedders.
        //       Cast back to F32 after the loop for the final norm (which cancels the damp). ──
        bool f16Loop = _f16Mode && batch == 1;
        if (f16Loop)
        {
            Tensor imgF16 = new Tensor(imgTokShape, DType.F16);
            backend.CastToF16(imgF16, imgTokens);
            imgTokens.Dispose();
            imgTokens = imgF16;
            Tensor txtF16 = new Tensor(txtTokShape, DType.F16);
            backend.CastToF16(txtF16, txtTokens);
            txtTokens.Dispose();
            txtTokens = txtF16;
        }
        DType act = imgTokens.DType;

        // ── 3. Shared modulation projections — computed once, reused across all blocks ──
        Tensor[] imgMod = _doubleModImg.Forward(backend, temb);   // 6 tensors [B, hidden]
        Tensor[] txtMod = _doubleModTxt.Forward(backend, temb);   // 6 tensors
        Tensor[] sgMod = _singleMod.Forward(backend, temb);       // 3 tensors

        // ── 4. Precompute 4-axis RoPE — sig-cached (Precompute rebuilds host tables + re-uploads the GPU
        //       cos/sin tables; it ran EVERY forward before) ──
        EnsureRope(txtSeqLen, hPacked, wPacked);

        // ── 5. Double-stream blocks (text + image as two parallel streams sharing joint attn) ──
        // Across-step First-Block cache: double block 0 always runs; its img stream is the gate indicator.
        // On a hit the rest of the stack (doubles 1..N + concat + singles + strip) is replaced by
        // block0Img + the cached img-portion residual. On a miss the anchor survives to StoreResidual below.
        Tensor currentImg = imgTokens;
        Tensor currentTxt = txtTokens;
        Tensor? cacheAnchor = null;
        bool cacheHit = false;
        int startDouble = 0;
        TensorShape imgOutShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgOut;
        if (stepCache is not null && _config.Depth > 0)
        {
            (Tensor img0, Tensor txt0) = _doubleBlocks[0].Forward(
                backend, currentImg, currentTxt, imgMod, txtMod, _rope);
            if (!ReferenceEquals(currentImg, imgTokens)) currentImg.Dispose();
            if (!ReferenceEquals(currentTxt, txtTokens)) currentTxt.Dispose();
            currentImg = img0;
            currentTxt = txt0;
            startDouble = 1;
            cacheHit = !stepCache.ShouldCompute(backend, currentImg);
            if (!cacheHit) cacheAnchor = currentImg;
        }

        if (cacheHit)
        {
            imgOut = stepCache!.ApplyResidual(backend, currentImg);
            currentImg.Dispose();
            currentTxt.Dispose();
            imgTokens.Dispose();
            txtTokens.Dispose();
        }
        else
        {
            for (int i = startDouble; i < _config.Depth; i++)
            {
                (Tensor newImg, Tensor newTxt) = _doubleBlocks[i].Forward(
                    backend, currentImg, currentTxt, imgMod, txtMod, _rope);
                if (!ReferenceEquals(currentImg, imgTokens) && !ReferenceEquals(currentImg, cacheAnchor)) currentImg.Dispose();
                if (!ReferenceEquals(currentTxt, txtTokens)) currentTxt.Dispose();
                currentImg = newImg;
                currentTxt = newTxt;
            }

            // ── 6. Concatenate [text, image] for single-stream processing (device op — the old host copy was a
            // full D2H sync of both block-loop outputs every forward) ──
            TensorShape concatShape = new TensorShape(batch, totalSeqLen, hidden);
            Tensor x = new Tensor(concatShape, act);
            backend.Concat(x, new Tensor[] { currentTxt, currentImg }, 1);
            if (!ReferenceEquals(currentImg, imgTokens) && !ReferenceEquals(currentImg, cacheAnchor)) currentImg.Dispose();
            if (!ReferenceEquals(currentTxt, txtTokens)) currentTxt.Dispose();
            imgTokens.Dispose();
            txtTokens.Dispose();

            // ── 7. Single-stream blocks (parallel attn+MLP on full concat sequence) ──
            for (int i = 0; i < _config.DepthSingleBlocks; i++)
            {
                Tensor newX = _singleBlocks[i].Forward(backend, x, sgMod, _rope);
                x.Dispose();
                x = newX;
            }

            // ── 8. Strip text prefix → image-only tokens (B=1: contiguous row-block → device SliceRows;
            // batched keeps the host copy) ──
            imgOut = new Tensor(imgOutShape, act);
            if (batch == 1)
                backend.SliceRows(imgOut, x, txtSeqLen);
            else
                ExtractImageTokens(imgOut, x, batch, txtSeqLen, imgSeqLen, hidden);
            x.Dispose();

            if (cacheAnchor is not null)
            {
                stepCache!.StoreResidual(backend, cacheAnchor, imgOut);
                cacheAnchor.Dispose();
            }
        }

        if (imgOut.DType == DType.F16)
        {
            // Back to F32 for the final norm + proj_out (velocity precision across Euler steps).
            Tensor imgOutF32 = new Tensor(imgOutShape, DType.F32);
            backend.CastToF32(imgOutF32, imgOut);
            imgOut.Dispose();
            imgOut = imgOutF32;
        }

        // ── 9. Final layer: AdaLN-Continuous (shift/scale only) + proj_out ──
        Tensor output = ApplyFinalLayer(backend, imgOut, temb, batch, imgSeqLen);
        imgOut.Dispose();
        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();
        for (int i = 0; i < sgMod.Length; i++) sgMod[i].Dispose();

        return output;
    }

    /// <summary>Precomputes the rope tables only when the (text len, grid) signature changes.</summary>
    private void EnsureRope(int txtSeqLen, int hPacked, int wPacked)
    {
        long sig = ((long)txtSeqLen << 32) ^ ((long)hPacked << 16) ^ (long)wPacked ^ 0x2F2F2F2F;
        if (sig == _ropeSig)
            return;
        Tensor posIds = Flux2PosEmbed.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        _rope.Precompute(posIds);
        posIds.Dispose();
        _ropeSig = sig;
    }

    // ── Persistent step-graph state (the Chroma round-3 recipe, single-forward variant — Klein is
    // distilled/no-CFG, Dev is guidance-embedded). Same contract as FluxTransformer: the pipeline routes the
    // latent through PrepareGraphLatent, never sweeps activations on the graph route, and invalidates before
    // any FreeWeights. ──
    private const int GraphCaptureCall = 3;

    /// <summary>Per-instance step-graph gate. Non-quant weights (Klein BF16 → cached F16 casts) keep the validated
    /// default-ON graph. GGUF/nvfp4-quantized weights (Dev Q4_K, the 24 GB recipe — a cached F16 cast of the 32B DiT
    /// cannot fit) dequantize TRANSIENTLY per GEMM, so during capture every per-Linear dequant alloc/free becomes a
    /// graph-memory node and instantiate must physically reserve that high-water beside the ~18 GB resident quant
    /// weights — measured capture OOM on 24 GB (worklog 2026-07-10), eager fallback every session. Quantized
    /// checkpoints therefore require the explicit <c>HARTSY_DIT_GRAPH=1</c> opt-in; the opt-in capture path
    /// pre-trims the eager pool (see <see cref="ForwardGraphable"/>) to give instantiate the best headroom.</summary>
    public bool StepGraphEnabled => _hasQuantizedWeights ? DitStepGraph.Enabled : DitStepGraph.EnabledDefaultOn;

    private Tensor? _latentFixed;
    private Tensor? _sinFixed;
    private Tensor? _guidanceSinFixed;
    private Tensor? _graphVelocity;
    private long _graphSig = long.MinValue;
    private int _graphSigCalls;
    private int _graphSigFlips;
    private bool _graphDead;

    /// <summary>Copies a fresh packed latent into the transformer-owned FIXED buffer the captured graph
    /// reads, and returns that buffer. A shape change resets the graph.</summary>
    public Tensor PrepareGraphLatent(IBackend backend, Tensor freshPackedLatent)
    {
        if (_latentFixed is not null && _latentFixed.Shape != freshPackedLatent.Shape)
        {
            InvalidateStepGraph(backend);
            _latentFixed.Dispose();
            _latentFixed = null;
            _graphVelocity?.Dispose();
            _graphVelocity = null;
        }
        _latentFixed ??= new Tensor(freshPackedLatent.Shape, DType.F32);
        backend.CopyInto(_latentFixed, freshPackedLatent);
        return _latentFixed;
    }

    /// <summary>Device copy of the fixed graph latent (for the pipeline's final read-back).</summary>
    public Tensor SnapshotGraphLatent(IBackend backend)
    {
        Tensor snap = new Tensor(_latentFixed!.Shape, DType.F32);
        backend.CopyInto(snap, _latentFixed);
        return snap;
    }

    /// <summary>Resets the backend graph slot and this transformer's signature. Call before
    /// FreeActivations / FreeWeights.</summary>
    public void InvalidateStepGraph(IBackend backend)
    {
        backend.StepGraphReset();
        if (ReferenceEquals(backend.StepGraphOwner, this))
            backend.StepGraphOwner = null;
        _graphSig = long.MinValue;
        _graphSigCalls = 0;
    }

    /// <summary>Step-graph-aware forward (see <c>FluxTransformer.ForwardGraphable</c> — identical state
    /// machine). <c>callerOwns</c> false = transformer-owned fixed velocity buffer, rewritten next step.</summary>
    public (Tensor velocity, bool callerOwns) ForwardGraphable(IBackend backend, Tensor packedLatent,
        Tensor textEmbeddings, float sigma, float guidanceScale, int hPacked, int wPacked)
    {
        int batch = (int)packedLatent.Shape[0];
        bool graphMode = StepGraphEnabled && backend.StepGraphSupported && !_graphDead
            && batch == 1 && ReferenceEquals(packedLatent, _latentFixed);
        if (!graphMode)
            return (Forward(backend, packedLatent, textEmbeddings, sigma, guidanceScale, hPacked, wPacked), true);

        int inCh = _config.TimestepChannels;
        Tensor sinHost = new Tensor(new TensorShape(1, inCh), DType.F32);
        ComputeSinusoidalTimestep(sinHost, sigma * 1000.0f, 1, inCh);
        _sinFixed ??= new Tensor(sinHost.Shape, DType.F32);
        backend.CopyInto(_sinFixed, sinHost);
        sinHost.Dispose();

        long sig = ((long)hPacked * 19349663L) ^ ((long)wPacked * 83492791L)
            ^ ((long)textEmbeddings.Shape[1] * 2654435761L)
            ^ ((long)BitConverter.SingleToInt32Bits(guidanceScale) << 13)
            ^ ((long)System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(textEmbeddings) << 17);
        bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, this);
        if (sig != _graphSig || ownerLost)
        {
            backend.StepGraphReset();
            if (!ownerLost && _graphSig != long.MinValue && ++_graphSigFlips > 8)
            {
                _graphDead = true;
                Logs.Warning("[Flux2 graph] signature flip storm — step-graph disabled for this session.");
            }
            _graphSig = sig;
            _graphSigCalls = 0;
            backend.StepGraphOwner = this;
        }
        else if (_graphSigCalls > GraphCaptureCall && !backend.StepGraphReady)
        {
            _graphSigCalls = 0;
        }
        _graphSigCalls++;
        if (_graphSigCalls == 1)
        {
            if (_config.GuidanceEmbed && _guidanceLinear1Weight != null)
            {
                Tensor gHost = new Tensor(new TensorShape(1, inCh), DType.F32);
                ComputeSinusoidalTimestep(gHost, guidanceScale * 1000.0f, 1, inCh);
                _guidanceSinFixed ??= new Tensor(gHost.Shape, DType.F32);
                backend.CopyInto(_guidanceSinFixed, gHost);
                gHost.Dispose();
            }
            // Pin step-invariant conditioning device-resident (pageable re-upload breaks capture).
            backend.PreloadWeights(new List<Tensor> { textEmbeddings });
            EnsureRope((int)textEmbeddings.Shape[1], hPacked, wPacked);
        }
        _graphVelocity ??= new Tensor(new TensorShape(1, (long)packedLatent.Shape[1], _config.OutChannels), DType.F32);

        if (backend.StepGraphReady && _graphSigCalls > GraphCaptureCall)
        {
            backend.StepGraphLaunch();
            return (_graphVelocity!, false);
        }

        bool capture = _graphSigCalls == GraphCaptureCall;
        if (capture)
        {
            // Quantized opt-in route: hand the eager warm-up passes' unused pool reservation back to the
            // driver before capture — instantiate reserves the graph pool's transient-dequant high-water ON
            // TOP of the async pool, and beside the resident Q4 weights that margin decides capture vs OOM.
            // Live allocations (fixed buffers, rope tables, preloaded weights) are untouched by the trim.
            if (_hasQuantizedWeights)
                backend.TrimMemoryPool();
            backend.StepGraphBegin();
        }
        try
        {
            RunStepIntoFixed(backend, packedLatent, textEmbeddings, hPacked, wPacked);
        }
        catch (Exception ex) when (capture)
        {
            backend.StepGraphReset();
            _graphDead = true;
            Logs.Warning($"[Flux2 graph] capture invalidated — falling back to eager: {ex}");
            RunStepIntoFixed(backend, packedLatent, textEmbeddings, hPacked, wPacked);
            return (_graphVelocity!, false);
        }
        if (capture)
        {
            try
            {
                backend.StepGraphEndAndLaunch();
                Logs.Info("[Flux2 graph] denoise step captured; replaying via cuGraphLaunch.");
            }
            catch (Exception ex)
            {
                backend.StepGraphReset();
                _graphDead = true;
                Logs.Warning($"[Flux2 graph] capture failed — falling back to eager: {ex.Message}");
                RunStepIntoFixed(backend, packedLatent, textEmbeddings, hPacked, wPacked);
            }
        }
        return (_graphVelocity!, false);
    }

    /// <summary>The captured (or capture-warming) step body: temb from the fixed device sin buffer(s), full
    /// forward, velocity lands in the fixed buffer via <c>CopyInto</c>.</summary>
    private void RunStepIntoFixed(IBackend backend, Tensor packedLatent, Tensor textEmbeddings,
        int hPacked, int wPacked)
    {
        Tensor temb = ComputeTembFromSin(backend, _sinFixed!, _guidanceSinFixed, 1);
        Tensor v = ForwardWithTemb(backend, packedLatent, textEmbeddings, temb, hPacked, wPacked);
        backend.CopyInto(_graphVelocity!, v);
        v.Dispose();
        temb.Dispose();
    }

    private Tensor ComputeTimestepEmbedding(IBackend backend, float sigma, float guidanceScale, int batch)
    {
        int inCh = _config.TimestepChannels;

        // Pipeline passes timestep/1000 as `sigma`, so scale back up to match the Flux.2
        // reference `timestep = timestep * 1000` before sinusoidal embedding.
        TensorShape sinShape = new TensorShape(batch, inCh);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        ComputeSinusoidalTimestep(sinEmbed, sigma * 1000.0f, batch, inCh);

        Tensor? guidanceSin = null;
        if (_config.GuidanceEmbed && _guidanceLinear1Weight != null)
        {
            guidanceSin = new Tensor(sinShape, DType.F32);
            ComputeSinusoidalTimestep(guidanceSin, guidanceScale * 1000.0f, batch, inCh);
        }

        Tensor temb = ComputeTembFromSin(backend, sinEmbed, guidanceSin, batch);
        sinEmbed.Dispose();
        guidanceSin?.Dispose();
        return temb;
    }

    /// <summary>temb MLP core from (already-built) sinusoidal embeddings — all device ops, so the step-graph
    /// path can feed FIXED device sin buffers and capture the whole temb compute.</summary>
    private Tensor ComputeTembFromSin(IBackend backend, Tensor sinEmbed, Tensor? guidanceSin, int batch)
    {
        int hidden = _config.HiddenSize;

        // Timestep MLP: Linear(inCh, hidden) → SiLU → Linear(hidden, hidden)
        TensorShape hidShape = new TensorShape(batch, hidden);
        Tensor t1 = new Tensor(hidShape, DType.F32);
        backend.Linear(t1, sinEmbed, _timestepLinear1Weight!, null);
        Tensor t1Act = new Tensor(hidShape, DType.F32);
        backend.Silu(t1Act, t1);
        t1.Dispose();
        Tensor temb = new Tensor(hidShape, DType.F32);
        backend.Linear(temb, t1Act, _timestepLinear2Weight!, null);
        t1Act.Dispose();

        if (guidanceSin is not null && _guidanceLinear1Weight != null)
        {
            // guidance_embed adds to temb (Dev only), same units as timestep (× 1000 by the caller).
            Tensor g1 = new Tensor(hidShape, DType.F32);
            backend.Linear(g1, guidanceSin, _guidanceLinear1Weight!, null);
            Tensor g1Act = new Tensor(hidShape, DType.F32);
            backend.Silu(g1Act, g1);
            g1.Dispose();
            Tensor gEmb = new Tensor(hidShape, DType.F32);
            backend.Linear(gEmb, g1Act, _guidanceLinear2Weight!, null);
            g1Act.Dispose();

            Tensor tembNew = new Tensor(hidShape, DType.F32);
            backend.Add(tembNew, temb, gEmb);
            temb.Dispose();
            gEmb.Dispose();
            temb = tembNew;
        }

        return temb;
    }

    /// <summary>Sinusoidal timestep embedding with flip_sin_to_cos=True, downscale_freq_shift=0.</summary>
    private static void ComputeSinusoidalTimestep(Tensor output, float timestep, int batch, int inCh)
    {
        float* outPtr = (float*)output.DataPointer;
        int halfDim = inCh / 2;
        // diffusers Timesteps uses max_period=10000 by default for the sinusoidal table
        float maxPeriod = 10000.0f;

        for (int b = 0; b < batch; b++)
        {
            int baseOffset = b * inCh;
            for (int i = 0; i < halfDim; i++)
            {
                float freq = MathF.Exp(-MathF.Log(maxPeriod) * i / halfDim);
                float angle = timestep * freq;
                outPtr[baseOffset + i] = MathF.Cos(angle);
                outPtr[baseOffset + halfDim + i] = MathF.Sin(angle);
            }
        }
    }

    /// <summary>Final layer: AdaLayerNormContinuous (SiLU(temb) → Linear → split [shift, scale]) → LayerNorm(no affine) → modulate <c>(1+scale)*x + shift</c> → proj_out. The converter applies BFL→diffusers half-swap on <c>norm_out.linear</c> so the layout here is <c>[scale, shift]</c>.</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor hidden, Tensor temb, int batch, int seqLen)
    {
        int dim = _config.HiddenSize;
        int outDim = _config.OutChannels;

        TensorShape tembShape = new TensorShape(batch, dim);
        Tensor activated = new Tensor(tembShape, DType.F32);
        backend.Silu(activated, temb);

        TensorShape modShape = new TensorShape(batch, dim * 2);
        Tensor modParams = new Tensor(modShape, DType.F32);
        backend.Linear(modParams, activated, _normOutLinearWeight!, null);
        activated.Dispose();

        TensorShape seqShape = new TensorShape(batch, seqLen, dim);
        Tensor normed = new Tensor(seqShape, DType.F32);
        Tensor modulated;
        if (batch == 1)
        {
            // Device AdaLN-Continuous (the Chroma ApplyContinuousNormDevice idiom): the old host loop read the
            // device-produced modParams via DataPointer — a full-pipeline drain every forward. Flux.2 layout is
            // [scale, shift] (converter half-swap), each a contiguous dim-length row of the flat projection.
            backend.LayerNormNoAffine(normed, hidden, _config.LayerNormEps);
            Tensor scaleRow = new Tensor(new TensorShape(1, dim), DType.F32);
            backend.SliceRows(scaleRow, modParams, 0);
            Tensor shiftRow = new Tensor(new TensorShape(1, dim), DType.F32);
            backend.SliceRows(shiftRow, modParams, 1);
            Tensor scalePlus1 = new Tensor(new TensorShape(1, dim), DType.F32);
            backend.AddScalar(scalePlus1, scaleRow, 1.0f);
            scaleRow.Dispose();
            modulated = new Tensor(seqShape, DType.F32);
            backend.AffineBroadcastLastDim(modulated, normed, scalePlus1, shiftRow);
            scalePlus1.Dispose();
            shiftRow.Dispose();
        }
        else
        {
            LayerNormNoAffine(normed, hidden, batch, seqLen, dim, _config.LayerNormEps);
            modulated = new Tensor(seqShape, DType.F32);
            float* normPtr = (float*)normed.DataPointer;
            float* modPtr = (float*)modParams.DataPointer;
            float* outModPtr = (float*)modulated.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                int modBase = b * dim * 2;
                for (int s = 0; s < seqLen; s++)
                {
                    int vecOffset = (b * seqLen + s) * dim;
                    for (int d = 0; d < dim; d++)
                    {
                        float scale = modPtr[modBase + d];
                        float shift = modPtr[modBase + dim + d];
                        outModPtr[vecOffset + d] = normPtr[vecOffset + d] * (1.0f + scale) + shift;
                    }
                }
            }
        }
        normed.Dispose();
        modParams.Dispose();

        TensorShape projShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected = new Tensor(projShape, DType.F32);
        backend.Linear(projected, modulated, _projOutWeight!, null);
        modulated.Dispose();
        return projected;
    }

    private static void LayerNormNoAffine(Tensor output, Tensor input, int batch, int seqLen, int dim, float eps)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int offset = (b * seqLen + s) * dim;
                float mean = 0f;
                for (int d = 0; d < dim; d++) mean += inPtr[offset + d];
                mean /= dim;
                float variance = 0f;
                for (int d = 0; d < dim; d++) { float diff = inPtr[offset + d] - mean; variance += diff * diff; }
                variance /= dim;
                float invStd = 1.0f / MathF.Sqrt(variance + eps);
                for (int d = 0; d < dim; d++) outPtr[offset + d] = (inPtr[offset + d] - mean) * invStd;
            }
        }
    }

    private static void ExtractImageTokens(Tensor output, Tensor input, int batch, int txtSeqLen, int imgSeqLen, int dim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int totalSeqLen = txtSeqLen + imgSeqLen;
        for (int b = 0; b < batch; b++)
        {
            long imgBytes = (long)imgSeqLen * dim * sizeof(float);
            Buffer.MemoryCopy(inPtr + b * totalSeqLen * dim + txtSeqLen * dim, outPtr + b * imgSeqLen * dim, imgBytes, imgBytes);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _xEmbedWeight = null;
            _contextEmbedWeight = null;
            _timestepLinear1Weight = null;
            _timestepLinear2Weight = null;
            _guidanceLinear1Weight = null;
            _guidanceLinear2Weight = null;
            _normOutLinearWeight = null;
            _projOutWeight = null;
        }
        if (_disposed == 1)
        {
            _latentFixed?.Dispose();
            _latentFixed = null;
            _sinFixed?.Dispose();
            _sinFixed = null;
            _guidanceSinFixed?.Dispose();
            _guidanceSinFixed = null;
            _graphVelocity?.Dispose();
            _graphVelocity = null;
        }
        GC.SuppressFinalize(this);
    }
}
