using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>HiDream-I1 image transformer (HiDreamImageTransformer2DModel) — text-to-image only. <para>Architecture: 16 double-stream blocks (joint MM-attention) + 32 single-stream blocks (parallel image+text attention), MoE SwiGLU FFN with a shared expert + top-2-of-4 routed experts on the image side (see <see cref="HiDreamBlock"/> for the gate softmax / top-k renorm), 3-axis RoPE on (layer-id, row, col).</para> <para>Text conditioning is the concatenation of: <list type="bullet"> <item>Each Llama-3.1 hidden state (one per block) projected through its OWN <c>caption_projection[i]</c> (sequential, i = block index);</item> <item>The T5-XXL hidden state projected through <c>caption_projection[-1]</c> (the final, 49th entry);</item> <item>The "initial encoder hidden states" = concat(T5 projection, last Llama projection); each block appends its own Llama projection to the running encoder.</item> </list> The pooled CLIP-L+CLIP-G embedding is fed into the timestep MLP via the pooled embedder.</para></summary>
public sealed unsafe class HiDreamTransformer : IDisposable
{
    private readonly HiDreamConfig _config;
    private readonly HiDreamBlock[] _doubleBlocks;
    private readonly HiDreamBlock[] _singleBlocks;
    private readonly HiDreamRope _rope;

    // Rope-table signature: Precompute rebuilds a ~300k-trig host table AND re-uploads the GPU cos/sin
    // tables — it ran twice per forward (identical args, dual- then single-stream) = ~100 rebuilds/gen.
    private long _ropeSig = long.MinValue;

    /// <summary>Precomputes the rope tables only when the (grid, text-length) signature changes.</summary>
    private void EnsureRope(int imgSeqLen, int patH, int patW, int txtLen)
    {
        long sig = ((long)imgSeqLen << 40) ^ ((long)patH << 28) ^ ((long)patW << 16) ^ (long)txtLen;
        if (sig == _ropeSig)
            return;
        Tensor posIds = HiDreamRope.BuildPositionIds(imgSeqLen, patH, patW, txtLen);
        _rope.Precompute(posIds);
        posIds.Dispose();
        _ropeSig = sig;
    }
    private int _disposed;

    // Diagnostic (HARTSY_HIDREAM_PROBE=1): per-block absmax/rms of the residual streams — forces a host
    // sync per probe, first forward only. Used to decide F16-activation viability (needs absmax << 65k).
    private static readonly bool StreamProbe = Environment.GetEnvironmentVariable("HARTSY_HIDREAM_PROBE") == "1";
    private static bool _probedOnce;

    /// <summary>First-forward-only absmax/rms probe of a stream tensor (host-sync; diagnostic only).</summary>
    private static void Probe(string label, Tensor t)
    {
        if (!StreamProbe || _probedOnce) return;
        float* p = (float*)t.DataPointer;
        float mx = 0f;
        double ss = 0;
        long n = t.ElementCount;
        for (long e = 0; e < n; e++) { float a = MathF.Abs(p[e]); if (a > mx) mx = a; ss += (double)p[e] * p[e]; }
        HartsyInference.Core.Logging.Logs.Info($"[HiDream probe] {label}: absmax={mx:F4} rms={Math.Sqrt(ss / n):F5}");
    }

    // x_embedder: Linear(in_channels * patch_size^2, inner_dim)
    private Tensor? _xEmbedWeight, _xEmbedBias;

    // t_embedder.timestep_embedder: Linear → SiLU → Linear (frequency_size=256 → inner_dim → inner_dim)
    private Tensor? _tEmbedLinear1Weight, _tEmbedLinear1Bias;
    private Tensor? _tEmbedLinear2Weight, _tEmbedLinear2Bias;

    // p_embedder.pooled_embedder: Linear → SiLU → Linear (text_emb_dim=2048 → inner_dim → inner_dim)
    private Tensor? _pEmbedLinear1Weight, _pEmbedLinear1Bias;
    private Tensor? _pEmbedLinear2Weight, _pEmbedLinear2Bias;

    // caption_projection[i].linear : Linear(caption_channels[i], inner_dim, bias=False)
    private Tensor[]? _captionProjectionWeights;

    // final_layer: AdaLN (SiLU + Linear(dim, 2*dim)) + Linear(dim, p^2 * out_channels)
    private Tensor? _finalAdaLnWeight, _finalAdaLnBias;
    private Tensor? _finalProjWeight, _finalProjBias;

    /// <summary>Creates a HiDream transformer from a config.</summary>
    public HiDreamTransformer(HiDreamConfig config)
    {
        _config = config;
        int hidden = config.InnerDim;
        int numHeads = config.NumAttentionHeads;
        int headDim = config.AttentionHeadDim;
        int ffDim = 4 * hidden; // diffusers reference: hidden_dim = 4 * dim

        _doubleBlocks = new HiDreamBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _doubleBlocks[i] = new HiDreamBlock(hidden, numHeads, headDim, ffDim,
                isSingle: false, numRoutedExperts: config.NumRoutedExperts,
                numActivatedExperts: config.NumActivatedExperts, qkNormEps: config.RmsNormEps);

        _singleBlocks = new HiDreamBlock[config.NumSingleLayers];
        for (int i = 0; i < config.NumSingleLayers; i++)
            _singleBlocks[i] = new HiDreamBlock(hidden, numHeads, headDim, ffDim,
                isSingle: true, numRoutedExperts: config.NumRoutedExperts,
                numActivatedExperts: config.NumActivatedExperts, qkNormEps: config.RmsNormEps);

        _rope = new HiDreamRope(config.AxesDimsRope, (int)config.RopeTheta);
    }

    /// <summary>Loads all transformer weights from the converted (diffusers-style) state dict.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // x_embedder: the diffusers reference uses a single Linear with name "x_embedder.proj".
        _xEmbedWeight = weights["x_embedder.proj.weight"];
        _xEmbedBias = weights["x_embedder.proj.bias"];

        // t_embedder.timestep_embedder.linear_{1,2}
        _tEmbedLinear1Weight = weights["t_embedder.timestep_embedder.linear_1.weight"];
        _tEmbedLinear1Bias = weights["t_embedder.timestep_embedder.linear_1.bias"];
        _tEmbedLinear2Weight = weights["t_embedder.timestep_embedder.linear_2.weight"];
        _tEmbedLinear2Bias = weights["t_embedder.timestep_embedder.linear_2.bias"];

        // p_embedder.pooled_embedder.linear_{1,2}
        _pEmbedLinear1Weight = weights["p_embedder.pooled_embedder.linear_1.weight"];
        _pEmbedLinear1Bias = weights["p_embedder.pooled_embedder.linear_1.bias"];
        _pEmbedLinear2Weight = weights["p_embedder.pooled_embedder.linear_2.weight"];
        _pEmbedLinear2Bias = weights["p_embedder.pooled_embedder.linear_2.bias"];

        // caption_projection[i].linear.weight (no bias). Diffusers builds ONE projection per block plus a final
        // one for T5: caption_channels = [llama_dim]*(num_layers+num_single_layers) + [t5_dim], i.e.
        // num_layers + num_single_layers + 1 entries (49 for V1). Llama layer i is projected through
        // caption_projection[i] (per-block, sequential); T5 through caption_projection[-1]. Loading only
        // CaptionChannels.Length (=2) collapsed every block onto caption_projection[0] and mis-mapped T5 to [1],
        // corrupting all conditioning (the brown-cloud bug).
        int numCaptionProjections = _config.NumLayers + _config.NumSingleLayers + 1;
        _captionProjectionWeights = new Tensor[numCaptionProjections];
        for (int i = 0; i < numCaptionProjections; i++)
            _captionProjectionWeights[i] = weights[$"caption_projection.{i}.linear.weight"];

        // Per-block weights
        for (int i = 0; i < _config.NumLayers; i++)
            _doubleBlocks[i].LoadWeights(weights, $"double_stream_blocks.{i}.block");
        for (int i = 0; i < _config.NumSingleLayers; i++)
            _singleBlocks[i].LoadWeights(weights, $"single_stream_blocks.{i}.block");

        // Final layer
        _finalAdaLnWeight = weights["final_layer.adaLN_modulation.1.weight"];
        _finalAdaLnBias = weights["final_layer.adaLN_modulation.1.bias"];
        _finalProjWeight = weights["final_layer.linear.weight"];
        _finalProjBias = weights["final_layer.linear.bias"];
    }

    /// <summary>Yields every weight tensor for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in EnumerateSharedWeights()) yield return w;
        for (int i = 0; i < _doubleBlocks.Length; i++)
            foreach (Tensor w in _doubleBlocks[i].EnumerateWeights()) yield return w;
        for (int i = 0; i < _singleBlocks.Length; i++)
            foreach (Tensor w in _singleBlocks[i].EnumerateWeights()) yield return w;
    }

    /// <summary>The number of streamable blocks, flat-indexed double-then-single: <c>[0, NumLayers)</c> are double-stream, <c>[NumLayers, NumLayers+NumSingleLayers)</c> are single-stream. See <see cref="ForwardSharded"/> for how a block-range split crosses this boundary.</summary>
    public int BlockCount => _doubleBlocks.Length + _singleBlocks.Length;

    /// <summary>The non-block weights: x_embedder, t_embedder, p_embedder, ALL <c>caption_projection[]</c> entries (every Llama layer + T5 is projected once, up front, before the block loop), and the final AdaLN + linear. Always resident on the block-range split's FIRST backend — see <see cref="ForwardSharded"/>.</summary>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        if (_xEmbedWeight is not null) yield return _xEmbedWeight;
        if (_xEmbedBias is not null) yield return _xEmbedBias;

        if (_tEmbedLinear1Weight is not null) yield return _tEmbedLinear1Weight;
        if (_tEmbedLinear1Bias is not null) yield return _tEmbedLinear1Bias;
        if (_tEmbedLinear2Weight is not null) yield return _tEmbedLinear2Weight;
        if (_tEmbedLinear2Bias is not null) yield return _tEmbedLinear2Bias;

        if (_pEmbedLinear1Weight is not null) yield return _pEmbedLinear1Weight;
        if (_pEmbedLinear1Bias is not null) yield return _pEmbedLinear1Bias;
        if (_pEmbedLinear2Weight is not null) yield return _pEmbedLinear2Weight;
        if (_pEmbedLinear2Bias is not null) yield return _pEmbedLinear2Bias;

        if (_captionProjectionWeights is not null)
            for (int i = 0; i < _captionProjectionWeights.Length; i++)
                yield return _captionProjectionWeights[i];

        if (_finalAdaLnWeight is not null) yield return _finalAdaLnWeight;
        if (_finalAdaLnBias is not null) yield return _finalAdaLnBias;
        if (_finalProjWeight is not null) yield return _finalProjWeight;
        if (_finalProjBias is not null) yield return _finalProjBias;
    }

    /// <summary>Enumerates only blocks <c>[startBlock, endBlock)</c>'s weights (flat double-then-single index, including every routed MoE expert) — the asymmetric preload a block-range shard needs (see <see cref="ForwardSharded"/>).</summary>
    public IEnumerable<Tensor> EnumerateBlockRangeWeights(int startBlock, int endBlock)
    {
        for (int i = startBlock; i < Math.Min(endBlock, _doubleBlocks.Length); i++)
            foreach (Tensor w in _doubleBlocks[i].EnumerateWeights()) yield return w;
        for (int i = Math.Max(startBlock, _doubleBlocks.Length); i < endBlock; i++)
            foreach (Tensor w in _singleBlocks[i - _doubleBlocks.Length].EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass for one denoising step. Predicts velocity in patchified-and-projected latent space and returns it in the same [B, C_out, H, W] layout as the input latent.</summary>
    /// <param name="latent">Noisy latent [B, in_channels, H, W].</param>
    /// <param name="timestep">Scalar timestep value (sigma * 1000 in flow matching).</param>
    /// <param name="t5Hidden">T5-XXL hidden states already projected to [B, S_t5, inner_dim] by the pipeline (caption_projection is run inside this method, so pass the raw T5 hidden of shape [B, S_t5, 4096]).</param>
    /// <param name="llamaHiddenLayers">List of Llama hidden states, one per <see cref="HiDreamConfig.LlamaLayers"/> entry. Each tensor is shape [B, S_l, caption_channels[0]] (= 4096). Length must equal NumLayers + NumSingleLayers.</param>
    /// <param name="pooledEmbeds">[B, text_emb_dim=2048] pooled CLIP-L+CLIP-G embedding.</param>
    /// <param name="stepCache">Optional across-step First-Block cache (docs/Research/STEP_ACCELERATION.md §2; same pattern as <see cref="Sd3Transformer.Forward"/>). HiDream is double-stream-then-single-stream, not SD3's uniform dual-stream loop, but the image token tensor stays the same shape (<c>[B, imgSeqLen, InnerDim]</c>) from the first double block's output through to the final pre-projection <c>imgFinal</c> — so the same "cache block 0's image output, skip straight to the final layer on a hit" trick applies across the double→single transition unchanged. On a hit, the entire rest of both block stacks is skipped, including the per-block Llama-conditioning concat every remaining block would have consumed (those projections are computed once up front regardless — this doesn't affect the caption-projection cost).</param>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep,
        Tensor t5Hidden, IReadOnlyList<Tensor> llamaHiddenLayers, Tensor pooledEmbeds, DeviceFeatureCache? stepCache = null)
    {
        ThrowIfDisposed();
        int batch = (int)latent.Shape[0];
        int inChannels = (int)latent.Shape[1];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];
        int patH = height / _config.PatchSize;
        int patW = width / _config.PatchSize;
        int imgSeqLen = patH * patW;
        int hidden = _config.InnerDim;

        if (llamaHiddenLayers.Count != _doubleBlocks.Length + _singleBlocks.Length)
            throw new InvalidOperationException(
                $"Expected {_doubleBlocks.Length + _singleBlocks.Length} Llama hidden layers (one per block), got {llamaHiddenLayers.Count}.");

        // ── 1. Patchify and embed ──
        Tensor patched = PatchifyLatent(latent, batch, inChannels, patH, patW, _config.PatchSize);
        TensorShape embedShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgTokens = new Tensor(embedShape, DType.F32);
        backend.Linear(imgTokens, patched, _xEmbedWeight!, _xEmbedBias);
        patched.Dispose();
        HiDreamDebugDump.Dump("x_embed", imgTokens);
        Probe("x_embed", imgTokens);

        // ── 2. Build temb = timestep_embed + pooled_embed ──
        Tensor temb = ComputeTimestepAndPooledEmbedding(backend, timestep, pooledEmbeds, batch);
        HiDreamDebugDump.Dump("temb", temb);
        Probe("temb", temb);

        // ── 3. Project caption inputs ──
        // For the per-block llama states we use caption_projection[0]; for T5 we use caption_projection[-1].
        // Each Llama layer i is projected through its OWN caption_projection[i] (per-block, sequential —
        // matching diffusers `caption_projection[i](encoder_hidden_states[i])`), NOT a shared [0].
        Tensor[] llamaProj = new Tensor[llamaHiddenLayers.Count];
        for (int i = 0; i < llamaHiddenLayers.Count; i++)
        {
            llamaProj[i] = ProjectCaption(backend, llamaHiddenLayers[i], _captionProjectionWeights![i]);
        }

        Tensor t5Proj = ProjectCaption(backend, t5Hidden, _captionProjectionWeights![^1]);
        HiDreamDebugDump.Dump("t5_proj", t5Proj);

        // ── 4. Initial encoder hidden states: concat(llama[-1], llama[-2]) ──
        // We treat the "last two" entries as the seed initial encoder, matching the diffusers
        // reference: initial_encoder_hidden_states = cat([encoder_hidden_states[-1],
        // encoder_hidden_states[-2]], dim=1) where encoder_hidden_states is the post-projection list
        // with T5 appended at the end. The "[-1]" is T5; "[-2]" is the last selected Llama layer.
        Tensor lastLlama = llamaProj[^1];
        Tensor initialEncoder = ConcatSeq(backend, t5Proj, lastLlama);
        int initialEncoderSeqLen = (int)initialEncoder.Shape[1];
        HiDreamDebugDump.Dump("initial_encoder", initialEncoder);
        Probe("initial_encoder", initialEncoder);

        // ── 5. Precompute RoPE for the (image + initial_encoder + per-block-llama) sequence ──
        // We use the longest text sequence the rope table needs — the maximum of (initial_encoder_seq_len + any
        // single llama_proj_seq_len) since per-block we concat one Llama layer to the initial encoder.
        // All per-block llama layers share the same seq len (they came from the same encoder forward),
        // so we precompute once with imgSeqLen + (initialEncoderSeqLen + llama_seq_len).
        int perLayerLlamaSeqLen = (int)llamaProj[0].Shape[1];
        int doubleStreamTotalSeqLen = imgSeqLen + initialEncoderSeqLen + perLayerLlamaSeqLen;
        EnsureRope(imgSeqLen, patH, patW, initialEncoderSeqLen + perLayerLlamaSeqLen);

        // ── 6. Double-stream blocks ──
        Tensor curImg = imgTokens;
        Tensor curEncoder = initialEncoder;
        int blockId = 0;
        Tensor? cacheAnchor = null;
        Tensor? hitReconstruction = null;

        for (int b = 0; b < _doubleBlocks.Length; b++)
        {
            // The encoder fed to this block is initial_encoder (sliced to its seed len every loop —
            // the block's text output may grow because we concatenated a fresh llama layer beneath
            // it; we re-slice after the block). The per-block llama is appended fresh.
            Tensor curLlama = llamaProj[blockId];
            Tensor blockEncoderIn = ConcatSeq(backend, curEncoder, curLlama);

            (Tensor newImg, Tensor newEncoderFull) = _doubleBlocks[b].ForwardDouble(
                backend, curImg, blockEncoderIn, temb, _rope, doubleStreamTotalSeqLen);

            blockEncoderIn.Dispose();

            // Re-slice the encoder back to its initial-encoder length so the next block sees a
            // matching shape (the [llama] tail was scratch).
            Tensor reslicedEncoder = SliceSeq(backend, newEncoderFull, initialEncoderSeqLen);
            newEncoderFull.Dispose();

            if (!ReferenceEquals(curImg, imgTokens) && !ReferenceEquals(curImg, cacheAnchor)) curImg.Dispose();
            if (!ReferenceEquals(curEncoder, initialEncoder)) curEncoder.Dispose();
            curImg = newImg;
            curEncoder = reslicedEncoder;
            HiDreamDebugDump.Dump($"double_block_{b}_img", curImg);
            Probe($"double_{b}_img", curImg);
            Probe($"double_{b}_enc", curEncoder);

            blockId++;

            // Gate right after the very first block in the whole double+single stack. On a hit, skip
            // straight past every remaining double block, the double→single transition, AND every single
            // block — the image token tensor is the same shape ([B, imgSeqLen, InnerDim]) at every one of
            // those points since InnerDim is constant across both stacks, so the cached residual reconstructs
            // the final pre-projection state directly. Real-weight confirmed (HiDreamStepCacheRealWeightTests.cs,
            // 2026-08-10): the anchor/reconstruction shape match holds across the double→single transition —
            // real reuses fired, no StoreResidual/ApplyResidual shape exception, output visually matched
            // cache-off. HiDream also needed the late-window gate engaged (threshold alone collapsed the image —
            // see that test's class doc for the finding), unlike SD3/Chroma/Flux which didn't need it.
            if (b == 0 && stepCache is not null)
            {
                if (stepCache.ShouldCompute(backend, curImg))
                {
                    cacheAnchor = curImg;
                }
                else
                {
                    hitReconstruction = stepCache.ApplyResidual(backend, curImg);
                    curImg.Dispose();
                    curEncoder.Dispose();
                    break;
                }
            }
        }
        bool cacheHit = hitReconstruction is not null;
        // curEncoder was already reassigned away from initialEncoder in the loop's first iteration (which
        // always runs before the cache gate can fire), on both the hit and miss path — safe unconditionally.
        if (!ReferenceEquals(curEncoder, initialEncoder)) initialEncoder.Dispose();
        // The first double block replaced curImg; the x_embed output itself was never freed (a ~40 MB/forward
        // VRAM leak at 1024² that compounded across the CFG denoise loop).
        if (!ReferenceEquals(curImg, imgTokens)) imgTokens.Dispose();

        Tensor imgFinal;
        if (cacheHit)
        {
            imgFinal = hitReconstruction!;
            // Single-stream phase never ran — nothing else references these.
            for (int i = 0; i < llamaProj.Length; i++) llamaProj[i].Dispose();
            t5Proj.Dispose();
        }
        else
        {
            // ── 7. Switch to single-stream: concat current image with the running encoder, then per-block llama ──
            Tensor jointBeforeSingle = ConcatSeq(backend, curImg, curEncoder);
            curImg.Dispose();
            curEncoder.Dispose();

            int singleStreamBaseSeqLen = imgSeqLen + initialEncoderSeqLen;
            int singleStreamTotalSeqLen = singleStreamBaseSeqLen + perLayerLlamaSeqLen;

            // Same rope table as the double-stream phase (identical args) — EnsureRope no-ops on the sig hit.
            EnsureRope(imgSeqLen, patH, patW, initialEncoderSeqLen + perLayerLlamaSeqLen);

            Tensor curJoint = jointBeforeSingle;
            for (int b = 0; b < _singleBlocks.Length; b++)
            {
                Tensor curLlama = llamaProj[blockId];
                Tensor blockIn = ConcatSeq(backend, curJoint, curLlama);

                Tensor newJointFull = _singleBlocks[b].ForwardSingle(
                    backend, blockIn, temb, _rope, imgSeqLen, singleStreamTotalSeqLen);
                blockIn.Dispose();

                // Slice off the appended llama tail.
                Tensor resliced = SliceSeq(backend, newJointFull, singleStreamBaseSeqLen);
                newJointFull.Dispose();

                curJoint.Dispose();
                curJoint = resliced;
                HiDreamDebugDump.Dump($"single_block_{b}", curJoint);
                Probe($"single_{b}", curJoint);
                blockId++;
            }

            // ── 8. Extract image tokens (first imgSeqLen of curJoint) ──
            imgFinal = SliceSeq(backend, curJoint, imgSeqLen);
            curJoint.Dispose();

            for (int i = 0; i < llamaProj.Length; i++) llamaProj[i].Dispose();
            t5Proj.Dispose();

            if (cacheAnchor is not null)
            {
                stepCache!.StoreResidual(backend, cacheAnchor, imgFinal);
                cacheAnchor.Dispose();
            }
        }

        // ── 9. Final layer ──
        Tensor projected = ApplyFinalLayer(backend, imgFinal, temb, batch, imgSeqLen);
        imgFinal.Dispose();
        temb.Dispose();
        HiDreamDebugDump.Dump("proj_out", projected);
        Probe("proj_out", projected);
        if (StreamProbe) _probedOnce = true;

        // ── 10. Unpatchify [B, S, p²*C] → [B, C, H, W] ──
        Tensor output = UnpatchifyLatent(projected, batch, _config.OutChannels, patH, patW, _config.PatchSize);
        projected.Dispose();
        HiDreamDebugDump.DumpOutput(output);
        return output;
    }

    /// <summary>Predicts velocity for one denoising step with the flat double-then-single block loop split across two backends: blocks <c>[0, splitBlock)</c> run on <paramref name="backendA"/>, blocks <c>[splitBlock, BlockCount)</c> on <paramref name="backendB"/> — a sequential pipeline split, not tensor-parallel, so there is <b>no latency win</b>; the win is VRAM pooling. <paramref name="backendA"/> must hold <see cref="EnumerateSharedWeights"/> + <see cref="EnumerateBlockRangeWeights"/><c>(0, splitBlock)</c>; <paramref name="backendB"/> must hold ONLY <see cref="EnumerateBlockRangeWeights"/><c>(splitBlock, BlockCount)</c>. <para>The double→single transition (concat image+encoder into one joint stream) happens exactly once, on whichever backend's range first crosses block index <see cref="HiDreamConfig.NumLayers"/> — see <see cref="ForwardBlocksRange"/>. The per-block Llama conditioning (<paramref name="llamaHiddenLayers"/>, one entry per block) is projected once up front (backendA, shared weights) and the entries B's range needs are peer-copied alongside the activation hand-off — they are activations, not weights, so they are NOT covered by <see cref="EnumerateBlockRangeWeights"/>.</para></summary>
    public Tensor ForwardSharded(IBackend backendA, IBackend backendB, Tensor latent, float timestep,
        Tensor t5Hidden, IReadOnlyList<Tensor> llamaHiddenLayers, Tensor pooledEmbeds, int splitBlock)
    {
        ThrowIfDisposed();
        int totalBlocks = BlockCount;
        if (splitBlock <= 0 || splitBlock >= totalBlocks)
            throw new ArgumentOutOfRangeException(nameof(splitBlock),
                $"splitBlock must be in (0, {totalBlocks}) — 0 or {totalBlocks} would put the whole transformer on one backend.");

        int batch = (int)latent.Shape[0];
        int inChannels = (int)latent.Shape[1];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];
        int patH = height / _config.PatchSize;
        int patW = width / _config.PatchSize;
        int imgSeqLen = patH * patW;
        int hidden = _config.InnerDim;

        if (llamaHiddenLayers.Count != totalBlocks)
            throw new InvalidOperationException(
                $"Expected {totalBlocks} Llama hidden layers (one per block), got {llamaHiddenLayers.Count}.");

        // ── prefix (shared-weight stage, always backendA): patchify+embed, temb, caption projections, rope ──
        Tensor patched = PatchifyLatent(latent, batch, inChannels, patH, patW, _config.PatchSize);
        Tensor imgTokens = new Tensor(new TensorShape(batch, imgSeqLen, hidden), DType.F32);
        backendA.Linear(imgTokens, patched, _xEmbedWeight!, _xEmbedBias);
        patched.Dispose();

        Tensor temb = ComputeTimestepAndPooledEmbedding(backendA, timestep, pooledEmbeds, batch);

        Tensor[] llamaProj = new Tensor[llamaHiddenLayers.Count];
        for (int i = 0; i < llamaHiddenLayers.Count; i++)
            llamaProj[i] = ProjectCaption(backendA, llamaHiddenLayers[i], _captionProjectionWeights![i]);
        Tensor t5Proj = ProjectCaption(backendA, t5Hidden, _captionProjectionWeights![^1]);

        Tensor lastLlama = llamaProj[^1];
        Tensor initialEncoder = ConcatSeq(backendA, t5Proj, lastLlama);
        int initialEncoderSeqLen = (int)initialEncoder.Shape[1];
        t5Proj.Dispose();

        int perLayerLlamaSeqLen = (int)llamaProj[0].Shape[1];
        int doubleStreamTotalSeqLen = imgSeqLen + initialEncoderSeqLen + perLayerLlamaSeqLen;
        int singleStreamBaseSeqLen = imgSeqLen + initialEncoderSeqLen;
        int singleStreamTotalSeqLen = singleStreamBaseSeqLen + perLayerLlamaSeqLen;
        EnsureRope(imgSeqLen, patH, patW, initialEncoderSeqLen + perLayerLlamaSeqLen);

        // ── backend A's block range ──
        (Tensor imgA, Tensor? encA) = ForwardBlocksRange(backendA, imgTokens, initialEncoder, temb, llamaProj,
            initialEncoderSeqLen, imgSeqLen, doubleStreamTotalSeqLen, singleStreamBaseSeqLen, singleStreamTotalSeqLen,
            0, splitBlock);
        imgTokens.Dispose();
        initialEncoder.Dispose();

        // ── hand off to backendB: img (+encoder if A's range never crossed into single-stream), temb, and the
        // REMAINING per-block Llama projections B's range reads (activations, not weights) ──
        Tensor imgB = new Tensor(imgA.Shape, imgA.DType);
        backendB.CopyFromPeer(imgB, imgA, backendA);
        imgA.Dispose();
        Tensor? encB = null;
        if (encA is not null)
        {
            encB = new Tensor(encA.Shape, encA.DType);
            backendB.CopyFromPeer(encB, encA, backendA);
            encA.Dispose();
        }
        Tensor tembB = new Tensor(temb.Shape, temb.DType);
        backendB.CopyFromPeer(tembB, temb, backendA);
        Tensor[] llamaProjB = new Tensor[llamaProj.Length];
        for (int i = splitBlock; i < llamaProj.Length; i++)
        {
            llamaProjB[i] = new Tensor(llamaProj[i].Shape, llamaProj[i].DType);
            backendB.CopyFromPeer(llamaProjB[i], llamaProj[i], backendA);
        }

        (Tensor imgFinalB, Tensor? encFinalB) = ForwardBlocksRange(backendB, imgB, encB, tembB, llamaProjB,
            initialEncoderSeqLen, imgSeqLen, doubleStreamTotalSeqLen, singleStreamBaseSeqLen, singleStreamTotalSeqLen,
            splitBlock, totalBlocks);
        imgB.Dispose();
        encB?.Dispose();
        tembB.Dispose();
        for (int i = splitBlock; i < llamaProjB.Length; i++) llamaProjB[i].Dispose();
        // B's range always ends at totalBlocks (past the last single-stream block), so it always crosses the
        // double→single transition — encFinalB is always null. Defensive: idempotent Dispose either way.
        encFinalB?.Dispose();

        for (int i = 0; i < llamaProj.Length; i++) llamaProj[i].Dispose();

        // ── hand back to backendA: caption_projection / final-layer weights live there ──
        Tensor jointFinal = new Tensor(imgFinalB.Shape, imgFinalB.DType);
        backendA.CopyFromPeer(jointFinal, imgFinalB, backendB);
        imgFinalB.Dispose();

        Tensor imgFinal = SliceSeq(backendA, jointFinal, imgSeqLen);
        jointFinal.Dispose();

        Tensor projected = ApplyFinalLayer(backendA, imgFinal, temb, batch, imgSeqLen);
        imgFinal.Dispose();
        temb.Dispose();

        Tensor output = UnpatchifyLatent(projected, batch, _config.OutChannels, patH, patW, _config.PatchSize);
        projected.Dispose();
        return output;
    }

    /// <summary>Runs blocks <c>[startBlock, endBlock)</c> in order — the seam a block-range shard splits on (see <see cref="ForwardSharded"/>). State is either a <c>(img, encoder)</c> pair (still double-stream) or a single joint tensor (<paramref name="encoder"/> null, already past the transition) — mirrors <see cref="Forward"/>'s own double-then-single sequencing exactly, just windowed and with the double→single concat happening inline the moment the range crosses block index <see cref="HiDreamConfig.NumLayers"/> (never twice: a range that ends exactly at that boundary returns the pair form un-transitioned, and the NEXT range does it when IT crosses).</summary>
    private (Tensor img, Tensor? encoder) ForwardBlocksRange(IBackend backend, Tensor img, Tensor? encoder,
        Tensor temb, Tensor[] llamaProj, int initialEncoderSeqLen, int imgSeqLen, int doubleStreamTotalSeqLen,
        int singleStreamBaseSeqLen, int singleStreamTotalSeqLen, int startBlock, int endBlock)
    {
        Tensor curImg = img;
        Tensor? curEncoder = encoder;
        int i = startBlock;

        for (; i < Math.Min(endBlock, _doubleBlocks.Length); i++)
        {
            Tensor curLlama = llamaProj[i];
            Tensor blockEncoderIn = ConcatSeq(backend, curEncoder!, curLlama);
            (Tensor newImg, Tensor newEncoderFull) = _doubleBlocks[i].ForwardDouble(
                backend, curImg, blockEncoderIn, temb, _rope, doubleStreamTotalSeqLen);
            blockEncoderIn.Dispose();
            Tensor reslicedEncoder = SliceSeq(backend, newEncoderFull, initialEncoderSeqLen);
            newEncoderFull.Dispose();
            curImg.Dispose();
            curEncoder!.Dispose();
            curImg = newImg;
            curEncoder = reslicedEncoder;
        }

        if (i == _doubleBlocks.Length && curEncoder is not null && endBlock > _doubleBlocks.Length)
        {
            Tensor joint = ConcatSeq(backend, curImg, curEncoder);
            curImg.Dispose();
            curEncoder.Dispose();
            curImg = joint;
            curEncoder = null;
        }

        for (; i < endBlock; i++)
        {
            int singleIdx = i - _doubleBlocks.Length;
            Tensor curLlama = llamaProj[i];
            Tensor blockIn = ConcatSeq(backend, curImg, curLlama);
            Tensor newJointFull = _singleBlocks[singleIdx].ForwardSingle(
                backend, blockIn, temb, _rope, imgSeqLen, singleStreamTotalSeqLen);
            blockIn.Dispose();
            Tensor resliced = SliceSeq(backend, newJointFull, singleStreamBaseSeqLen);
            newJointFull.Dispose();
            curImg.Dispose();
            curImg = resliced;
        }

        return (curImg, curEncoder);
    }

    /// <summary>Pipeline-level debug hook: dumps the post-denoise, pre-VAE latent.</summary>
    public static void DumpFinalLatent(Tensor latent) => HiDreamDebugDump.Dump("final_latent", latent);

    /// <summary>Computes <c>temb = timestep_embed(timesteps) + pooled_embed(pooled)</c>. Both run through a TimestepEmbedding (Linear → SiLU → Linear) — same shape recipe as Flux/SD3 except the pooled path takes the 2048-dim CLIP concat directly (no separate sinusoidal stage).</summary>
    private Tensor ComputeTimestepAndPooledEmbedding(IBackend backend, float timestep, Tensor pooled, int batch)
    {
        int hidden = _config.InnerDim;

        // Timestep: sinusoidal(256) → Linear → SiLU → Linear
        TensorShape sinShape = new TensorShape(batch, 256);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmbed, timestep, batch, embDim: 256);

        TensorShape hidShape = new TensorShape(batch, hidden);

        Tensor t1 = new Tensor(hidShape, DType.F32);
        backend.Linear(t1, sinEmbed, _tEmbedLinear1Weight!, _tEmbedLinear1Bias);
        sinEmbed.Dispose();

        Tensor t1Act = new Tensor(hidShape, DType.F32);
        backend.Silu(t1Act, t1);
        t1.Dispose();

        Tensor tEmb = new Tensor(hidShape, DType.F32);
        backend.Linear(tEmb, t1Act, _tEmbedLinear2Weight!, _tEmbedLinear2Bias);
        t1Act.Dispose();

        // Pooled: Linear(2048→hidden) → SiLU → Linear(hidden→hidden)
        Tensor p1 = new Tensor(hidShape, DType.F32);
        backend.Linear(p1, pooled, _pEmbedLinear1Weight!, _pEmbedLinear1Bias);

        Tensor p1Act = new Tensor(hidShape, DType.F32);
        backend.Silu(p1Act, p1);
        p1.Dispose();

        Tensor pEmb = new Tensor(hidShape, DType.F32);
        backend.Linear(pEmb, p1Act, _pEmbedLinear2Weight!, _pEmbedLinear2Bias);
        p1Act.Dispose();

        Tensor temb = new Tensor(hidShape, DType.F32);
        backend.Add(temb, tEmb, pEmb);
        tEmb.Dispose();
        pEmb.Dispose();
        return temb;
    }

    /// <summary>Projects a caption hidden state [B, S, in_dim] through caption_projection[i].linear (no bias) to [B, S, inner_dim].</summary>
    private static Tensor ProjectCaption(IBackend backend, Tensor caption, Tensor weight)
    {
        int batch = (int)caption.Shape[0];
        int seqLen = (int)caption.Shape[1];
        int outDim = (int)weight.Shape[0];
        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, caption, weight, null);
        return output;
    }

    /// <summary>Final layer: SiLU(temb) → Linear → split [shift, scale]; LayerNorm(no affine) on hidden; modulate by (1+scale)*norm + shift; final Linear to [B, S, p²*out_channels].</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor hidden, Tensor temb, int batch, int seqLen)
    {
        int dim = _config.InnerDim;
        int outDim = _config.PatchSize * _config.PatchSize * _config.OutChannels;
        TensorShape hidShape = new TensorShape(batch, seqLen, dim);

        TensorShape tembShape = new TensorShape(batch, dim);
        Tensor activated = new Tensor(tembShape, DType.F32);
        backend.Silu(activated, temb);

        TensorShape modShape = new TensorShape(batch, dim * 2);
        Tensor modParams = new Tensor(modShape, DType.F32);
        backend.Linear(modParams, activated, _finalAdaLnWeight!, _finalAdaLnBias);
        activated.Dispose();

        // GPU-resident final modulation: LayerNorm(no-affine) → norm*(1+scale)+shift, modParams split [shift, scale]
        // (diffusers HiDreamImageOutEmbed order).
        Tensor normed = new Tensor(hidShape, DType.F32);
        backend.LayerNormNoAffine(normed, hidden, 1e-6f);

        Tensor shift = new Tensor(new TensorShape(batch, dim), DType.F32);
        backend.SliceLastDim(shift, modParams, 0);
        Tensor scale = new Tensor(new TensorShape(batch, dim), DType.F32);
        backend.SliceLastDim(scale, modParams, dim);
        Tensor scalePlus1 = new Tensor(new TensorShape(batch, dim), DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        scale.Dispose();

        Tensor modulated = new Tensor(hidShape, DType.F32);
        backend.AffineBroadcastLastDim(modulated, normed, scalePlus1, shift);
        normed.Dispose();
        shift.Dispose();
        scalePlus1.Dispose();
        modParams.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected = new Tensor(outShape, DType.F32);
        backend.Linear(projected, modulated, _finalProjWeight!, _finalProjBias);
        modulated.Dispose();
        return projected;
    }

    /// <summary>Patchifies [B, C, H, W] → [B, S_img, C * patch_size²] for a square latent (the only shape we accept here — t2i always produces a square or rectangular latent that's already aligned to <c>patch_size</c>). The diffusers reference also has a non-square branch that pads to <c>max_seq</c> and produces a hidden_states_mask; we don't need that for fixed-resolution t2i.</summary>
    private static Tensor PatchifyLatent(Tensor latent, int batch, int channels, int patH, int patW, int patchSize)
    {
        int S = patH * patW;
        int F = patchSize * patchSize * channels;
        TensorShape outShape = new TensorShape(batch, S, F);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // Diffusers reshape order for square latent:
        //   x.reshape(B, C, patH, p, patW, p) -> permute(0, 2, 4, 3, 5, 1) -> reshape(B, S, p*p*C)
        // → per output token (gy, gx), the layout is [py, px, c] i.e. spatial-first within a patch,
        //   channel-last. Loop accordingly.
        int height = patH * patchSize;
        int width = patW * patchSize;
        for (int b = 0; b < batch; b++)
        {
            for (int gy = 0; gy < patH; gy++)
            {
                for (int gx = 0; gx < patW; gx++)
                {
                    int tokenIdx = gy * patW + gx;
                    int outBase = (b * S + tokenIdx) * F;
                    for (int py = 0; py < patchSize; py++)
                    {
                        for (int px = 0; px < patchSize; px++)
                        {
                            int patchPixel = py * patchSize + px;
                            int outPixelOff = patchPixel * channels;
                            int yPx = gy * patchSize + py;
                            int xPx = gx * patchSize + px;
                            for (int c = 0; c < channels; c++)
                            {
                                int srcIdx = ((b * channels + c) * height + yPx) * width + xPx;
                                outPtr[outBase + outPixelOff + c] = inPtr[srcIdx];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Unpatchifies [B, S, p²*C] → [B, C, H, W] (matching diffusers' inference branch, which permutes (0, 5, 1, 3, 2, 4) on the [B, pH, pW, p, p, C] view). The value is <b>negated</b> to match the ComfyUI reference, which returns <c>-output</c> (HiDream's velocity-prediction sign convention).</summary>
    private static Tensor UnpatchifyLatent(Tensor input, int batch, int channels, int patH, int patW, int patchSize)
    {
        int height = patH * patchSize;
        int width = patW * patchSize;
        TensorShape outShape = new TensorShape(batch, channels, height, width);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        int S = patH * patW;
        int F = patchSize * patchSize * channels;

        for (int b = 0; b < batch; b++)
        {
            for (int gy = 0; gy < patH; gy++)
            {
                for (int gx = 0; gx < patW; gx++)
                {
                    int tokenIdx = gy * patW + gx;
                    int inBase = (b * S + tokenIdx) * F;
                    for (int py = 0; py < patchSize; py++)
                    {
                        for (int px = 0; px < patchSize; px++)
                        {
                            int patchPixel = py * patchSize + px;
                            int yPx = gy * patchSize + py;
                            int xPx = gx * patchSize + px;
                            for (int c = 0; c < channels; c++)
                            {
                                int srcIdx = inBase + patchPixel * channels + c;
                                int dstIdx = ((b * channels + c) * height + yPx) * width + xPx;
                                outPtr[dstIdx] = -inPtr[srcIdx];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <summary>GPU-resident concat of two [B, S1, D] and [B, S2, D] tensors along the sequence dim → [B, S1+S2, D].</summary>
    private static Tensor ConcatSeq(IBackend backend, Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int dim = (int)a.Shape[2];
        int total = (int)a.Shape[1] + (int)b.Shape[1];
        Tensor output = new Tensor(new TensorShape(batch, total, dim), DType.F32);
        backend.Concat(output, new Tensor[] { a, b }, 1);
        return output;
    }

    /// <summary>GPU-resident slice of the first <paramref name="firstSeqLen"/> sequence positions of a [B, S, D] tensor (contiguous row-block, <see cref="IBackend.SliceRows"/>).</summary>
    private static Tensor SliceSeq(IBackend backend, Tensor input, int firstSeqLen)
    {
        int batch = (int)input.Shape[0];
        int dim = (int)input.Shape[2];
        int totalSeq = (int)input.Shape[1];
        if (firstSeqLen > totalSeq)
            throw new ArgumentOutOfRangeException(nameof(firstSeqLen),
                $"firstSeqLen={firstSeqLen} exceeds totalSeq={totalSeq}");
        Tensor output = new Tensor(new TensorShape(batch, firstSeqLen, dim), DType.F32);
        backend.SliceRows(output, input, 0);
        return output;
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Releases all tensor references held by this transformer.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _xEmbedWeight = null; _xEmbedBias = null;
            _tEmbedLinear1Weight = null; _tEmbedLinear1Bias = null;
            _tEmbedLinear2Weight = null; _tEmbedLinear2Bias = null;
            _pEmbedLinear1Weight = null; _pEmbedLinear1Bias = null;
            _pEmbedLinear2Weight = null; _pEmbedLinear2Bias = null;
            _captionProjectionWeights = null;
            _finalAdaLnWeight = null; _finalAdaLnBias = null;
            _finalProjWeight = null; _finalProjBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
