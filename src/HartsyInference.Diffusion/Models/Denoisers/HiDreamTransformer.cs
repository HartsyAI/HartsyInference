using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>HiDream-I1 image transformer (HiDreamImageTransformer2DModel) — text-to-image only.
/// <para>Architecture: 16 double-stream blocks (joint MM-attention) + 32 single-stream blocks
/// (parallel image+text attention), MoE SwiGLU FFN with a shared expert + top-2-of-4 routed experts on the
/// image side (see <see cref="HiDreamBlock"/> for the gate softmax / top-k renorm), 3-axis RoPE on
/// (layer-id, row, col).</para>
/// <para>Text conditioning is the concatenation of:
/// <list type="bullet">
/// <item>The last two Llama-3.1 hidden states projected through <c>caption_projection[0]</c> (forming the "initial encoder hidden states");</item>
/// <item>The T5-XXL hidden state projected through <c>caption_projection[-1]</c>;</item>
/// <item>One Llama hidden state per block, projected through <c>caption_projection[0]</c> and concatenated to the running encoder.</item>
/// </list>
/// The pooled CLIP-L+CLIP-G embedding is fed into the timestep MLP via the pooled embedder.</para></summary>
public sealed unsafe class HiDreamTransformer : IDisposable
{
    private readonly HiDreamConfig _config;
    private readonly HiDreamBlock[] _doubleBlocks;
    private readonly HiDreamBlock[] _singleBlocks;
    private readonly HiDreamRope _rope;
    private int _disposed;

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

        // caption_projection[i].linear.weight (no bias)
        int numCaptionProjections = _config.CaptionChannels.Length;
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

        for (int i = 0; i < _doubleBlocks.Length; i++)
            foreach (Tensor w in _doubleBlocks[i].EnumerateWeights()) yield return w;
        for (int i = 0; i < _singleBlocks.Length; i++)
            foreach (Tensor w in _singleBlocks[i].EnumerateWeights()) yield return w;

        if (_finalAdaLnWeight is not null) yield return _finalAdaLnWeight;
        if (_finalAdaLnBias is not null) yield return _finalAdaLnBias;
        if (_finalProjWeight is not null) yield return _finalProjWeight;
        if (_finalProjBias is not null) yield return _finalProjBias;
    }

    /// <summary>Forward pass for one denoising step. Predicts velocity in patchified-and-projected
    /// latent space and returns it in the same [B, C_out, H, W] layout as the input latent.</summary>
    /// <param name="latent">Noisy latent [B, in_channels, H, W].</param>
    /// <param name="timestep">Scalar timestep value (sigma * 1000 in flow matching).</param>
    /// <param name="t5Hidden">T5-XXL hidden states already projected to [B, S_t5, inner_dim] by the pipeline (caption_projection is run inside this method, so pass the raw T5 hidden of shape [B, S_t5, 4096]).</param>
    /// <param name="llamaHiddenLayers">List of Llama hidden states, one per <see cref="HiDreamConfig.LlamaLayers"/> entry. Each tensor is shape [B, S_l, caption_channels[0]] (= 4096). Length must equal NumLayers + NumSingleLayers.</param>
    /// <param name="pooledEmbeds">[B, text_emb_dim=2048] pooled CLIP-L+CLIP-G embedding.</param>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep,
        Tensor t5Hidden, IReadOnlyList<Tensor> llamaHiddenLayers, Tensor pooledEmbeds)
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

        // ── 2. Build temb = timestep_embed + pooled_embed ──
        Tensor temb = ComputeTimestepAndPooledEmbedding(backend, timestep, pooledEmbeds, batch);
        HiDreamDebugDump.Dump("temb", temb);

        // ── 3. Project caption inputs ──
        // For the per-block llama states we use caption_projection[0]; for T5 we use caption_projection[-1].
        Tensor[] llamaProj = new Tensor[llamaHiddenLayers.Count];
        for (int i = 0; i < llamaHiddenLayers.Count; i++)
        {
            llamaProj[i] = ProjectCaption(backend, llamaHiddenLayers[i], _captionProjectionWeights![0]);
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

        // ── 5. Precompute RoPE for the (image + initial_encoder + per-block-llama) sequence ──
        // We use the longest text sequence the rope table needs — the maximum of (initial_encoder_seq_len + any
        // single llama_proj_seq_len) since per-block we concat one Llama layer to the initial encoder.
        // All per-block llama layers share the same seq len (they came from the same encoder forward),
        // so we precompute once with imgSeqLen + (initialEncoderSeqLen + llama_seq_len).
        int perLayerLlamaSeqLen = (int)llamaProj[0].Shape[1];
        int doubleStreamTotalSeqLen = imgSeqLen + initialEncoderSeqLen + perLayerLlamaSeqLen;
        Tensor posIds = HiDreamRope.BuildPositionIds(imgSeqLen, patH, patW, initialEncoderSeqLen + perLayerLlamaSeqLen);
        _rope.Precompute(posIds);
        posIds.Dispose();

        // ── 6. Double-stream blocks ──
        Tensor curImg = imgTokens;
        Tensor curEncoder = initialEncoder;
        int blockId = 0;

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

            if (!ReferenceEquals(curImg, imgTokens)) curImg.Dispose();
            if (!ReferenceEquals(curEncoder, initialEncoder)) curEncoder.Dispose();
            curImg = newImg;
            curEncoder = reslicedEncoder;
            HiDreamDebugDump.Dump($"double_block_{b}_img", curImg);

            blockId++;
        }
        if (!ReferenceEquals(curEncoder, initialEncoder)) initialEncoder.Dispose();

        // ── 7. Switch to single-stream: concat current image with the running encoder, then per-block llama ──
        Tensor jointBeforeSingle = ConcatSeq(backend, curImg, curEncoder);
        if (!ReferenceEquals(curImg, imgTokens)) curImg.Dispose();
        else imgTokens.Dispose();
        curEncoder.Dispose();

        int singleStreamBaseSeqLen = imgSeqLen + initialEncoderSeqLen;
        int singleStreamTotalSeqLen = singleStreamBaseSeqLen + perLayerLlamaSeqLen;

        // Re-precompute rope to the maximum single-stream total seq length (image + enc + llama-per-block).
        Tensor posIdsSingle = HiDreamRope.BuildPositionIds(imgSeqLen, patH, patW, initialEncoderSeqLen + perLayerLlamaSeqLen);
        _rope.Precompute(posIdsSingle);
        posIdsSingle.Dispose();

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
            blockId++;
        }

        // ── 8. Extract image tokens (first imgSeqLen of curJoint) ──
        Tensor imgFinal = SliceSeq(backend, curJoint, imgSeqLen);
        curJoint.Dispose();

        // Free per-layer projections.
        for (int i = 0; i < llamaProj.Length; i++) llamaProj[i].Dispose();
        t5Proj.Dispose();

        // ── 9. Final layer ──
        Tensor projected = ApplyFinalLayer(backend, imgFinal, temb, batch, imgSeqLen);
        imgFinal.Dispose();
        temb.Dispose();
        HiDreamDebugDump.Dump("proj_out", projected);

        // ── 10. Unpatchify [B, S, p²*C] → [B, C, H, W] ──
        Tensor output = UnpatchifyLatent(projected, batch, _config.OutChannels, patH, patW, _config.PatchSize);
        projected.Dispose();
        HiDreamDebugDump.DumpOutput(output);
        return output;
    }

    /// <summary>Pipeline-level debug hook: dumps the post-denoise, pre-VAE latent.</summary>
    public static void DumpFinalLatent(Tensor latent) => HiDreamDebugDump.Dump("final_latent", latent);

    /// <summary>Computes <c>temb = timestep_embed(timesteps) + pooled_embed(pooled)</c>. Both run through
    /// a TimestepEmbedding (Linear → SiLU → Linear) — same shape recipe as Flux/SD3 except the pooled
    /// path takes the 2048-dim CLIP concat directly (no separate sinusoidal stage).</summary>
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

    /// <summary>Projects a caption hidden state [B, S, in_dim] through caption_projection[i].linear (no bias)
    /// to [B, S, inner_dim].</summary>
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

    /// <summary>Final layer: SiLU(temb) → Linear → split [shift, scale]; LayerNorm(no affine) on hidden;
    /// modulate by (1+scale)*norm + shift; final Linear to [B, S, p²*out_channels].</summary>
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

        Tensor normed = new Tensor(hidShape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, hidden, batch, seqLen, dim);

        Tensor modulated = new Tensor(hidShape, DType.F32);
        float* modPtr = (float*)modParams.DataPointer;
        float* normPtr = (float*)normed.DataPointer;
        float* modulatedPtr = (float*)modulated.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int modBase = b * dim * 2;
            for (int s = 0; s < seqLen; s++)
            {
                int vecOffset = (b * seqLen + s) * dim;
                for (int d = 0; d < dim; d++)
                {
                    // Diffusers HiDreamImageOutEmbed order: [shift, scale]
                    float shift = modPtr[modBase + d];
                    float scale = modPtr[modBase + dim + d];
                    modulatedPtr[vecOffset + d] = normPtr[vecOffset + d] * (1.0f + scale) + shift;
                }
            }
        }
        normed.Dispose();
        modParams.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected = new Tensor(outShape, DType.F32);
        backend.Linear(projected, modulated, _finalProjWeight!, _finalProjBias);
        modulated.Dispose();
        return projected;
    }

    /// <summary>Patchifies [B, C, H, W] → [B, S_img, C * patch_size²] for a square latent (the only
    /// shape we accept here — t2i always produces a square or rectangular latent that's already aligned
    /// to <c>patch_size</c>). The diffusers reference also has a non-square branch that pads to
    /// <c>max_seq</c> and produces a hidden_states_mask; we don't need that for fixed-resolution t2i.</summary>
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

    /// <summary>Unpatchifies [B, S, p²*C] → [B, C, H, W] (matching diffusers' inference branch, which
    /// permutes (0, 5, 1, 3, 2, 4) on the [B, pH, pW, p, p, C] view). The value is <b>negated</b> to match the
    /// ComfyUI reference, which returns <c>-output</c> (HiDream's velocity-prediction sign convention).</summary>
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

    /// <summary>GPU-resident slice of the first <paramref name="firstSeqLen"/> sequence positions of a [B, S, D]
    /// tensor (contiguous row-block, <see cref="IBackend.SliceRows"/>).</summary>
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
