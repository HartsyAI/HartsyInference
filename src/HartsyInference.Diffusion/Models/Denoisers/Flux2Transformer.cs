using HartsyInference.Core.Backends;
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
        _xEmbedWeight = weights["x_embedder.weight"];
        _contextEmbedWeight = weights["context_embedder.weight"];

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
            _doubleBlocks[i].LoadWeights(weights, $"transformer_blocks.{i}");

        for (int i = 0; i < _config.DepthSingleBlocks; i++)
            _singleBlocks[i].LoadWeights(weights, $"single_transformer_blocks.{i}");

        _normOutLinearWeight = weights["norm_out.linear.weight"];
        _projOutWeight = weights["proj_out.weight"];
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
        float sigma, float guidanceScale, int hPacked, int wPacked)
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

        // ── 2. Timestep (+ guidance) embedding ──
        Tensor temb = ComputeTimestepEmbedding(backend, sigma, guidanceScale, batch);

        // ── 3. Shared modulation projections — computed once, reused across all blocks ──
        Tensor[] imgMod = _doubleModImg.Forward(backend, temb);   // 6 tensors [B, hidden]
        Tensor[] txtMod = _doubleModTxt.Forward(backend, temb);   // 6 tensors
        Tensor[] sgMod = _singleMod.Forward(backend, temb);       // 3 tensors

        // ── 4. Precompute 4-axis RoPE for [text, image] concat sequence ──
        Tensor posIds = Flux2PosEmbed.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        _rope.Precompute(posIds);
        posIds.Dispose();

        // ── 5. Double-stream blocks (text + image as two parallel streams sharing joint attn) ──
        Tensor currentImg = imgTokens;
        Tensor currentTxt = txtTokens;
        for (int i = 0; i < _config.Depth; i++)
        {
            (Tensor newImg, Tensor newTxt) = _doubleBlocks[i].Forward(
                backend, currentImg, currentTxt, imgMod, txtMod, _rope);
            if (!ReferenceEquals(currentImg, imgTokens)) currentImg.Dispose();
            if (!ReferenceEquals(currentTxt, txtTokens)) currentTxt.Dispose();
            currentImg = newImg;
            currentTxt = newTxt;
        }

        // ── 6. Concatenate [text, image] for single-stream processing (device op — the old host copy was a
        // full D2H sync of both block-loop outputs every forward) ──
        TensorShape concatShape = new TensorShape(batch, totalSeqLen, hidden);
        Tensor x = new Tensor(concatShape, DType.F32);
        backend.Concat(x, new Tensor[] { currentTxt, currentImg }, 1);
        if (!ReferenceEquals(currentImg, imgTokens)) currentImg.Dispose();
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
        TensorShape imgOutShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgOut = new Tensor(imgOutShape, DType.F32);
        if (batch == 1)
            backend.SliceRows(imgOut, x, txtSeqLen);
        else
            ExtractImageTokens(imgOut, x, batch, txtSeqLen, imgSeqLen, hidden);
        x.Dispose();

        // ── 9. Final layer: AdaLN-Continuous (shift/scale only) + proj_out ──
        Tensor output = ApplyFinalLayer(backend, imgOut, temb, batch, imgSeqLen);
        imgOut.Dispose();
        temb.Dispose();
        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();
        for (int i = 0; i < sgMod.Length; i++) sgMod[i].Dispose();

        return output;
    }

    private Tensor ComputeTimestepEmbedding(IBackend backend, float sigma, float guidanceScale, int batch)
    {
        int hidden = _config.HiddenSize;
        int inCh = _config.TimestepChannels;

        // Pipeline passes timestep/1000 as `sigma`, so scale back up to match the Flux.2
        // reference `timestep = timestep * 1000` before sinusoidal embedding.
        float scaledTimestep = sigma * 1000.0f;

        TensorShape sinShape = new TensorShape(batch, inCh);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        ComputeSinusoidalTimestep(sinEmbed, scaledTimestep, batch, inCh);

        // Timestep MLP: Linear(inCh, hidden) → SiLU → Linear(hidden, hidden)
        TensorShape hidShape = new TensorShape(batch, hidden);
        Tensor t1 = new Tensor(hidShape, DType.F32);
        backend.Linear(t1, sinEmbed, _timestepLinear1Weight!, null);
        sinEmbed.Dispose();
        Tensor t1Act = new Tensor(hidShape, DType.F32);
        backend.Silu(t1Act, t1);
        t1.Dispose();
        Tensor temb = new Tensor(hidShape, DType.F32);
        backend.Linear(temb, t1Act, _timestepLinear2Weight!, null);
        t1Act.Dispose();

        if (_config.GuidanceEmbed && _guidanceLinear1Weight != null)
        {
            // guidance_embed adds to temb (Dev only). guidance is in same units as timestep
            // (× 1000 internally per the diffusers reference).
            Tensor gSin = new Tensor(sinShape, DType.F32);
            ComputeSinusoidalTimestep(gSin, guidanceScale * 1000.0f, batch, inCh);

            Tensor g1 = new Tensor(hidShape, DType.F32);
            backend.Linear(g1, gSin, _guidanceLinear1Weight!, null);
            gSin.Dispose();
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
        GC.SuppressFinalize(this);
    }
}
