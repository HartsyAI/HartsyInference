using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>UNet2DConditionModel for Stable Diffusion. Supports SD1.5, SDXL base, and SDXL refiner architectures via UNetConfig.</summary>
public sealed class UNet
{
    private readonly UNetConfig _config;

    // conv_in: Conv2d(inChannels, modelChannels, 3, padding=1)
    private Tensor? _convInWeight;
    private Tensor? _convInBias;

    // Timestep embedding
    private readonly TimestepEmbedding _timeEmbedding;

    // ADM conditioning (SDXL only — null for SD1.5)
    private readonly AdditionEmbedding? _addEmbedding;

    // Down blocks
    private readonly DownBlock[] _downBlocks;

    // Mid block: ResNet → CrossAttention → ResNet
    private readonly UNetResNetBlock _midResNet0;
    private readonly CrossAttentionBlock _midAttention;
    private readonly UNetResNetBlock _midResNet1;

    // Up blocks
    private readonly UpBlock[] _upBlocks;

    // conv_norm_out + conv_out
    private Tensor? _normOutWeight;
    private Tensor? _normOutBias;
    private Tensor? _convOutWeight;
    private Tensor? _convOutBias;

    /// <summary>The configuration this UNet was built with.</summary>
    public UNetConfig Config => _config;

    /// <summary>Creates a UNet with the specified configuration.</summary>
    public UNet(UNetConfig config)
    {
        _config = config;

        int[] blockCh = config.BlockOutChannels;
        int timeDim = config.ModelChannels * 4; // 1280 for SD1.5/SDXL base, 1536 for SDXL refiner
        int numBlocks = blockCh.Length;

        // Timestep embedding: sinusoidal(modelChannels) → MLP → timeDim
        _timeEmbedding = new TimestepEmbedding(config.ModelChannels, timeDim);

        // ADM conditioning for SDXL (micro-conditioning: size/crop/target scalars + pooled text)
        if (config.AdmInChannels > 0)
        {
            _addEmbedding = new AdditionEmbedding(config.AdmInChannels, timeDim, config.AdditionTimeEmbedDim);
        }

        // Down blocks
        _downBlocks = new DownBlock[numBlocks];
        for (int i = 0; i < numBlocks; i++)
        {
            int inCh = i == 0 ? config.ModelChannels : blockCh[i - 1];
            int outCh = blockCh[i];
            bool hasAttn = config.DownBlockHasAttention[i];
            bool hasDown = i < numBlocks - 1;

            int numHeads = config.NumAttentionHeads[i];
            int transformerLayers = config.TransformerLayersPerBlock[i];
            _downBlocks[i] = new DownBlock(
                inCh, outCh, timeDim, config.LayersPerBlock,
                hasAttn, hasDown, numHeads, config.CrossAttentionDim, transformerLayers);
        }

        // Mid block — uses last level's config
        int midCh = blockCh[^1];
        _midResNet0 = new UNetResNetBlock(midCh, midCh, timeDim);
        int midNumHeads = config.NumAttentionHeads[^1];
        int midTransformerLayers = config.TransformerLayersPerBlock[^1];
        _midAttention = new CrossAttentionBlock(midCh, midNumHeads, config.CrossAttentionDim, midTransformerLayers);
        _midResNet1 = new UNetResNetBlock(midCh, midCh, timeDim);

        // Up blocks (reversed channel order with skip connections)
        _upBlocks = new UpBlock[numBlocks];
        int[] reversedCh = new int[numBlocks];
        for (int i = 0; i < numBlocks; i++)
        {
            reversedCh[i] = blockCh[numBlocks - 1 - i];
        }

        for (int i = 0; i < numBlocks; i++)
        {
            int outCh = reversedCh[i];
            int prevOutCh = i == 0 ? midCh : reversedCh[i - 1];
            int inputCh = reversedCh[Math.Min(i + 1, numBlocks - 1)];
            bool hasAttn = config.UpBlockHasAttention[i];
            bool hasUp = i < numBlocks - 1;

            int upLevelIdx = numBlocks - 1 - i;
            int upNumHeads = config.NumAttentionHeads[upLevelIdx];
            int upTransformerLayers = config.TransformerLayersPerBlock[upLevelIdx];
            _upBlocks[i] = new UpBlock(
                inputCh, outCh, prevOutCh, timeDim, config.LayersPerBlock + 1,
                hasAttn, hasUp, upNumHeads, config.CrossAttentionDim, upTransformerLayers);
        }
    }

    /// <summary>Loads all UNet weights from a dictionary of named tensors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        string p = string.IsNullOrEmpty(prefix) ? "" : $"{prefix}.";

        _convInWeight = weights[$"{p}conv_in.weight"];
        _convInBias = weights[$"{p}conv_in.bias"];

        _timeEmbedding.LoadWeights(weights, $"{p}time_embedding");

        if (_addEmbedding is not null)
        {
            _addEmbedding.LoadWeights(weights, $"{p}add_embedding");
        }

        for (int i = 0; i < _downBlocks.Length; i++)
        {
            _downBlocks[i].LoadWeights(weights, $"{p}down_blocks.{i}");
        }

        _midResNet0.LoadWeights(weights, $"{p}mid_block.resnets.0");
        _midAttention.LoadWeights(weights, $"{p}mid_block.attentions.0");
        _midResNet1.LoadWeights(weights, $"{p}mid_block.resnets.1");

        for (int i = 0; i < _upBlocks.Length; i++)
        {
            _upBlocks[i].LoadWeights(weights, $"{p}up_blocks.{i}");
        }

        _normOutWeight = weights[$"{p}conv_norm_out.weight"];
        _normOutBias = weights[$"{p}conv_norm_out.bias"];
        _convOutWeight = weights[$"{p}conv_out.weight"];
        _convOutBias = weights[$"{p}conv_out.bias"];
    }

    /// <summary>Enumerates all weight tensors held by this UNet and its sub-blocks.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convInWeight is not null) yield return _convInWeight;
        if (_convInBias is not null) yield return _convInBias;
        foreach (Tensor w in _timeEmbedding.EnumerateWeights()) yield return w;
        if (_addEmbedding is not null)
        {
            foreach (Tensor w in _addEmbedding.EnumerateWeights()) yield return w;
        }
        for (int i = 0; i < _downBlocks.Length; i++)
        {
            foreach (Tensor w in _downBlocks[i].EnumerateWeights()) yield return w;
        }
        foreach (Tensor w in _midResNet0.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midAttention.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midResNet1.EnumerateWeights()) yield return w;
        for (int i = 0; i < _upBlocks.Length; i++)
        {
            foreach (Tensor w in _upBlocks[i].EnumerateWeights()) yield return w;
        }
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
        if (_convOutWeight is not null) yield return _convOutWeight;
        if (_convOutBias is not null) yield return _convOutBias;
    }

    /// <summary>Forward pass for SD1.5 (no ADM conditioning). Noisy latents [B, 4, H, W] + timestep + text embeddings [B, seqLen, crossDim] → noise prediction [B, 4, H, W].</summary>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeddings)
    {
        return Forward(backend, latent, timestep, textEmbeddings, null, default, null, null, null, null, null, null);
    }

    /// <summary>Forward pass with optional SDXL ADM conditioning. For SD1.5, pooledTextEmb and sizeCondition can be null/empty.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeddings, Tensor? pooledTextEmb, ReadOnlySpan<float> sizeCondition)
    {
        return Forward(backend, latent, timestep, textEmbeddings, pooledTextEmb, sizeCondition, null, null, null, null, null, null);
    }

    /// <summary>Forward pass with optional SDXL ADM + optional ControlNet residual injection (no IPA).</summary>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeddings, Tensor? pooledTextEmb, ReadOnlySpan<float> sizeCondition,
        IReadOnlyList<Tensor>? controlNetDownResiduals, Tensor? controlNetMidResidual)
    {
        return Forward(backend, latent, timestep, textEmbeddings, pooledTextEmb, sizeCondition, controlNetDownResiduals, controlNetMidResidual, null, null, null, null);
    }

    /// <summary>Computes the step-invariant ADM micro-conditioning embedding once so the denoise loop can pass it back through every <c>Forward</c> via <c>cachedAdmEmb</c>. Returns <c>[batch, timeDim]</c> F32; caller owns and disposes it. Null when this UNet has no ADM conditioning (SD1.5).</summary>
    public Tensor? ComputeAdmEmbedding(IBackend backend, Tensor pooledTextEmb, ReadOnlySpan<float> sizeCondition, int batch)
    {
        return _addEmbedding?.Forward(backend, pooledTextEmb, sizeCondition, batch);
    }

    /// <summary>Total cross-attention sub-layer count this UNet exposes — sum of the count across all down blocks, the mid block, and all up blocks (in that order). IP-Adapter checkpoints store one <c>to_k_ip</c> / <c>to_v_ip</c> pair per cross-attention sub-layer; the count must match exactly.</summary>
    public int CrossAttentionLayerCount
    {
        get
        {
            int count = 0;
            for (int i = 0; i < _downBlocks.Length; i++) count += _downBlocks[i].CrossAttentionLayerCount;
            count += _midAttention.NumTransformerBlocks;
            for (int i = 0; i < _upBlocks.Length; i++) count += _upBlocks[i].CrossAttentionLayerCount;
            return count;
        }
    }

    /// <summary>Forward pass with optional SDXL ADM, ControlNet residuals, and IP-Adapter image-attention injection.
    /// <para>When IPA params are supplied, each cross-attention sub-layer in the UNet runs an additional attention pass over <paramref name="ipaImageTokens"/> using its dedicated <c>to_k_ip</c> / <c>to_v_ip</c> projection from <paramref name="ipaToKIpAll"/> / <paramref name="ipaToVIpAll"/>; results are scaled by <paramref name="ipaScale"/> and added to the text-attention output before the cross-attention's shared <c>to_out</c>. The K/V lists are flat — one entry per cross-attention sub-layer, in down→mid→up traversal order — and must have length equal to <see cref="CrossAttentionLayerCount"/>.</para></summary>
    /// <param name="ipaImageTokens">Image-prompt tokens <c>[B, numImageTokens, crossAttentionDim]</c> from <see cref="HartsyInference.Diffusion.Adapters.IpAdapter.ProjectImage"/>. Null = no IPA.</param>
    /// <param name="ipaToKIpAll">Per-cross-attention-sub-layer K_ip projection weights, in UNet traversal order. Length must match <see cref="CrossAttentionLayerCount"/>.</param>
    /// <param name="ipaToVIpAll">Same as <paramref name="ipaToKIpAll"/> but for V_ip.</param>
    /// <param name="ipaScalePerLayer">Per-cross-attn-layer IP-Adapter scale array. Same length as <paramref name="ipaToKIpAll"/>. The pipeline computes this from the user's base scale + weight type + step-fraction gating once per step. Layers with scale 0 short-circuit the image attention entirely.</param>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor textEmbeddings, Tensor? pooledTextEmb, ReadOnlySpan<float> sizeCondition,
        IReadOnlyList<Tensor>? controlNetDownResiduals, Tensor? controlNetMidResidual,
        Tensor? ipaImageTokens, IReadOnlyList<Tensor>? ipaToKIpAll, IReadOnlyList<Tensor>? ipaToVIpAll, IReadOnlyList<float>? ipaScalePerLayer,
        Tensor? cachedAdmEmb = null)
    {
        bool hasIpa = ipaImageTokens is not null && ipaToKIpAll is not null && ipaToVIpAll is not null && ipaScalePerLayer is not null;
        if (hasIpa && ipaToKIpAll!.Count != CrossAttentionLayerCount)
        {
            throw new ArgumentException(
                $"IP-Adapter K/V list length ({ipaToKIpAll.Count}) doesn't match UNet cross-attention sub-layer count ({CrossAttentionLayerCount}). The checkpoint isn't for this UNet config.",
                nameof(ipaToKIpAll));
        }
        if (hasIpa && ipaScalePerLayer!.Count != CrossAttentionLayerCount)
        {
            throw new ArgumentException(
                $"IP-Adapter per-layer scale list length ({ipaScalePerLayer.Count}) doesn't match UNet cross-attention sub-layer count ({CrossAttentionLayerCount}).",
                nameof(ipaScalePerLayer));
        }
        int batch = (int)latent.Shape[0];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];

        DType dtype = latent.DType;

        // 1. Timestep embedding (computed in F32, cast to match latent dtype)
        Span<float> timesteps = stackalloc float[batch];
        timesteps.Fill(timestep);
        Tensor temb = _timeEmbedding.Forward(backend, timesteps, batch);

        // 2. ADM conditioning (SDXL only): add micro-conditioning to timestep embedding.
        //    The ADM embedding is step-invariant (pooled text + size scalars only) — pipelines that run a
        //    fixed conditioning across the loop pass it precomputed via cachedAdmEmb, which skips the
        //    per-forward host sinusoid build + pooled DataPointer read (a GPU→host sync per forward).
        if (_addEmbedding is not null && (cachedAdmEmb is not null || pooledTextEmb is not null))
        {
            Tensor addEmb = cachedAdmEmb ?? _addEmbedding.Forward(backend, pooledTextEmb!, sizeCondition, batch);
            Tensor combinedTemb = new Tensor(temb.Shape, DType.F32);
            backend.Add(combinedTemb, temb, addEmb);
            temb.Dispose();
            if (cachedAdmEmb is null) addEmb.Dispose();
            temb = combinedTemb;
        }

        // Cast temb to match latent dtype (embeddings are computed in F32). temb is locally
        // owned — let the helper dispose source on cast.
        temb = Utilities.DtypeCastHelper.EnsureDtype(backend, temb, dtype);

        // Cast text embeddings to match latent dtype (CLIP produces F32, UNet may run F16).
        // textEmbeddings is a parameter — don't dispose source. Track ownership of the casted
        // copy so we can dispose it later if a cast actually happened.
        Tensor textEmbeddingsOriginal = textEmbeddings;
        textEmbeddings = Utilities.DtypeCastHelper.EnsureDtype(backend, textEmbeddings, dtype, disposeSourceOnCast: false);
        bool ownsTextEmbeddings = textEmbeddings != textEmbeddingsOriginal;

        // 3. conv_in
        TensorShape convInShape = new TensorShape(batch, _config.ModelChannels, height, width);
        Tensor hidden = new Tensor(convInShape, dtype);
        backend.Conv2D(hidden, latent, _convInWeight!, _convInBias, 1, 1, 1, 1);

        // 4. Down blocks — collect all skip connections.
        //    IPA: maintain a flat offset into the per-cross-attn-layer K/V lists, advancing
        //    by each block's cross-attn count. Order matches diffusers: down → mid → up.
        List<Tensor> allSkips = new List<Tensor> { UNetBlockHelpers.CloneOnDevice(backend, hidden) };
        int ipaOffset = 0;
        for (int i = 0; i < _downBlocks.Length; i++)
        {
            (Tensor downOut, List<Tensor> skips) = _downBlocks[i].Forward(backend, hidden, temb, textEmbeddings,
                ipaImageTokens, ipaToKIpAll, ipaToVIpAll, ipaOffset, ipaScalePerLayer);
            ipaOffset += _downBlocks[i].CrossAttentionLayerCount;
            hidden.Dispose();
            hidden = downOut;
            allSkips.AddRange(skips);
        }

        // 4b. ControlNet residual injection — add CN residuals onto each skip in place.
        //     Must run before the up blocks consume them; allSkips index 0 is the conv_in
        //     output and matches the diffusers convention exactly.
        if (controlNetDownResiduals is not null)
        {
            if (controlNetDownResiduals.Count != allSkips.Count)
            {
                throw new ArgumentException(
                    $"ControlNet down residual count ({controlNetDownResiduals.Count}) must match UNet skip count ({allSkips.Count}). " +
                    "The ControlNet checkpoint's encoder shape doesn't match this UNet config.",
                    nameof(controlNetDownResiduals));
            }
            for (int i = 0; i < allSkips.Count; i++)
            {
                Tensor sum = new Tensor(allSkips[i].Shape, allSkips[i].DType);
                backend.Add(sum, allSkips[i], controlNetDownResiduals[i]);
                allSkips[i].Dispose();
                allSkips[i] = sum;
            }
        }

        // 5. Mid block — IPA injects on the single CrossAttentionBlock between the two ResNets.
        Tensor midRes0 = _midResNet0.Forward(backend, hidden, temb);
        hidden.Dispose();

        Tensor midAttn = _midAttention.Forward(backend, midRes0, textEmbeddings,
            ipaImageTokens, ipaToKIpAll, ipaToVIpAll, ipaOffset, ipaScalePerLayer);
        ipaOffset += _midAttention.NumTransformerBlocks;
        midRes0.Dispose();

        Tensor midRes1 = _midResNet1.Forward(backend, midAttn, temb);
        midAttn.Dispose();

        // 5b. ControlNet mid residual injection.
        if (controlNetMidResidual is not null)
        {
            Tensor sumMid = new Tensor(midRes1.Shape, midRes1.DType);
            backend.Add(sumMid, midRes1, controlNetMidResidual);
            midRes1.Dispose();
            midRes1 = sumMid;
        }

        hidden = midRes1;

        // 6. Up blocks — consume skip connections in reverse, continue advancing the IPA cursor.
        for (int i = 0; i < _upBlocks.Length; i++)
        {
            Tensor upOut = _upBlocks[i].Forward(backend, hidden, temb, textEmbeddings, allSkips,
                ipaImageTokens, ipaToKIpAll, ipaToVIpAll, ipaOffset, ipaScalePerLayer);
            ipaOffset += _upBlocks[i].CrossAttentionLayerCount;
            hidden.Dispose();
            hidden = upOut;
        }

        temb.Dispose();
        if (ownsTextEmbeddings) textEmbeddings.Dispose();

        // 7. Fused GroupNorm+SiLU → conv_out
        int finalH = (int)hidden.Shape[2];
        int finalW = (int)hidden.Shape[3];
        int finalCh = _config.ModelChannels;

        TensorShape finalShape = new TensorShape(batch, finalCh, finalH, finalW);
        Tensor siluOut = new Tensor(finalShape, dtype);
        backend.GroupNormSilu(siluOut, hidden, _normOutWeight!, _normOutBias!, _config.NormNumGroups, _config.NormEps);
        hidden.Dispose();

        TensorShape outShape = new TensorShape(batch, _config.OutChannels, finalH, finalW);
        Tensor output = new Tensor(outShape, dtype);
        backend.Conv2D(output, siluOut, _convOutWeight!, _convOutBias, 1, 1, 1, 1);
        siluOut.Dispose();

        return output;
    }
}
