using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>ControlNet adapter for spatial conditioning of UNet-based denoisers (SD1.5, SDXL). The architecture is a near-mirror of the base UNet's encoder half (timestep + ADM embedding, conv_in, down blocks, mid block) with two ControlNet-specific additions: a hint encoder that downsamples the control image to the latent grid, and a tower of 1×1 zero-initialized convolutions that gate every skip-connection output before it leaves the adapter.
///
/// <para>The forward pass produces residuals — one per UNet skip-connection injection point plus one for the mid block — which the caller (a pipeline) adds onto the base UNet's corresponding tensors before the up blocks consume them. With the zero-init scheme, an untrained / zeroed-out ControlNet contributes nothing and the base model's output is unchanged; a trained ControlNet steers the generation toward the conditioning signal proportional to <c>conditioningScale</c>.</para>
///
/// <para>Implementation reuses <see cref="DownBlock"/>, <see cref="UNetResNetBlock"/>, <see cref="CrossAttentionBlock"/>, <see cref="TimestepEmbedding"/>, and <see cref="AdditionEmbedding"/> verbatim — the diffusers ControlNet uses the same module shapes as the base UNet, so checkpoint key names line up 1:1 (<c>down_blocks.{i}.resnets.{j}.*</c>, <c>mid_block.resnets.{i}.*</c>, etc.). Only the hint encoder and the zero conv tower are new.</para>
///
/// <para><b>Flux ControlNet</b> uses a different architecture (DiT blocks, no UNet down/mid) and lives in the separate <see cref="FluxControlNet"/> adapter. This class covers <see cref="ControlNetBaseModel.Sd15"/> and <see cref="ControlNetBaseModel.Sdxl"/>.</para></summary>
public sealed unsafe class ControlNet : IDisposable
{
    private readonly ControlNetConfig _config;
    private readonly UNetConfig _baseConfig;
    private int _disposed;

    // ── Mirror of the base UNet's encoder half ─────────────────────
    private readonly TimestepEmbedding _timeEmbedding;
    private readonly AdditionEmbedding? _addEmbedding;
    private Tensor? _convInWeight, _convInBias;
    private readonly DownBlock[] _downBlocks;
    private readonly UNetResNetBlock _midResNet0;
    private readonly CrossAttentionBlock _midAttention;
    private readonly UNetResNetBlock _midResNet1;

    // ── ControlNet-specific ─────────────────────────────────────────
    private readonly ControlNetCondEmbedding _condEmbedding;
    /// <summary>1×1 zero-initialized convs, one per UNet skip injection point. Count derived from the base config: <c>1 (conv_in)</c> plus, for each down block, <c>LayersPerBlock</c> resnet/attn skips and <c>1</c> downsample skip when the block downsamples.</summary>
    private readonly Tensor?[] _zeroConvDownWeights;
    private readonly Tensor?[] _zeroConvDownBiases;
    private Tensor? _zeroConvMidWeight, _zeroConvMidBias;

    /// <summary>Total skip count this ControlNet emits. Pipelines can pre-validate that the base UNet's skip count matches before calling Forward.</summary>
    public int DownResidualCount => _zeroConvDownWeights.Length;

    /// <summary>Creates a ControlNet adapter from the ControlNet config plus the base UNet's config (needed to size the mirrored encoder half — block channels, attention heads, layers, ADM dim, etc.).</summary>
    public ControlNet(ControlNetConfig config, UNetConfig baseConfig)
    {
        if (config.BaseModel == ControlNetBaseModel.Flux)
        {
            throw new NotSupportedException(
                "Flux ControlNet uses a DiT-based architecture and is not handled by this adapter. " +
                "Construct a FluxControlNet from ControlNetFile.FluxConfig instead.");
        }
        _config = config;
        _baseConfig = baseConfig;

        int[] blockCh = baseConfig.BlockOutChannels;
        int timeDim = baseConfig.ModelChannels * 4;
        int numBlocks = blockCh.Length;

        _timeEmbedding = new TimestepEmbedding(baseConfig.ModelChannels, timeDim);
        if (baseConfig.AdmInChannels > 0)
        {
            _addEmbedding = new AdditionEmbedding(baseConfig.AdmInChannels, timeDim, baseConfig.AdditionTimeEmbedDim);
        }

        _downBlocks = new DownBlock[numBlocks];
        for (int i = 0; i < numBlocks; i++)
        {
            int inCh = i == 0 ? baseConfig.ModelChannels : blockCh[i - 1];
            int outCh = blockCh[i];
            bool hasAttn = baseConfig.DownBlockHasAttention[i];
            bool hasDown = i < numBlocks - 1;
            int numHeads = baseConfig.NumAttentionHeads[i];
            int transformerLayers = baseConfig.TransformerLayersPerBlock[i];
            _downBlocks[i] = new DownBlock(
                inCh, outCh, timeDim, baseConfig.LayersPerBlock,
                hasAttn, hasDown, numHeads, baseConfig.CrossAttentionDim, transformerLayers);
        }

        int midCh = blockCh[^1];
        _midResNet0 = new UNetResNetBlock(midCh, midCh, timeDim);
        int midNumHeads = baseConfig.NumAttentionHeads[^1];
        int midTransformerLayers = baseConfig.TransformerLayersPerBlock[^1];
        _midAttention = new CrossAttentionBlock(midCh, midNumHeads, baseConfig.CrossAttentionDim, midTransformerLayers);
        _midResNet1 = new UNetResNetBlock(midCh, midCh, timeDim);

        _condEmbedding = new ControlNetCondEmbedding(config.ConditioningChannels, baseConfig.ModelChannels);

        // Skip count: 1 (conv_in) + per-block (LayersPerBlock + downsample-or-not).
        int numDownSkips = 1;
        for (int i = 0; i < numBlocks; i++)
        {
            numDownSkips += baseConfig.LayersPerBlock;
            if (i < numBlocks - 1) numDownSkips++; // downsample produces one extra skip
        }
        _zeroConvDownWeights = new Tensor?[numDownSkips];
        _zeroConvDownBiases = new Tensor?[numDownSkips];
    }

    /// <summary>Loads ControlNet weights. Expected key layout matches diffusers' <c>ControlNetModel</c>: same prefix scheme as the base UNet (<c>conv_in</c>, <c>time_embedding</c>, <c>add_embedding</c>, <c>down_blocks.{i}.*</c>, <c>mid_block.*</c>) plus <c>controlnet_cond_embedding.*</c>, <c>controlnet_down_blocks.{i}.weight/bias</c> (zero convs), and <c>controlnet_mid_block.weight/bias</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _convInWeight = weights["conv_in.weight"];
        _convInBias = weights["conv_in.bias"];
        _timeEmbedding.LoadWeights(weights, "time_embedding");
        if (_addEmbedding is not null)
        {
            _addEmbedding.LoadWeights(weights, "add_embedding");
        }
        for (int i = 0; i < _downBlocks.Length; i++)
        {
            _downBlocks[i].LoadWeights(weights, $"down_blocks.{i}");
        }
        _midResNet0.LoadWeights(weights, "mid_block.resnets.0");
        _midAttention.LoadWeights(weights, "mid_block.attentions.0");
        _midResNet1.LoadWeights(weights, "mid_block.resnets.1");
        _condEmbedding.LoadWeights(weights);
        for (int i = 0; i < _zeroConvDownWeights.Length; i++)
        {
            _zeroConvDownWeights[i] = weights[$"controlnet_down_blocks.{i}.weight"];
            _zeroConvDownBiases[i] = weights[$"controlnet_down_blocks.{i}.bias"];
        }
        _zeroConvMidWeight = weights["controlnet_mid_block.weight"];
        _zeroConvMidBias = weights["controlnet_mid_block.bias"];
    }

    /// <summary>Forward pass. Produces residuals to inject into the base UNet — one per skip connection, plus one for the mid block. Caller adds these onto the corresponding base-model tensors before the up blocks run.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">Noisy latent <c>[B, 4, H, W]</c> from the current denoising step. ControlNet sees the same input the base UNet does.</param>
    /// <param name="conditionImage">Control image <c>[B, condChannels, latentH·8, latentW·8]</c> (canny edges, depth map, openpose skeleton, …) in <c>[-1, 1]</c> or <c>[0, 1]</c> per the checkpoint's expected range. The hint encoder downsamples internally to match the latent grid.</param>
    /// <param name="timestep">Current denoising timestep (scalar — ControlNet uses the same schedule as the base model).</param>
    /// <param name="textEmbeddings">Text encoder hidden states <c>[B, seqLen, crossDim]</c>. SDXL ControlNet uses dual-CLIP embeddings just like the base UNet.</param>
    /// <param name="pooledTextEmb">SDXL: pooled CLIP-G output <c>[B, 1280]</c>. Null for SD1.5.</param>
    /// <param name="sizeCondition">SDXL: micro-conditioning scalars <c>[origH, origW, cropTop, cropLeft, targetH, targetW]</c>. Empty for SD1.5.</param>
    /// <param name="conditioningScale">Multiplier applied to every output residual (0 = ControlNet contributes nothing, 1 = full strength). Per-block control would require structural changes to the zero-conv layout — out of scope for v1.</param>
    /// <returns><c>downResiduals</c>: one per UNet skip slot, in skip-list order; <c>midResidual</c>: residual added onto the mid block's output. Caller owns and must dispose all returned tensors.</returns>
    public (Tensor[] downResiduals, Tensor midResidual) Forward(
        IBackend backend,
        Tensor latent,
        Tensor conditionImage,
        float timestep,
        Tensor textEmbeddings,
        Tensor? pooledTextEmb,
        ReadOnlySpan<float> sizeCondition,
        float conditioningScale = 1.0f)
    {
        ThrowIfDisposed();
        int batch = (int)latent.Shape[0];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];
        DType dtype = latent.DType;

        // 1. Conditioning hint: downsample control image to latent grid.
        //    Cast to UNet dtype if needed (callers usually hand us a F32 image and a F16 UNet).
        Tensor condInput = Utilities.DtypeCastHelper.EnsureDtype(backend, conditionImage, dtype, disposeSourceOnCast: false);
        Tensor condHint = _condEmbedding.Forward(backend, condInput);
        if (condInput != conditionImage) condInput.Dispose();

        // 2. Timestep + ADM embedding (matches UNet.Forward exactly).
        Span<float> timesteps = stackalloc float[batch];
        timesteps.Fill(timestep);
        Tensor temb = _timeEmbedding.Forward(backend, timesteps, batch);
        if (_addEmbedding is not null && pooledTextEmb is not null)
        {
            Tensor addEmb = _addEmbedding.Forward(backend, pooledTextEmb, sizeCondition, batch);
            Tensor combinedTemb = new Tensor(temb.Shape, DType.F32);
            backend.Add(combinedTemb, temb, addEmb);
            temb.Dispose();
            addEmb.Dispose();
            temb = combinedTemb;
        }
        // temb is locally owned (computed above) — let the helper dispose source on cast.
        temb = Utilities.DtypeCastHelper.EnsureDtype(backend, temb, dtype);

        // textEmbeddings is a parameter — don't dispose source. Track ownership of the casted
        // copy so we can dispose it later if a cast actually happened.
        Tensor textEmbeddingsOriginal = textEmbeddings;
        textEmbeddings = Utilities.DtypeCastHelper.EnsureDtype(backend, textEmbeddings, dtype, disposeSourceOnCast: false);
        bool ownsText = textEmbeddings != textEmbeddingsOriginal;

        // 3. conv_in
        TensorShape convInShape = new TensorShape(batch, _baseConfig.ModelChannels, height, width);
        Tensor hidden = new Tensor(convInShape, dtype);
        backend.Conv2D(hidden, latent, _convInWeight!, _convInBias, 1, 1, 1, 1);

        // 4. Inject conditioning hint: hidden = hidden + condHint
        Tensor sumHidden = new Tensor(hidden.Shape, dtype);
        backend.Add(sumHidden, hidden, condHint);
        hidden.Dispose();
        condHint.Dispose();
        hidden = sumHidden;

        // 5. Down blocks — collect skips identically to UNet.Forward.
        List<Tensor> allSkips = new() { hidden.To(hidden.Device) };
        for (int i = 0; i < _downBlocks.Length; i++)
        {
            (Tensor downOut, List<Tensor> skips) = _downBlocks[i].Forward(backend, hidden, temb, textEmbeddings);
            hidden.Dispose();
            hidden = downOut;
            allSkips.AddRange(skips);
        }

        // 6. Mid block (resnet → cross-attn → resnet).
        Tensor midRes0 = _midResNet0.Forward(backend, hidden, temb);
        hidden.Dispose();
        Tensor midAttn = _midAttention.Forward(backend, midRes0, textEmbeddings);
        midRes0.Dispose();
        Tensor midRes1 = _midResNet1.Forward(backend, midAttn, temb);
        midAttn.Dispose();

        temb.Dispose();
        if (ownsText) textEmbeddings.Dispose();

        // 7. Apply zero convs to each skip output and the mid output. Each zero conv is a
        //    1×1 Conv2D from inCh → inCh (the zero-init weight + bias are loaded as-is from
        //    the checkpoint; pretrained ControlNets have non-zero weights). Conditioning
        //    scale applied here so the pipeline can do plain element-wise additions.
        if (allSkips.Count != _zeroConvDownWeights.Length)
        {
            throw new InvalidOperationException(
                $"ControlNet skip count mismatch: got {allSkips.Count} skips from down blocks but loaded " +
                $"{_zeroConvDownWeights.Length} zero convs. Base UNet config doesn't match the ControlNet checkpoint.");
        }
        Tensor[] downResiduals = new Tensor[allSkips.Count];
        bool needsScale = MathF.Abs(conditioningScale - 1.0f) > 1e-6f;
        for (int i = 0; i < allSkips.Count; i++)
        {
            Tensor skip = allSkips[i];
            int outCh = (int)_zeroConvDownWeights[i]!.Shape[0];
            TensorShape outShape = new TensorShape((int)skip.Shape[0], outCh, (int)skip.Shape[2], (int)skip.Shape[3]);
            Tensor residual = new Tensor(outShape, dtype);
            backend.Conv2D(residual, skip, _zeroConvDownWeights[i]!, _zeroConvDownBiases[i], 1, 1, 0, 0);
            skip.Dispose();
            if (needsScale)
            {
                Tensor scaled = new Tensor(residual.Shape, dtype);
                backend.Scale(scaled, residual, conditioningScale);
                residual.Dispose();
                residual = scaled;
            }
            downResiduals[i] = residual;
        }

        int midOutCh = (int)_zeroConvMidWeight!.Shape[0];
        TensorShape midOutShape = new TensorShape(
            (int)midRes1.Shape[0], midOutCh, (int)midRes1.Shape[2], (int)midRes1.Shape[3]);
        Tensor midResidual = new Tensor(midOutShape, dtype);
        backend.Conv2D(midResidual, midRes1, _zeroConvMidWeight!, _zeroConvMidBias, 1, 1, 0, 0);
        midRes1.Dispose();
        if (needsScale)
        {
            Tensor scaledMid = new Tensor(midResidual.Shape, dtype);
            backend.Scale(scaledMid, midResidual, conditioningScale);
            midResidual.Dispose();
            midResidual = scaledMid;
        }

        return (downResiduals, midResidual);
    }

    /// <summary>Runs every supplied ControlNet at the current step and sums their residuals into a single set. Each ControlNet sees the same latent + timestep + cond text embedding and produces its own per-skip residual stack and mid residual; this helper collapses them by element-wise addition so the UNet path doesn't need to know how many ControlNets are stacked. For SD1.5 pass <paramref name="condPooled"/> = null and an empty <paramref name="sizeCondition"/>. Caller owns and disposes all returned tensors.</summary>
    public static (Tensor[] downResiduals, Tensor midResidual) ForwardStacked(
        IBackend backend,
        IReadOnlyList<ControlNetConditioning> controlNets,
        Tensor latent,
        float timestep,
        Tensor condTextEmb,
        Tensor? condPooled,
        ReadOnlySpan<float> sizeCondition)
    {
        ControlNetConditioning first = controlNets[0];
        (Tensor[] down, Tensor mid) = first.Adapter.Forward(backend, latent, first.ConditionImage, timestep, condTextEmb, condPooled, sizeCondition, first.Scale);
        for (int c = 1; c < controlNets.Count; c++)
        {
            ControlNetConditioning next = controlNets[c];
            (Tensor[] downNext, Tensor midNext) = next.Adapter.Forward(backend, latent, next.ConditionImage, timestep, condTextEmb, condPooled, sizeCondition, next.Scale);
            if (downNext.Length != down.Length)
            {
                foreach (Tensor d in downNext) d.Dispose();
                midNext.Dispose();
                throw new InvalidOperationException(
                    $"Stacked ControlNet residual count mismatch: {down.Length} vs {downNext.Length}. " +
                    "All stacked ControlNets must target the same base UNet config.");
            }
            for (int i = 0; i < down.Length; i++)
            {
                Tensor sum = new Tensor(down[i].Shape, down[i].DType);
                backend.Add(sum, down[i], downNext[i]);
                down[i].Dispose();
                downNext[i].Dispose();
                down[i] = sum;
            }
            Tensor sumMid = new Tensor(mid.Shape, mid.DType);
            backend.Add(sumMid, mid, midNext);
            mid.Dispose();
            midNext.Dispose();
            mid = sumMid;
        }
        return (down, mid);
    }

    /// <summary>Yields all weight tensors for GPU preloading / freeing.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convInWeight is not null) yield return _convInWeight;
        if (_convInBias is not null) yield return _convInBias;
        foreach (Tensor w in _timeEmbedding.EnumerateWeights()) yield return w;
        if (_addEmbedding is not null)
        {
            foreach (Tensor w in _addEmbedding.EnumerateWeights()) yield return w;
        }
        foreach (Tensor w in _condEmbedding.EnumerateWeights()) yield return w;
        for (int i = 0; i < _downBlocks.Length; i++)
        {
            foreach (Tensor w in _downBlocks[i].EnumerateWeights()) yield return w;
        }
        foreach (Tensor w in _midResNet0.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midAttention.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midResNet1.EnumerateWeights()) yield return w;
        foreach (Tensor? w in _zeroConvDownWeights) if (w is not null) yield return w;
        foreach (Tensor? b in _zeroConvDownBiases) if (b is not null) yield return b;
        if (_zeroConvMidWeight is not null) yield return _zeroConvMidWeight;
        if (_zeroConvMidBias is not null) yield return _zeroConvMidBias;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Releases all tensor references. The condition encoder and the underlying weight tensors are not disposed here — they're owned by the safetensors loader that produced them and live until the loader itself disposes.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _convInWeight = null; _convInBias = null;
            Array.Clear(_zeroConvDownWeights);
            Array.Clear(_zeroConvDownBiases);
            _zeroConvMidWeight = null; _zeroConvMidBias = null;
            _condEmbedding.Dispose();
        }
        GC.SuppressFinalize(this);
    }
}
