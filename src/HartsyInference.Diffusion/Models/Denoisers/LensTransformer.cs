using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Microsoft Lens MMDiT (<c>LensTransformer2DModel</c>). Processes packed image patch tokens (128 channels per token after the pipeline-side 2×2 patchify of the 32-channel Flux.2 VAE latent) plus four-layer GPT-OSS text features (each <c>[B, S_txt, 2880]</c>) through 48 dual-stream blocks. Top-level layout mirrors <c>lens/transformer.py:LensTransformer2DModel</c>: <c>img_in</c> Linear → per-layer <c>txt_norm[i]</c> RMSNorm + channel-concat (4 × 2880 = 11520) → <c>txt_in</c> Linear(11520 → 1536) → <c>time_text_embed</c> sinusoidal+MLP → 48 × <see cref="LensTransformerBlock"/> → <c>norm_out</c> AdaLN-Continuous → <c>proj_out</c> Linear(1536 → 128). Output is the predicted velocity in packed token form <c>[B, S_img, 128]</c>; pipeline rearranges to <c>[B, 32, H/16, W/16]</c> for the VAE.</summary>
public sealed unsafe class LensTransformer : IDisposable
{
    private readonly LensConfig _config;
    private readonly LensTransformerBlock[] _blocks;
    private readonly LensRope _rope;
    private int _disposed;

    // Per-prompt text-token cache: the four encoder captures are constant across denoise steps, so the
    // txt_norm → concat → txt_in stack runs ONCE per prompt instead of once per forward (2 entries: cond + uncond,
    // keyed on the caller's encoder-layer list reference). Cleared on Dispose.
    private readonly List<(object Key, Tensor Tokens)> _txtTokenCache = new(2);

    private Tensor? _imgInWeight, _imgInBias;

    private Tensor?[] _txtNormWeights;

    private Tensor? _txtInWeight, _txtInBias;

    private Tensor? _timestepLinear1Weight, _timestepLinear1Bias;
    private Tensor? _timestepLinear2Weight, _timestepLinear2Bias;

    private Tensor? _normOutLinearWeight, _normOutLinearBias;
    private Tensor? _projOutWeight, _projOutBias;

    /// <summary>Creates a Lens transformer from configuration.</summary>
    public LensTransformer(LensConfig config)
    {
        _config = config;

        _blocks = new LensTransformerBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            _blocks[i] = new LensTransformerBlock(
                config.HiddenSize,
                config.NumHeads,
                config.HeadDim,
                config.MlpDim,
                config.QkNormEps,
                config.StreamNormEps);
        }

        _rope = new LensRope(config.AxesDimsRope, config.RopeTheta);
        _txtNormWeights = new Tensor?[config.SelectedEncoderLayers.Length];
    }

    /// <summary>Loads all transformer weights from named tensors using diffusers naming. Expects four <c>txt_norm.{i}.weight</c> entries (one per selected encoder layer) — see <see cref="LensConfig.SelectedEncoderLayers"/>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _imgInWeight = weights["img_in.weight"];
        _imgInBias = weights["img_in.bias"];

        for (int i = 0; i < _config.SelectedEncoderLayers.Length; i++)
            _txtNormWeights[i] = TensorCasts.EnsureF32(weights[$"txt_norm.{i}.weight"]);

        _txtInWeight = weights["txt_in.weight"];
        _txtInBias = weights["txt_in.bias"];

        _timestepLinear1Weight = weights["time_text_embed.timestep_embedder.linear_1.weight"];
        _timestepLinear1Bias = weights["time_text_embed.timestep_embedder.linear_1.bias"];
        _timestepLinear2Weight = weights["time_text_embed.timestep_embedder.linear_2.weight"];
        _timestepLinear2Bias = weights["time_text_embed.timestep_embedder.linear_2.bias"];

        for (int i = 0; i < _config.NumLayers; i++)
            _blocks[i].LoadWeights(weights, $"transformer_blocks.{i}");

        _normOutLinearWeight = weights["norm_out.linear.weight"];
        _normOutLinearBias = weights["norm_out.linear.bias"];
        _projOutWeight = weights["proj_out.weight"];
        _projOutBias = weights["proj_out.bias"];
    }

    /// <summary>Yields every weight tensor for GPU preloading via <see cref="IBackend.PreloadWeights"/>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_imgInWeight is not null) yield return _imgInWeight;
        if (_imgInBias is not null) yield return _imgInBias;

        for (int i = 0; i < _txtNormWeights.Length; i++)
            if (_txtNormWeights[i] is not null) yield return _txtNormWeights[i]!;

        if (_txtInWeight is not null) yield return _txtInWeight;
        if (_txtInBias is not null) yield return _txtInBias;
        if (_timestepLinear1Weight is not null) yield return _timestepLinear1Weight;
        if (_timestepLinear1Bias is not null) yield return _timestepLinear1Bias;
        if (_timestepLinear2Weight is not null) yield return _timestepLinear2Weight;
        if (_timestepLinear2Bias is not null) yield return _timestepLinear2Bias;

        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;

        if (_normOutLinearWeight is not null) yield return _normOutLinearWeight;
        if (_normOutLinearBias is not null) yield return _normOutLinearBias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Forward pass: predicts velocity for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="packedLatent">Packed latent tokens <c>[B, imgSeqLen, in_channels]</c> (128 channels for the released weights).</param>
    /// <param name="encoderLayers">Four GPT-OSS hidden-state captures at layers [5, 11, 17, 23]; each <c>[B, txtSeqLen, encoderHiddenDim]</c>.</param>
    /// <param name="timestep">Normalized timestep — pipeline passes raw scheduler t scaled to <c>[0, 1]</c> (i.e. <c>t / 1000</c>); the embedder multiplies by 1000 internally to recover the per-step sinusoidal input range.</param>
    /// <param name="hPacked">Packed-grid height (<c>latent_h</c>).</param>
    /// <param name="wPacked">Packed-grid width (<c>latent_w</c>).</param>
    public Tensor Forward(IBackend backend, Tensor packedLatent, IReadOnlyList<Tensor> encoderLayers, float timestep,
        int hPacked, int wPacked)
    {
        if (encoderLayers.Count != _config.SelectedEncoderLayers.Length)
            throw new ArgumentException(
                $"encoderLayers must have {_config.SelectedEncoderLayers.Length} entries (one per selected encoder layer); got {encoderLayers.Count}.",
                nameof(encoderLayers));

        int batch = (int)packedLatent.Shape[0];
        // The GPU-resident path (SliceRows chunking, contiguous seq-dim concat/split) relies on batch-1 layout;
        // the pipeline always runs CFG as two batch-1 forwards, matching Qwen-Image/Chroma.
        if (batch != 1)
            throw new ArgumentException($"LensTransformer.Forward is batch-1 only (got batch={batch}); run CFG as separate forwards.", nameof(packedLatent));
        int imgSeqLen = (int)packedLatent.Shape[1];
        int txtSeqLen = (int)encoderLayers[0].Shape[1];
        int hidden = _config.HiddenSize;
        int encDim = _config.EncoderHiddenDim;
        int numLayers = _config.SelectedEncoderLayers.Length;

        TensorShape imgTokShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgTokens = new Tensor(imgTokShape, DType.F32);
        backend.Linear(imgTokens, packedLatent, _imgInWeight!, _imgInBias);
        LensDebugDump.Dump("img_in", imgTokens);

        // Text tokens are step-invariant — served from the per-prompt cache; the block loop consumes and
        // disposes its text stream, so hand it a device copy of the cached tensor.
        Tensor cachedTxt = GetOrBuildTextTokens(backend, encoderLayers, batch, txtSeqLen, encDim, numLayers);
        TensorShape txtTokShape = new TensorShape(batch, txtSeqLen, hidden);
        Tensor txtTokens = new Tensor(txtTokShape, DType.F32);
        backend.CopyInto(txtTokens, cachedTxt);

        Tensor temb = ComputeTimestepEmbedding(backend, timestep, batch);
        LensDebugDump.Dump("time_text_embed", temb);

        int txtPositionStart = LensRope.ComputeTextPositionStart(hPacked, wPacked);

        Tensor currentImg = imgTokens;
        Tensor currentTxt = txtTokens;

        for (int i = 0; i < _config.NumLayers; i++)
        {
            (Tensor newTxt, Tensor newImg) = _blocks[i].Forward(
                backend, currentImg, currentTxt, temb, _rope,
                hPacked, wPacked, txtPositionStart);

            currentImg.Dispose();
            currentTxt.Dispose();

            currentImg = newImg;
            currentTxt = newTxt;

            LensDebugDump.Dump($"block_{i}_image", currentImg);
            LensDebugDump.Dump($"block_{i}_text", currentTxt);
        }

        currentTxt.Dispose();

        Tensor output = ApplyFinalLayer(backend, currentImg, temb, batch, imgSeqLen);
        LensDebugDump.Dump("proj_out", output);
        currentImg.Dispose();
        temb.Dispose();

        LensDebugDump.DumpOutput(output);
        return output;
    }

    /// <summary>Returns the cached text tokens for this encoder-capture set, building them on first use:
    /// per-layer RMSNorm (own learnable scale) → channel-concat along the last dim (<c>[B, S_txt, 4·2880]</c>,
    /// upstream <c>cat([txt_norm[i](encoder[i]) for i in range(4)], dim=-1)</c>) → <c>txt_in</c> Linear. All on
    /// the backend so the result stays device-resident. Keyed on the caller's list reference (cond + uncond).</summary>
    private Tensor GetOrBuildTextTokens(IBackend backend, IReadOnlyList<Tensor> encoderLayers,
        int batch, int txtSeqLen, int encDim, int numLayers)
    {
        for (int i = 0; i < _txtTokenCache.Count; i++)
            if (ReferenceEquals(_txtTokenCache[i].Key, encoderLayers))
                return _txtTokenCache[i].Tokens;

        Tensor[] normed = new Tensor[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            normed[i] = new Tensor(encoderLayers[i].Shape, DType.F32);
            backend.RmsNorm(normed[i], encoderLayers[i], _txtNormWeights[i]!, 1e-5f);
        }

        TensorShape concatShape = new TensorShape(batch, txtSeqLen, numLayers * encDim);
        Tensor concat = new Tensor(concatShape, DType.F32);
        backend.Concat(concat, normed, 2);
        for (int i = 0; i < numLayers; i++)
            normed[i].Dispose();
        LensDebugDump.Dump("txt_concat", concat);

        TensorShape txtTokShape = new TensorShape(batch, txtSeqLen, _config.HiddenSize);
        Tensor txtTokens = new Tensor(txtTokShape, DType.F32);
        backend.Linear(txtTokens, concat, _txtInWeight!, _txtInBias);
        concat.Dispose();
        LensDebugDump.Dump("txt_in", txtTokens);

        // 2 live entries (cond + uncond); a third distinct prompt evicts the oldest.
        if (_txtTokenCache.Count >= 2)
        {
            _txtTokenCache[0].Tokens.Dispose();
            _txtTokenCache.RemoveAt(0);
        }
        _txtTokenCache.Add((encoderLayers, txtTokens));
        return txtTokens;
    }

    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep, int batch)
    {
        int hidden = _config.HiddenSize;
        float scaledTimestep = timestep * 1000.0f;

        TensorShape sinShape = new TensorShape(batch, 256);
        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmbed, scaledTimestep, batch, embDim: 256);

        TensorShape hidShape = new TensorShape(batch, hidden);
        Tensor t1 = new Tensor(hidShape, DType.F32);
        backend.Linear(t1, sinEmbed, _timestepLinear1Weight!, _timestepLinear1Bias);
        sinEmbed.Dispose();

        Tensor t1Activated = new Tensor(hidShape, DType.F32);
        backend.Silu(t1Activated, t1);
        t1.Dispose();

        Tensor temb = new Tensor(hidShape, DType.F32);
        backend.Linear(temb, t1Activated, _timestepLinear2Weight!, _timestepLinear2Bias);
        t1Activated.Dispose();

        return temb;
    }

    /// <summary>AdaLN-Continuous final layer: <c>SiLU(temb) → Linear(hidden → 2*hidden) → [scale, shift]</c> → LayerNorm-no-affine + modulate → <c>proj_out</c>. <b>Lens chunks <c>[scale, shift]</c> — scale FIRST</b> (upstream: <c>scale, shift = torch.chunk(emb, 2, dim=-1)</c>), the opposite of Flux's LastLayer and of the diffusers-default <c>[shift, scale]</c> used by Qwen-Image/SD3 final layers. All ops run on the backend so the hidden stream never leaves the device (the old host modulation loop drained the stream once per forward).</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor hidden, Tensor temb, int batch, int seqLen)
    {
        int dim = _config.HiddenSize;
        int outDim = _config.PatchSize * _config.PatchSize * _config.OutChannels;
        TensorShape hidShape = new TensorShape(batch, seqLen, dim);

        TensorShape tembShape = new TensorShape(batch, dim);
        Tensor activated = new Tensor(tembShape, DType.F32);
        backend.Silu(activated, temb);

        TensorShape modParamShape = new TensorShape(batch, dim * 2);
        Tensor modParams = new Tensor(modParamShape, DType.F32);
        backend.Linear(modParams, activated, _normOutLinearWeight!, _normOutLinearBias);
        activated.Dispose();

        // B=1 on the inference path: chunk p of the flat [1, 2*dim] projection is the contiguous element
        // range [p*dim, (p+1)*dim) — exactly SliceRows' contract. Scale is chunk 0, shift chunk 1.
        TensorShape chunkShape = new TensorShape(batch, dim);
        Tensor scaleChunk = new Tensor(chunkShape, DType.F32);
        backend.SliceRows(scaleChunk, modParams, 0);
        Tensor shiftChunk = new Tensor(chunkShape, DType.F32);
        backend.SliceRows(shiftChunk, modParams, 1);
        modParams.Dispose();

        Tensor normed = new Tensor(hidShape, DType.F32);
        backend.LayerNormNoAffine(normed, hidden, 1e-6f);
        Tensor modulated = DiTUtils.Modulate(backend, normed, shiftChunk, scaleChunk, hidShape);
        normed.Dispose();
        scaleChunk.Dispose();
        shiftChunk.Dispose();
        LensDebugDump.Dump("norm_out", modulated);

        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected = new Tensor(outShape, DType.F32);
        backend.Linear(projected, modulated, _projOutWeight!, _projOutBias);
        modulated.Dispose();

        return projected;
    }

    /// <summary>Releases all tensor references and the per-prompt text-token cache.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            for (int i = 0; i < _txtTokenCache.Count; i++) _txtTokenCache[i].Tokens.Dispose();
            _txtTokenCache.Clear();
            _imgInWeight = null; _imgInBias = null;
            for (int i = 0; i < _txtNormWeights.Length; i++) _txtNormWeights[i] = null;
            _txtInWeight = null; _txtInBias = null;
            _timestepLinear1Weight = null; _timestepLinear1Bias = null;
            _timestepLinear2Weight = null; _timestepLinear2Bias = null;
            _normOutLinearWeight = null; _normOutLinearBias = null;
            _projOutWeight = null; _projOutBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
