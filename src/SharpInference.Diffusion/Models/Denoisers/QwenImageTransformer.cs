using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Qwen-Image MMDiT transformer (<c>QwenImageTransformer2DModel</c>). Processes packed image patch tokens and Qwen2.5-VL text embeddings through 60 dual-stream transformer blocks with QK-norm, AdaLN-Zero modulation, and 3-axis (frame, height, width) RoPE applied separately to each stream before joint attention. Top-level layout follows <c>diffusers/models/transformers/transformer_qwenimage.py</c>: <c>img_in</c> Linear → <c>txt_norm</c> RMSNorm → <c>txt_in</c> Linear → <c>time_text_embed</c> sinusoidal+MLP → 60 × <see cref="QwenImageBlock"/> → <c>norm_out</c> AdaLN-continuous → <c>proj_out</c> Linear. Outputs predicted velocity in packed token form <c>[B, imgSeqLen, patch_size² * out_channels]</c>; pipeline unpacks back to <c>[B, C, H, W]</c>.</summary>
public sealed unsafe class QwenImageTransformer : IDisposable
{
    private readonly QwenImageConfig _config;
    private readonly QwenImageBlock[] _blocks;
    private readonly QwenImageRope _rope;
    private int _disposed;

    private Tensor? _imgInWeight, _imgInBias;

    private Tensor? _txtNormWeight;

    private Tensor? _txtInWeight, _txtInBias;

    private Tensor? _timestepLinear1Weight, _timestepLinear1Bias;
    private Tensor? _timestepLinear2Weight, _timestepLinear2Bias;

    private Tensor? _normOutLinearWeight, _normOutLinearBias;
    private Tensor? _projOutWeight, _projOutBias;

    /// <summary>Creates a Qwen-Image transformer from configuration.</summary>
    public QwenImageTransformer(QwenImageConfig config)
    {
        _config = config;
        int mlpDim = (int)(config.HiddenSize * config.MlpRatio);

        _blocks = new QwenImageBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++)
        {
            _blocks[i] = new QwenImageBlock(
                config.HiddenSize,
                config.NumHeads,
                config.HeadDim,
                mlpDim,
                config.QkNormEps);
        }

        _rope = new QwenImageRope(theta: config.RopeTheta);
    }

    /// <summary>Loads all transformer weights from named tensors using diffusers naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _imgInWeight = weights["img_in.weight"];
        _imgInBias = weights["img_in.bias"];

        _txtNormWeight = CastToF32IfNeeded(weights["txt_norm.weight"]);

        _txtInWeight = weights["txt_in.weight"];
        _txtInBias = weights["txt_in.bias"];

        _timestepLinear1Weight = weights["time_text_embed.timestep_embedder.linear_1.weight"];
        _timestepLinear1Bias = weights["time_text_embed.timestep_embedder.linear_1.bias"];
        _timestepLinear2Weight = weights["time_text_embed.timestep_embedder.linear_2.weight"];
        _timestepLinear2Bias = weights["time_text_embed.timestep_embedder.linear_2.bias"];

        for (int i = 0; i < _config.Depth; i++)
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
        if (_txtNormWeight is not null) yield return _txtNormWeight;
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
    /// <param name="packedLatent">Packed latent tokens [B, imgSeqLen, patch_size² * in_channels].</param>
    /// <param name="encoderHidden">Qwen2.5-VL encoder hidden states [B, txtSeqLen, encoderDim].</param>
    /// <param name="timestep">Normalized timestep in [0, 1] (diffusers passes <c>t / 1000</c>).</param>
    /// <param name="hPacked">Packed-grid height (<c>latent_h / patch_size</c>).</param>
    /// <param name="wPacked">Packed-grid width (<c>latent_w / patch_size</c>).</param>
    public Tensor Forward(IBackend backend, Tensor packedLatent, Tensor encoderHidden, float timestep,
        int hPacked, int wPacked)
    {
        int batch = (int)packedLatent.Shape[0];
        int imgSeqLen = (int)packedLatent.Shape[1];
        int txtSeqLen = (int)encoderHidden.Shape[1];
        int hidden = _config.HiddenSize;

        TensorShape imgTokShape = new TensorShape(batch, imgSeqLen, hidden);
        Tensor imgTokens = new Tensor(imgTokShape, DType.F32);
        backend.Linear(imgTokens, packedLatent, _imgInWeight!, _imgInBias);
        QwenImageDebugDump.Dump("img_in", imgTokens);

        TensorShape txtNormShape = encoderHidden.Shape;
        Tensor txtNormed = new Tensor(txtNormShape, DType.F32);
        backend.RmsNorm(txtNormed, encoderHidden, _txtNormWeight!, 1e-6f);

        TensorShape txtTokShape = new TensorShape(batch, txtSeqLen, hidden);
        Tensor txtTokens = new Tensor(txtTokShape, DType.F32);
        backend.Linear(txtTokens, txtNormed, _txtInWeight!, _txtInBias);
        txtNormed.Dispose();
        QwenImageDebugDump.Dump("txt_in", txtTokens);

        Tensor temb = ComputeTimestepEmbedding(backend, timestep, batch);
        QwenImageDebugDump.Dump("time_text_embed", temb);

        int txtPositionStart = QwenImageRope.ComputeTextPositionStart(hPacked, wPacked);

        Tensor currentImg = imgTokens;
        Tensor currentTxt = txtTokens;

        for (int i = 0; i < _config.Depth; i++)
        {
            (Tensor newImg, Tensor newTxt) = _blocks[i].Forward(
                backend, currentImg, currentTxt, temb, _rope,
                hPacked, wPacked, txtPositionStart);

            currentImg.Dispose();
            currentTxt.Dispose();

            currentImg = newImg;
            currentTxt = newTxt;

            QwenImageDebugDump.Dump($"block_{i}_image", currentImg);
            QwenImageDebugDump.Dump($"block_{i}_text", currentTxt);
        }

        currentTxt.Dispose();

        Tensor output = ApplyFinalLayer(backend, currentImg, temb, batch, imgSeqLen);
        QwenImageDebugDump.Dump("proj_out", output);
        currentImg.Dispose();
        temb.Dispose();

        QwenImageDebugDump.DumpOutput(output);
        return output;
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

    /// <summary>Final AdaLN-continuous: <c>SiLU(temb) → Linear → [shift, scale]</c> then unparameterized LayerNorm + modulate + <c>proj_out</c>. Diffusers' AdaLayerNormContinuous chunks <c>[shift, scale]</c> in that order — matches the Linear output layout used by SD3's final layer.</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor hidden, Tensor temb, int batch, int seqLen)
    {
        int dim = _config.HiddenSize;
        int outDim = _config.PatchSize * _config.PatchSize * _config.InChannels;
        TensorShape hidShape = new TensorShape(batch, seqLen, dim);

        TensorShape tembShape = new TensorShape(batch, dim);
        Tensor activated = new Tensor(tembShape, DType.F32);
        backend.Silu(activated, temb);

        TensorShape modParamShape = new TensorShape(batch, dim * 2);
        Tensor modParams = new Tensor(modParamShape, DType.F32);
        backend.Linear(modParams, activated, _normOutLinearWeight!, _normOutLinearBias);
        activated.Dispose();

        Tensor normed = new Tensor(hidShape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, hidden, batch, seqLen, dim);

        Tensor modulated = new Tensor(hidShape, DType.F32);
        float* modPtr = (float*)modParams.DataPointer;
        float* normPtr = (float*)normed.DataPointer;
        float* outModPtr = (float*)modulated.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int modBase = b * dim * 2;
            for (int s = 0; s < seqLen; s++)
            {
                int vecOffset = (b * seqLen + s) * dim;
                for (int d = 0; d < dim; d++)
                {
                    float shift = modPtr[modBase + d];
                    float scale = modPtr[modBase + dim + d];
                    outModPtr[vecOffset + d] = normPtr[vecOffset + d] * (1.0f + scale) + shift;
                }
            }
        }
        normed.Dispose();
        modParams.Dispose();
        QwenImageDebugDump.Dump("norm_out", modulated);

        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor projected = new Tensor(outShape, DType.F32);
        backend.Linear(projected, modulated, _projOutWeight!, _projOutBias);
        modulated.Dispose();

        return projected;
    }

    /// <summary>Pipeline-level debug hook: dumps the post-denoise, pre-VAE latent under <c>$QWEN_IMAGE_DEBUG_DIR/final_latent.bin</c> when the env var is set. Used by the layer-by-layer diff harness to capture pipeline state.</summary>
    public static void DumpFinalLatent(Tensor latent) => QwenImageDebugDump.Dump("final_latent", latent);

    private static Tensor CastToF32IfNeeded(Tensor t) =>
        t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    /// <summary>Releases all tensor references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _imgInWeight = null; _imgInBias = null;
            _txtNormWeight = null;
            _txtInWeight = null; _txtInBias = null;
            _timestepLinear1Weight = null; _timestepLinear1Bias = null;
            _timestepLinear2Weight = null; _timestepLinear2Bias = null;
            _normOutLinearWeight = null; _normOutLinearBias = null;
            _projOutWeight = null; _projOutBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
