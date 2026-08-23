using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>Qwen-Image DiT ControlNet adapter — a faithful port of diffusers' <c>QwenImageControlNetModel</c> (the InstantX architecture). It is a shallow copy of the base <c>QwenImageTransformer</c>: its own <c>img_in</c> / <c>txt_norm</c> / <c>txt_in</c> / <c>time_text_embed</c> and a reduced stack of the SAME dual-stream blocks (<see cref="QwenImageBlock"/>, identical checkpoint key layout), plus two ControlNet-specific parts:
/// <list type="bullet">
///   <item>a zero-init <c>controlnet_x_embedder</c> Linear(64→hidden) whose projection of the packed VAE-encoded control image is ADDED to the img-embedded noise tokens,</item>
///   <item>one zero-init output Linear(hidden→hidden) per block (<c>controlnet_blocks.{i}</c>) that gates each block's image stream into a residual.</item>
/// </list>
/// The residuals are consumed by <c>QwenImageTransformer.Forward</c> via <see cref="QwenImageControlNetResiduals"/> with diffusers' interval mapping. Weight tensors are owned by the <see cref="ControlNetFile"/> that produced them; this class only borrows references (same ownership model as <see cref="FluxControlNet"/>). The released checkpoints have no mode embedder — union behavior is trained in, so the control image is taken as-is.</summary>
public sealed class QwenImageControlNet : IDisposable
{
    private readonly QwenImageControlNetConfig _config;
    private readonly QwenImageBlock[] _blocks;
    private readonly QwenImageRope _rope;
    private int _disposed;

    private Tensor? _imgInWeight, _imgInBias;
    private Tensor? _controlXEmbedWeight, _controlXEmbedBias;
    private Tensor? _txtNormWeight;
    private bool _ownsTxtNormWeight;
    private Tensor? _txtInWeight, _txtInBias;
    private Tensor? _timestepLinear1Weight, _timestepLinear1Bias;
    private Tensor? _timestepLinear2Weight, _timestepLinear2Bias;
    private readonly Tensor?[] _cnBlockWeights;
    private readonly Tensor?[] _cnBlockBiases;

    /// <summary>Creates the adapter modules from configuration; call <see cref="LoadWeights"/> before Forward.</summary>
    public QwenImageControlNet(QwenImageControlNetConfig config)
    {
        _config = config;
        _blocks = new QwenImageBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++)
        {
            _blocks[i] = new QwenImageBlock(
                config.HiddenSize, config.NumHeads, config.HeadDim, config.MlpDim, config.QkNormEps);
        }
        _rope = new QwenImageRope(theta: config.RopeTheta, ropeText: config.RopeText);
        _cnBlockWeights = new Tensor?[config.Depth];
        _cnBlockBiases = new Tensor?[config.Depth];
    }

    /// <summary>The config this adapter was built from.</summary>
    public QwenImageControlNetConfig Config => _config;

    /// <summary>Loads all weights using diffusers <c>QwenImageControlNetModel</c> naming. Block keys are identical to the base transformer's (<c>transformer_blocks.{i}.*</c>); the ControlNet extras are <c>controlnet_x_embedder</c> and <c>controlnet_blocks.{i}</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _imgInWeight = weights["img_in.weight"];
        _imgInBias = weights["img_in.bias"];
        _controlXEmbedWeight = weights["controlnet_x_embedder.weight"];
        _controlXEmbedBias = weights["controlnet_x_embedder.bias"];

        // RmsNorm needs an F32 scale vector (base-transformer parity); own the cast copy when one is made.
        Tensor txtNorm = weights["txt_norm.weight"];
        _ownsTxtNormWeight = txtNorm.DType != DType.F32;
        _txtNormWeight = _ownsTxtNormWeight ? txtNorm.CastTo(DType.F32) : txtNorm;

        _txtInWeight = weights["txt_in.weight"];
        _txtInBias = weights["txt_in.bias"];

        _timestepLinear1Weight = weights["time_text_embed.timestep_embedder.linear_1.weight"];
        _timestepLinear1Bias = weights["time_text_embed.timestep_embedder.linear_1.bias"];
        _timestepLinear2Weight = weights["time_text_embed.timestep_embedder.linear_2.weight"];
        _timestepLinear2Bias = weights["time_text_embed.timestep_embedder.linear_2.bias"];

        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"transformer_blocks.{i}");

        for (int i = 0; i < _cnBlockWeights.Length; i++)
        {
            _cnBlockWeights[i] = weights[$"controlnet_blocks.{i}.weight"];
            _cnBlockBiases[i] = weights[$"controlnet_blocks.{i}.bias"];
        }
    }

    /// <summary>Runs the ControlNet for one denoising step and returns the per-block residuals. Matches the reference pipeline's calling convention: run ONCE per step with the CONDITIONAL text stream; the same residuals feed both CFG passes of the base transformer.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="packedLatent">Packed noisy latent <c>[1, imgSeqLen, 64]</c> — the SAME tensor the base transformer sees this step.</param>
    /// <param name="packedControl">Packed VAE-encoded control image <c>[1, imgSeqLen, 64]</c> (built once per generation via the pipeline's PackLatent).</param>
    /// <param name="encoderHidden">Conditional Qwen2.5-VL hidden states <c>[1, txtSeqLen, contextDim]</c>.</param>
    /// <param name="timestep">Normalized timestep in [0, 1] (the transformer's <c>t / 1000</c>).</param>
    /// <param name="hPacked">Packed image grid height (latent_h / 2).</param>
    /// <param name="wPacked">Packed image grid width (latent_w / 2).</param>
    /// <param name="conditioningScale">Multiplier applied to every residual (diffusers <c>controlnet_conditioning_scale</c>).</param>
    /// <returns>Residuals for the base transformer; caller owns and disposes (see <see cref="QwenImageControlNetResiduals.DisposeAll"/>).</returns>
    public QwenImageControlNetResiduals Forward(
        IBackend backend,
        Tensor packedLatent,
        Tensor packedControl,
        Tensor encoderHidden,
        float timestep,
        int hPacked,
        int wPacked,
        float conditioningScale = 1.0f)
    {
        ThrowIfDisposed();
        if (packedLatent.Shape[0] != 1)
            throw new HartsyInferenceException($"QwenImageControlNet.Forward requires batch 1; got {packedLatent.Shape}.");
        if (!packedLatent.Shape.Equals(packedControl.Shape))
        {
            throw new HartsyInferenceException(
                $"QwenImageControlNet packed control shape {packedControl.Shape} must match the packed latent {packedLatent.Shape} " +
                "(same resolution, VAE-encoded and 2×2-packed).");
        }

        int imgSeqLen = (int)packedLatent.Shape[1];
        int txtSeqLen = (int)encoderHidden.Shape[1];
        int hidden = _config.HiddenSize;
        TensorShape imgShape = new TensorShape(1, imgSeqLen, hidden);

        // ── 1. img_in(noise) + controlnet_x_embedder(control) ──
        Tensor imgTokens = new Tensor(imgShape, DType.F32);
        backend.Linear(imgTokens, packedLatent, _imgInWeight!, _imgInBias);
        Tensor ctrlTokens = new Tensor(imgShape, DType.F32);
        backend.Linear(ctrlTokens, packedControl, _controlXEmbedWeight!, _controlXEmbedBias);
        Tensor img = new Tensor(imgShape, DType.F32);
        backend.Add(img, imgTokens, ctrlTokens);
        imgTokens.Dispose();
        ctrlTokens.Dispose();

        // ── 2. Timestep embedding — same sinusoidal+MLP shape as the base transformer ──
        Tensor temb = ComputeTimestepEmbedding(backend, timestep);

        // ── 3. Text stream: RMSNorm then project (base-transformer txt_norm → txt_in) ──
        Tensor txtNormed = new Tensor(encoderHidden.Shape, DType.F32);
        backend.RmsNorm(txtNormed, encoderHidden, _txtNormWeight!, 1e-6f);
        Tensor txt = new Tensor(new TensorShape(1, txtSeqLen, hidden), DType.F32);
        backend.Linear(txt, txtNormed, _txtInWeight!, _txtInBias);
        txtNormed.Dispose();

        // ── 4. Dual-stream blocks; each block's image stream feeds its zero-init output Linear ──
        int txtPositionStart = QwenImageRope.ComputeTextPositionStart(hPacked, wPacked);
        Tensor[] residuals = new Tensor[_blocks.Length];
        for (int i = 0; i < _blocks.Length; i++)
        {
            (Tensor newImg, Tensor newTxt) = _blocks[i].Forward(
                backend, img, txt, temb, _rope, hPacked, wPacked, txtPositionStart);
            img.Dispose();
            txt.Dispose();
            img = newImg;
            txt = newTxt;
            residuals[i] = FluxControlNet.ProjectResidual(backend, img, _cnBlockWeights[i]!, _cnBlockBiases[i], conditioningScale, imgShape);
        }

        img.Dispose();
        txt.Dispose();
        temb.Dispose();

        return new QwenImageControlNetResiduals { BlockResiduals = residuals };
    }

    /// <summary>Yields all weight tensors for GPU preloading / freeing.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_imgInWeight is not null) yield return _imgInWeight;
        if (_imgInBias is not null) yield return _imgInBias;
        if (_controlXEmbedWeight is not null) yield return _controlXEmbedWeight;
        if (_controlXEmbedBias is not null) yield return _controlXEmbedBias;
        if (_txtNormWeight is not null) yield return _txtNormWeight;
        if (_txtInWeight is not null) yield return _txtInWeight;
        if (_txtInBias is not null) yield return _txtInBias;
        if (_timestepLinear1Weight is not null) yield return _timestepLinear1Weight;
        if (_timestepLinear1Bias is not null) yield return _timestepLinear1Bias;
        if (_timestepLinear2Weight is not null) yield return _timestepLinear2Weight;
        if (_timestepLinear2Bias is not null) yield return _timestepLinear2Bias;
        for (int i = 0; i < _blocks.Length; i++)
        {
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
        }
        foreach (Tensor? w in _cnBlockWeights) if (w is not null) yield return w;
        foreach (Tensor? b in _cnBlockBiases) if (b is not null) yield return b;
    }

    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep)
    {
        int hidden = _config.HiddenSize;
        Tensor sinEmbed = new Tensor(new TensorShape(1, 256), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinEmbed, timestep * 1000.0f, 1, embDim: 256);

        TensorShape hidShape = new TensorShape(1, hidden);
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

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Releases borrowed weight references (and the owned F32 txt_norm cast copy). The underlying weight tensors belong to the safetensors loader that produced them.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _imgInWeight = null; _imgInBias = null;
            _controlXEmbedWeight = null; _controlXEmbedBias = null;
            if (_ownsTxtNormWeight) _txtNormWeight?.Dispose();
            _txtNormWeight = null;
            _txtInWeight = null; _txtInBias = null;
            _timestepLinear1Weight = null; _timestepLinear1Bias = null;
            _timestepLinear2Weight = null; _timestepLinear2Bias = null;
            Array.Clear(_cnBlockWeights);
            Array.Clear(_cnBlockBiases);
        }
        GC.SuppressFinalize(this);
    }
}
