using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>Flux DiT ControlNet adapter — a faithful port of diffusers' <c>FluxControlNetModel</c> (the
/// InstantX / Shakker-Labs architecture). It is a shallow copy of the base <see cref="FluxTransformer"/>:
/// its own <c>x_embedder</c> / <c>context_embedder</c> / <c>time_text_embed</c> and a reduced stack of the SAME
/// double/single-stream blocks (<see cref="FluxDoubleStreamBlock"/> / <see cref="FluxSingleStreamBlock"/>,
/// identical checkpoint key layout), plus three ControlNet-specific parts:
/// <list type="bullet">
/// <item>a zero-init <c>controlnet_x_embedder</c> Linear(64→hidden) whose projection of the packed VAE-encoded
/// control image is ADDED to the x-embedded noise tokens,</item>
/// <item>one zero-init output Linear(hidden→hidden) per block (<c>controlnet_blocks.{i}</c> /
/// <c>controlnet_single_blocks.{i}</c>) that gates each block's image stream into a residual,</item>
/// <item>for union checkpoints, a <c>controlnet_mode_embedder</c> whose selected row is prepended to the
/// projected text tokens as an extra zero-position token (diffusers union mapping — see
/// <see cref="FluxUnionControlMode"/>).</item>
/// </list>
/// The residuals are consumed by <see cref="FluxTransformer.Forward"/> via <see cref="FluxControlNetResiduals"/>
/// with diffusers' interval mapping. Weight tensors are owned by the <see cref="ControlNetFile"/> that produced
/// them; this class only borrows references (same ownership model as <see cref="ControlNet"/>).</summary>
public sealed unsafe class FluxControlNet : IDisposable
{
    private readonly FluxControlNetConfig _config;
    private readonly FluxDoubleStreamBlock[] _doubleBlocks;
    private readonly FluxSingleStreamBlock[] _singleBlocks;
    private readonly FluxRope _rope;
    private long _ropeSig = long.MinValue;
    private int _disposed;

    private Tensor? _xEmbedWeight, _xEmbedBias;
    private Tensor? _controlXEmbedWeight, _controlXEmbedBias;
    private Tensor? _contextEmbedWeight, _contextEmbedBias;

    private Tensor? _timestepLinear1Weight, _timestepLinear1Bias;
    private Tensor? _timestepLinear2Weight, _timestepLinear2Bias;
    private Tensor? _textLinear1Weight, _textLinear1Bias;
    private Tensor? _textLinear2Weight, _textLinear2Bias;
    private Tensor? _guidanceLinear1Weight, _guidanceLinear1Bias;
    private Tensor? _guidanceLinear2Weight, _guidanceLinear2Bias;

    private readonly Tensor?[] _cnBlockWeights;
    private readonly Tensor?[] _cnBlockBiases;
    private readonly Tensor?[] _cnSingleBlockWeights;
    private readonly Tensor?[] _cnSingleBlockBiases;

    /// <summary>Union mode embedding table cast to F32 for the host row gather. Owned by this instance.</summary>
    private Tensor? _modeEmbedTableF32;

    /// <summary>Creates the adapter modules from configuration; call <see cref="LoadWeights"/> before Forward.</summary>
    public FluxControlNet(FluxControlNetConfig config)
    {
        _config = config;
        int mlpDim = (int)(config.HiddenSize * config.MlpRatio);

        _doubleBlocks = new FluxDoubleStreamBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++)
        {
            _doubleBlocks[i] = new FluxDoubleStreamBlock(
                config.HiddenSize, config.NumHeads, mlpDim, config.QkvBias, config.QkNormEps);
        }

        _singleBlocks = new FluxSingleStreamBlock[config.DepthSingleBlocks];
        for (int i = 0; i < config.DepthSingleBlocks; i++)
        {
            _singleBlocks[i] = new FluxSingleStreamBlock(
                config.HiddenSize, config.NumHeads, mlpDim, config.QkNormEps);
        }

        _rope = new FluxRope(config.AxesDim, config.Theta);
        _cnBlockWeights = new Tensor?[config.Depth];
        _cnBlockBiases = new Tensor?[config.Depth];
        _cnSingleBlockWeights = new Tensor?[config.DepthSingleBlocks];
        _cnSingleBlockBiases = new Tensor?[config.DepthSingleBlocks];
    }

    /// <summary>The config this adapter was built from.</summary>
    public FluxControlNetConfig Config => _config;

    /// <summary>Loads all weights using diffusers <c>FluxControlNetModel</c> naming. Block keys are identical to
    /// the base transformer's (<c>transformer_blocks.{i}.*</c> / <c>single_transformer_blocks.{i}.*</c>); the
    /// ControlNet extras are <c>controlnet_x_embedder</c>, <c>controlnet_blocks.{i}</c>,
    /// <c>controlnet_single_blocks.{i}</c>, and (union only) <c>controlnet_mode_embedder.weight</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _xEmbedWeight = weights["x_embedder.weight"];
        _xEmbedBias = weights["x_embedder.bias"];
        _controlXEmbedWeight = weights["controlnet_x_embedder.weight"];
        _controlXEmbedBias = weights["controlnet_x_embedder.bias"];
        _contextEmbedWeight = weights["context_embedder.weight"];
        _contextEmbedBias = weights["context_embedder.bias"];

        _timestepLinear1Weight = weights["time_text_embed.timestep_embedder.linear_1.weight"];
        _timestepLinear1Bias = weights["time_text_embed.timestep_embedder.linear_1.bias"];
        _timestepLinear2Weight = weights["time_text_embed.timestep_embedder.linear_2.weight"];
        _timestepLinear2Bias = weights["time_text_embed.timestep_embedder.linear_2.bias"];
        _textLinear1Weight = weights["time_text_embed.text_embedder.linear_1.weight"];
        _textLinear1Bias = weights["time_text_embed.text_embedder.linear_1.bias"];
        _textLinear2Weight = weights["time_text_embed.text_embedder.linear_2.weight"];
        _textLinear2Bias = weights["time_text_embed.text_embedder.linear_2.bias"];
        if (_config.GuidanceEmbed)
        {
            _guidanceLinear1Weight = weights["time_text_embed.guidance_embedder.linear_1.weight"];
            _guidanceLinear1Bias = weights["time_text_embed.guidance_embedder.linear_1.bias"];
            _guidanceLinear2Weight = weights["time_text_embed.guidance_embedder.linear_2.weight"];
            _guidanceLinear2Bias = weights["time_text_embed.guidance_embedder.linear_2.bias"];
        }

        for (int i = 0; i < _doubleBlocks.Length; i++)
            _doubleBlocks[i].LoadWeights(weights, $"transformer_blocks.{i}");
        for (int i = 0; i < _singleBlocks.Length; i++)
            _singleBlocks[i].LoadWeights(weights, $"single_transformer_blocks.{i}");

        for (int i = 0; i < _cnBlockWeights.Length; i++)
        {
            _cnBlockWeights[i] = weights[$"controlnet_blocks.{i}.weight"];
            _cnBlockBiases[i] = weights[$"controlnet_blocks.{i}.bias"];
        }
        for (int i = 0; i < _cnSingleBlockWeights.Length; i++)
        {
            _cnSingleBlockWeights[i] = weights[$"controlnet_single_blocks.{i}.weight"];
            _cnSingleBlockBiases[i] = weights[$"controlnet_single_blocks.{i}.bias"];
        }

        if (_config.IsUnion)
        {
            Tensor table = weights["controlnet_mode_embedder.weight"];
            _modeEmbedTableF32?.Dispose();
            // Host row gather needs F32; the table is tiny ([numMode, hidden]) so an owned cast copy is cheap.
            _modeEmbedTableF32 = table.DType == DType.F32 ? table.To(table.Device) : table.CastTo(DType.F32);
        }
    }

    /// <summary>Runs the ControlNet for one denoising step and returns the per-block residuals.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="packedLatent">Packed noisy latent <c>[1, imgSeqLen, 64]</c> — the SAME tensor the base transformer sees this step.</param>
    /// <param name="packedControl">Packed VAE-encoded control image <c>[1, imgSeqLen, 64]</c> (built once per generation via the pipeline's PackLatent).</param>
    /// <param name="sigma">Current sigma in [0, 1] (the transformer's timestep · 1/1000).</param>
    /// <param name="t5Embeddings">T5 text embeddings <c>[1, txtSeqLen, contextInDim]</c>.</param>
    /// <param name="clipPooled">CLIP-L pooled embedding <c>[1, pooledDim]</c>.</param>
    /// <param name="guidanceScale">Embedded distilled guidance (Dev); ignored when the checkpoint has no guidance embedder.</param>
    /// <param name="hPacked">Packed image grid height (latent_h / 2).</param>
    /// <param name="wPacked">Packed image grid width (latent_w / 2).</param>
    /// <param name="unionModeId">Union checkpoints: the <see cref="FluxUnionControlMode"/> row id. Must be null for non-union checkpoints.</param>
    /// <param name="conditioningScale">Multiplier applied to every residual (diffusers <c>conditioning_scale</c>).</param>
    /// <returns>Residuals for the base transformer; caller owns and disposes (see <see cref="FluxControlNetResiduals.DisposeAll"/>).</returns>
    public FluxControlNetResiduals Forward(
        IBackend backend,
        Tensor packedLatent,
        Tensor packedControl,
        float sigma,
        Tensor t5Embeddings,
        Tensor clipPooled,
        float guidanceScale,
        int hPacked,
        int wPacked,
        int? unionModeId = null,
        float conditioningScale = 1.0f)
    {
        ThrowIfDisposed();
        if (packedLatent.Shape[0] != 1)
            throw new HartsyInferenceException($"FluxControlNet.Forward requires batch 1; got {packedLatent.Shape}.");
        if (!packedLatent.Shape.Equals(packedControl.Shape))
        {
            throw new HartsyInferenceException(
                $"FluxControlNet packed control shape {packedControl.Shape} must match the packed latent {packedLatent.Shape} " +
                "(same resolution, VAE-encoded and 2×2-packed).");
        }
        if (_config.IsUnion && unionModeId is null)
            throw new HartsyInferenceException("This Flux ControlNet is a union checkpoint — a control mode id is required (see FluxUnionControlMode).");
        if (!_config.IsUnion && unionModeId is not null)
            throw new HartsyInferenceException("Control mode id supplied but this Flux ControlNet has no mode embedder (single-mode or Union-Pro-2.0 checkpoint).");
        if (unionModeId is int m && (m < 0 || m >= _config.NumMode))
            throw new HartsyInferenceException($"Union control mode id {m} out of range [0, {_config.NumMode}).");

        int imgSeqLen = (int)packedLatent.Shape[1];
        int txtSeqLen = (int)t5Embeddings.Shape[1];
        int hidden = _config.HiddenSize;

        // ── 1. x_embedder(noise) + controlnet_x_embedder(control) ──
        TensorShape imgShape = new TensorShape(1, imgSeqLen, hidden);
        Tensor imgTokens = new Tensor(imgShape, DType.F32);
        backend.Linear(imgTokens, packedLatent, _xEmbedWeight!, _xEmbedBias);
        Tensor ctrlTokens = new Tensor(imgShape, DType.F32);
        backend.Linear(ctrlTokens, packedControl, _controlXEmbedWeight!, _controlXEmbedBias);
        Tensor img = new Tensor(imgShape, DType.F32);
        backend.Add(img, imgTokens, ctrlTokens);
        imgTokens.Dispose();
        ctrlTokens.Dispose();

        // ── 2. Timestep + pooled-text (+ guidance) embedding — same MLPs as the base transformer ──
        Tensor temb = ComputeTemb(backend, sigma, clipPooled, guidanceScale);

        // ── 3. Project text tokens; union prepends the mode-embedding row as an extra zero-position token ──
        Tensor txt = new Tensor(new TensorShape(1, txtSeqLen, hidden), DType.F32);
        backend.Linear(txt, t5Embeddings, _contextEmbedWeight!, _contextEmbedBias);
        int condTxtLen = txtSeqLen;
        if (unionModeId is int modeId)
        {
            Tensor modeRow = GatherModeRow(modeId);
            Tensor extended = new Tensor(new TensorShape(1, txtSeqLen + 1, hidden), DType.F32);
            backend.Concat(extended, new Tensor[] { modeRow, txt }, 1);
            modeRow.Dispose();
            txt.Dispose();
            txt = extended;
            condTxtLen = txtSeqLen + 1;
        }

        // ── 4. RoPE tables (sig-cached like the base transformer; the mode token rides position 0 like text) ──
        EnsureRope(condTxtLen, hPacked, wPacked);

        // ── 5. Double-stream blocks; each block's image stream feeds its zero-init output Linear ──
        Tensor[] doubleResiduals = new Tensor[_doubleBlocks.Length];
        for (int i = 0; i < _doubleBlocks.Length; i++)
        {
            (Tensor newImg, Tensor newTxt) = _doubleBlocks[i].Forward(backend, img, txt, temb, _rope);
            img.Dispose();
            txt.Dispose();
            img = newImg;
            txt = newTxt;
            doubleResiduals[i] = ProjectResidual(backend, img, _cnBlockWeights[i]!, _cnBlockBiases[i], conditioningScale, imgShape);
        }

        // ── 6. Single-stream blocks on the concatenated [txt | img] sequence; residuals are the image rows ──
        Tensor[] singleResiduals = new Tensor[_singleBlocks.Length];
        if (_singleBlocks.Length > 0)
        {
            TensorShape concatShape = new TensorShape(1, condTxtLen + imgSeqLen, hidden);
            Tensor x = new Tensor(concatShape, DType.F32);
            backend.Concat(x, new Tensor[] { txt, img }, 1);
            for (int i = 0; i < _singleBlocks.Length; i++)
            {
                Tensor newX = _singleBlocks[i].Forward(backend, x, temb, _rope);
                x.Dispose();
                x = newX;
                Tensor imgRows = new Tensor(imgShape, DType.F32);
                backend.SliceRows(imgRows, x, condTxtLen);
                singleResiduals[i] = ProjectResidual(backend, imgRows, _cnSingleBlockWeights[i]!, _cnSingleBlockBiases[i], conditioningScale, imgShape);
                imgRows.Dispose();
            }
            x.Dispose();
        }

        img.Dispose();
        txt.Dispose();
        temb.Dispose();

        return new FluxControlNetResiduals
        {
            DoubleBlockResiduals = doubleResiduals,
            SingleBlockResiduals = singleResiduals,
        };
    }

    /// <summary>Yields all weight tensors for GPU preloading / freeing.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_xEmbedWeight is not null) yield return _xEmbedWeight;
        if (_xEmbedBias is not null) yield return _xEmbedBias;
        if (_controlXEmbedWeight is not null) yield return _controlXEmbedWeight;
        if (_controlXEmbedBias is not null) yield return _controlXEmbedBias;
        if (_contextEmbedWeight is not null) yield return _contextEmbedWeight;
        if (_contextEmbedBias is not null) yield return _contextEmbedBias;
        if (_timestepLinear1Weight is not null) yield return _timestepLinear1Weight;
        if (_timestepLinear1Bias is not null) yield return _timestepLinear1Bias;
        if (_timestepLinear2Weight is not null) yield return _timestepLinear2Weight;
        if (_timestepLinear2Bias is not null) yield return _timestepLinear2Bias;
        if (_textLinear1Weight is not null) yield return _textLinear1Weight;
        if (_textLinear1Bias is not null) yield return _textLinear1Bias;
        if (_textLinear2Weight is not null) yield return _textLinear2Weight;
        if (_textLinear2Bias is not null) yield return _textLinear2Bias;
        if (_guidanceLinear1Weight is not null) yield return _guidanceLinear1Weight;
        if (_guidanceLinear1Bias is not null) yield return _guidanceLinear1Bias;
        if (_guidanceLinear2Weight is not null) yield return _guidanceLinear2Weight;
        if (_guidanceLinear2Bias is not null) yield return _guidanceLinear2Bias;
        for (int i = 0; i < _doubleBlocks.Length; i++)
        {
            foreach (Tensor w in _doubleBlocks[i].EnumerateWeights()) yield return w;
        }
        for (int i = 0; i < _singleBlocks.Length; i++)
        {
            foreach (Tensor w in _singleBlocks[i].EnumerateWeights()) yield return w;
        }
        foreach (Tensor? w in _cnBlockWeights) if (w is not null) yield return w;
        foreach (Tensor? b in _cnBlockBiases) if (b is not null) yield return b;
        foreach (Tensor? w in _cnSingleBlockWeights) if (w is not null) yield return w;
        foreach (Tensor? b in _cnSingleBlockBiases) if (b is not null) yield return b;
    }

    /// <summary>Applies the per-block zero-init output Linear and the conditioning scale.</summary>
    private static Tensor ProjectResidual(IBackend backend, Tensor blockImg, Tensor weight, Tensor? bias,
        float conditioningScale, TensorShape imgShape)
    {
        Tensor residual = new Tensor(imgShape, DType.F32);
        backend.Linear(residual, blockImg, weight, bias);
        if (MathF.Abs(conditioningScale - 1.0f) <= 1e-6f)
            return residual;
        Tensor scaled = new Tensor(imgShape, DType.F32);
        backend.Scale(scaled, residual, conditioningScale);
        residual.Dispose();
        return scaled;
    }

    /// <summary>Copies row <paramref name="modeId"/> of the F32 mode table into a <c>[1, 1, hidden]</c> token.</summary>
    private Tensor GatherModeRow(int modeId)
    {
        int hidden = _config.HiddenSize;
        Tensor row = new Tensor(new TensorShape(1, 1, hidden), DType.F32);
        ReadOnlySpan<float> table = _modeEmbedTableF32!.AsReadOnlySpan<float>();
        table.Slice(modeId * hidden, hidden).CopyTo(row.AsSpan<float>());
        return row;
    }

    private void EnsureRope(int txtSeqLen, int hPacked, int wPacked)
    {
        long sig = ((long)txtSeqLen << 32) ^ ((long)hPacked << 16) ^ (long)wPacked ^ 0x3C963C96;
        if (sig == _ropeSig)
            return;
        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen, hPacked, wPacked);
        _rope.Precompute(posIds);
        posIds.Dispose();
        _ropeSig = sig;
    }

    private Tensor ComputeTemb(IBackend backend, float sigma, Tensor clipPooled, float guidanceScale)
    {
        int hidden = _config.HiddenSize;
        TensorShape sinShape = new TensorShape(1, 256);
        TensorShape hidShape = new TensorShape(1, hidden);

        Tensor sinEmbed = new Tensor(sinShape, DType.F32);
        FluxTransformer.ComputeSinusoidalTimestep(sinEmbed, sigma * 1000.0f, 1);
        Tensor temb = MlpSiluMlp(backend, sinEmbed, hidShape,
            _timestepLinear1Weight!, _timestepLinear1Bias, _timestepLinear2Weight!, _timestepLinear2Bias);
        sinEmbed.Dispose();

        Tensor pooled = MlpSiluMlp(backend, clipPooled, hidShape,
            _textLinear1Weight!, _textLinear1Bias, _textLinear2Weight!, _textLinear2Bias);
        Tensor sum = new Tensor(hidShape, DType.F32);
        backend.Add(sum, temb, pooled);
        temb.Dispose();
        pooled.Dispose();
        temb = sum;

        if (_config.GuidanceEmbed && _guidanceLinear1Weight is not null)
        {
            Tensor guidanceSin = new Tensor(sinShape, DType.F32);
            FluxTransformer.ComputeSinusoidalTimestep(guidanceSin, guidanceScale * 1000.0f, 1);
            Tensor gEmb = MlpSiluMlp(backend, guidanceSin, hidShape,
                _guidanceLinear1Weight!, _guidanceLinear1Bias, _guidanceLinear2Weight!, _guidanceLinear2Bias);
            guidanceSin.Dispose();
            Tensor withGuidance = new Tensor(hidShape, DType.F32);
            backend.Add(withGuidance, temb, gEmb);
            temb.Dispose();
            gEmb.Dispose();
            temb = withGuidance;
        }

        return temb;
    }

    private static Tensor MlpSiluMlp(IBackend backend, Tensor input, TensorShape hidShape,
        Tensor w1, Tensor? b1, Tensor w2, Tensor? b2)
    {
        Tensor h = new Tensor(hidShape, DType.F32);
        backend.Linear(h, input, w1, b1);
        Tensor act = new Tensor(hidShape, DType.F32);
        backend.Silu(act, h);
        h.Dispose();
        Tensor output = new Tensor(hidShape, DType.F32);
        backend.Linear(output, act, w2, b2);
        act.Dispose();
        return output;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Releases borrowed weight references and the owned F32 mode table. The underlying weight tensors
    /// belong to the safetensors loader that produced them.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _xEmbedWeight = null; _xEmbedBias = null;
            _controlXEmbedWeight = null; _controlXEmbedBias = null;
            _contextEmbedWeight = null; _contextEmbedBias = null;
            _timestepLinear1Weight = null; _timestepLinear1Bias = null;
            _timestepLinear2Weight = null; _timestepLinear2Bias = null;
            _textLinear1Weight = null; _textLinear1Bias = null;
            _textLinear2Weight = null; _textLinear2Bias = null;
            _guidanceLinear1Weight = null; _guidanceLinear1Bias = null;
            _guidanceLinear2Weight = null; _guidanceLinear2Bias = null;
            Array.Clear(_cnBlockWeights);
            Array.Clear(_cnBlockBiases);
            Array.Clear(_cnSingleBlockWeights);
            Array.Clear(_cnSingleBlockBiases);
            _modeEmbedTableF32?.Dispose();
            _modeEmbedTableF32 = null;
        }
        GC.SuppressFinalize(this);
    }
}
