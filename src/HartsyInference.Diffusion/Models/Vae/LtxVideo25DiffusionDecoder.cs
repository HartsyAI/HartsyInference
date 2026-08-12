using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>LTX-2.5 video VAE decoder (<c>NADiffusionDecoder</c> / the reference's <c>DiffusionVideoDecoder</c>): the
/// convolutional LTX-2 decoder is replaced by a neighborhood-attention transformer that decodes by denoising.
///
/// <para>Stages 1-4 deterministically upsample the un-normalized latent into a stage-5 <em>context</em> volume — NA
/// blocks at 2048/1024/512/512 channels, each followed by a linear pixel-shuffle upsample. Stage 5 then runs
/// <c>diff_blocks</c> over patchified noised pixels, conditioned on that context through a per-block projection plus
/// shared AdaLN scale/shift. The shipped checkpoint is single-step <c>x0</c>: one pass over stage 5 <em>is</em> the
/// image, so there is no Euler loop to run.</para>
///
/// <para>Geometry: temporal ×8 and spatial ×32 (×8 from the upsamples, ×4 from the patch). Output frames are
/// <c>8·T−7</c>, not <c>8·(T−1)+1</c>, because three of the four upsamples drop the frame their temporal
/// pixel-shuffle duplicates. Two latent frames are replicated past the end before stage 1 and the resulting 16 output
/// frames cropped after stage 4, so the real last frames get a symmetric attention window.</para>
///
/// <para>Noise is a parameter rather than a seed: the reference fixes <c>torch.manual_seed(0)</c>, but C# and torch
/// RNGs differ, so a caller comparing against it must inject the same tensor. <c>decoder.type_emb</c> exists in the
/// checkpoint and is read by neither reference implementation, so it is deliberately not loaded.</para></summary>
public sealed unsafe class LtxVideo25DiffusionDecoder : IDisposable
{
    private readonly LtxVideo25DiffusionDecoderConfig _config;
    private readonly float[]? _latentsMean;
    private readonly float[]? _latentsStd;

    private LtxVideo25WeightScope? _scope;
    private LtxVideo25NaBlock[][] _detStages = [];
    private LtxVideo25PixelShuffleUpsample[] _upsamples = [];
    private LtxVideo25NaBlock[] _diffBlocks = [];
    private Tensor? _convInWeight, _convInBias, _convInXtWeight, _convInXtBias;
    private Tensor? _tEmbWeight0, _tEmbBias0, _tEmbWeight2, _tEmbBias2;
    private Tensor? _adaLnWeight, _adaLnBias, _normOutWeight, _convOutWeight, _convOutBias;

    public LtxVideo25DiffusionDecoder(LtxVideo25DiffusionDecoderConfig? config = null,
        float[]? latentsMean = null, float[]? latentsStd = null)
    {
        _config = config ?? new LtxVideo25DiffusionDecoderConfig();
        _latentsMean = latentsMean;
        _latentsStd = latentsStd;
        if (_config.StageChannels.Length != _config.StageDepths.Length || _config.StageChannels.Length != _config.Upsamples.Length + 1)
            throw new ArgumentException("stage_channels, stage_depths and upsamples must describe the same stage count.", nameof(config));
    }

    public LtxVideo25DiffusionDecoderConfig Config => _config;

    /// <summary>Checkpoint keys read by the last <see cref="LoadWeights"/>; the caller can diff this against the
    /// bucket to catch a key the decoder silently ignores.</summary>
    public IReadOnlyCollection<string> ConsumedKeys => _scope?.Consumed ?? (IReadOnlyCollection<string>)Array.Empty<string>();

    /// <summary>Output frames for a latent frame count (<c>8t−7</c> for the shipped config).</summary>
    public int OutputFrames(int latentFrames) => _config.OutputFrames(latentFrames);

    /// <summary>Shape of the noise <see cref="Decode"/> expects for a latent <c>[1, C, t, h, w]</c>.</summary>
    public TensorShape NoiseShape(int latentFrames, int latentHeight, int latentWidth) => new TensorShape(
        [1, _config.OutChannels, OutputFrames(latentFrames),
         (long)latentHeight * _config.SpatialUpscale, (long)latentWidth * _config.SpatialUpscale]);

    /// <summary>Loads the whole <c>decoder.*</c> subtree. Keys keep their checkpoint prefix, matching the
    /// <c>VaeDiffusionDecoder</c> bucket produced by <c>LtxVideo2CheckpointConverter</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        LtxVideo25WeightScope scope = new LtxVideo25WeightScope(weights);
        _scope = scope;

        _convInWeight = scope.Raw("decoder.conv_in.weight");
        _convInBias = scope.OptionalF32("decoder.conv_in.bias");
        if (_convInWeight.Shape[0] != _config.StageChannels[0] || _convInWeight.Shape[1] != _config.InChannels)
            throw new InvalidOperationException($"'decoder.conv_in.weight' is {_convInWeight.Shape}, expected [{_config.StageChannels[0]}, {_config.InChannels}].");

        int stageCount = _config.Upsamples.Length;
        _detStages = new LtxVideo25NaBlock[stageCount][];
        _upsamples = new LtxVideo25PixelShuffleUpsample[stageCount];
        for (int stage = 0; stage < stageCount; stage++)
        {
            int channels = _config.StageChannels[stage];
            _detStages[stage] = new LtxVideo25NaBlock[_config.StageDepths[stage]];
            for (int block = 0; block < _detStages[stage].Length; block++)
            {
                _detStages[stage][block] = new LtxVideo25NaBlock(channels, _config.StageKernels[stage], _config, isDiffusion: false);
                _detStages[stage][block].LoadWeights(scope, $"decoder.det_stages.{stage}.{block}");
            }
            ((int T, int H, int W) stride, int reduction) = _config.Upsamples[stage];
            _upsamples[stage] = new LtxVideo25PixelShuffleUpsample(channels, stride, reduction);
            _upsamples[stage].LoadWeights(scope, $"decoder.upsamples.{stage}");
            if (_upsamples[stage].OutChannels != _config.StageChannels[stage + 1])
                throw new InvalidOperationException($"upsample {stage} yields {_upsamples[stage].OutChannels} channels, but stage {stage + 1} expects {_config.StageChannels[stage + 1]}.");
        }

        _tEmbWeight0 = scope.Raw("decoder.t_embedder.mlp.0.weight");
        _tEmbBias0 = scope.OptionalF32("decoder.t_embedder.mlp.0.bias");
        _tEmbWeight2 = scope.Raw("decoder.t_embedder.mlp.2.weight");
        _tEmbBias2 = scope.OptionalF32("decoder.t_embedder.mlp.2.bias");
        if (_tEmbWeight0.Shape[1] != _config.TimestepFreqDim || _tEmbWeight0.Shape[0] != _config.TimestepEmbedDim)
            throw new InvalidOperationException($"'decoder.t_embedder.mlp.0.weight' is {_tEmbWeight0.Shape}, expected [{_config.TimestepEmbedDim}, {_config.TimestepFreqDim}].");

        int stage5 = _config.StageChannels[^1];
        int packedChannels = _config.OutChannels * _config.PatchSize * _config.PatchSize;
        _convInXtWeight = scope.Raw("decoder.conv_in_x_t.weight");
        _convInXtBias = scope.OptionalF32("decoder.conv_in_x_t.bias");
        if (_convInXtWeight.Shape[0] != stage5 || _convInXtWeight.Shape[1] != packedChannels)
            throw new InvalidOperationException($"'decoder.conv_in_x_t.weight' is {_convInXtWeight.Shape}, expected [{stage5}, {packedChannels}].");

        _adaLnWeight = scope.Raw("decoder.shared_adaln.proj.weight");
        _adaLnBias = scope.OptionalF32("decoder.shared_adaln.proj.bias");

        _diffBlocks = new LtxVideo25NaBlock[_config.StageDepths[^1]];
        for (int block = 0; block < _diffBlocks.Length; block++)
        {
            _diffBlocks[block] = new LtxVideo25NaBlock(stage5, _config.Stage5Kernel, _config, isDiffusion: true);
            _diffBlocks[block].LoadWeights(scope, $"decoder.diff_blocks.{block}");
        }

        _normOutWeight = scope.F32("decoder.norm_out.weight");
        _convOutWeight = scope.Raw("decoder.conv_out.weight");
        _convOutBias = scope.OptionalF32("decoder.conv_out.bias");
        if (_convOutWeight.Shape[0] != packedChannels || _convOutWeight.Shape[1] != stage5)
            throw new InvalidOperationException($"'decoder.conv_out.weight' is {_convOutWeight.Shape}, expected [{packedChannels}, {stage5}].");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _convInWeight, _convInBias, _convInXtWeight, _convInXtBias, _tEmbWeight0,
                                      _tEmbBias0, _tEmbWeight2, _tEmbBias2, _adaLnWeight, _adaLnBias,
                                      _normOutWeight, _convOutWeight, _convOutBias })
        {
            if (t is not null) yield return t;
        }
        foreach (LtxVideo25NaBlock[] stage in _detStages)
            foreach (LtxVideo25NaBlock block in stage)
                foreach (Tensor t in block.EnumerateWeights()) yield return t;
        foreach (LtxVideo25PixelShuffleUpsample upsample in _upsamples)
            foreach (Tensor t in upsample.EnumerateWeights()) yield return t;
        foreach (LtxVideo25NaBlock block in _diffBlocks)
            foreach (Tensor t in block.EnumerateWeights()) yield return t;
    }

    /// <summary>Stages 1-4: latent <c>[1, C, t, h, w]</c> → context tokens <c>[(8t−7)·8h·8w, stage5]</c>, with the
    /// trailing replicated frames already cropped off.</summary>
    public Tensor EncodeContext(IBackend backend, Tensor latent, out int frames, out int height, out int width)
    {
        ValidateLatent(latent);
        int latentFrames = (int)latent.Shape[2], latentHeight = (int)latent.Shape[3], latentWidth = (int)latent.Shape[4];
        int pad = _config.TrailingPadLatentFrames;

        Tensor tokens = LatentToTokens(latent, latentFrames, latentHeight, latentWidth, pad);
        int currentT = latentFrames + pad, currentH = latentHeight, currentW = latentWidth;

        Tensor x = new Tensor(new TensorShape(tokens.Shape[0], _config.StageChannels[0]), DType.F32);
        backend.Linear(x, tokens, _convInWeight!, _convInBias);
        tokens.Dispose();

        for (int stage = 0; stage < _detStages.Length; stage++)
        {
            foreach (LtxVideo25NaBlock block in _detStages[stage])
                block.Forward(backend, x, null, null, currentT, currentH, currentW);
            Tensor next = _upsamples[stage].Forward(backend, x, currentT, currentH, currentW,
                dropLeadingFrame: true, out currentT, out currentH, out currentW);
            x.Dispose();
            x = next;
        }

        int keep = currentT - pad * _config.TemporalUpscale;
        if (keep <= 0)
            throw new InvalidOperationException($"Latent of {latentFrames} frames leaves {keep} context frames after the trailing-pad crop.");
        frames = keep;
        height = currentH;
        width = currentW;
        if (keep == currentT) return x;

        long channels = x.Shape[1];
        Tensor cropped = new Tensor(new TensorShape((long)keep * currentH * currentW, channels), DType.F32);
        long bytes = cropped.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)cropped.DataPointer, bytes, bytes);
        x.Dispose();
        return cropped;
    }

    /// <summary>Decodes a latent <c>[1, C, t, h, w]</c> to RGB <c>[1, 3, 8t−7, 32h, 32w]</c>. <paramref name="noise"/>
    /// must match <see cref="NoiseShape"/>; the caller owns it.</summary>
    public Tensor Decode(IBackend backend, Tensor latent, Tensor noise)
    {
        ArgumentNullException.ThrowIfNull(noise);
        if (_config.NumInferenceSteps != 1)
        {
            throw new NotSupportedException(
                "Only the shipped single-step x0 schedule is implemented; a multi-step checkpoint would need the "
                + "reference's Euler update (velocity = (x_t − x0)/t, x_t −= (t − t_next)·velocity) around this call.");
        }
        using Tensor context = EncodeContext(backend, latent, out int frames, out int height, out int width);
        TensorShape expected = NoiseShape((int)latent.Shape[2], (int)latent.Shape[3], (int)latent.Shape[4]);
        if (!noise.Shape.Equals(expected))
            throw new ArgumentException($"noise is {noise.Shape}, expected {expected}.", nameof(noise));
        return DiffusionStep(backend, context, noise, frames, height, width, timestep: 1f);
    }

    /// <summary>One stage-5 pass: patchified noised pixels + context + AdaLN(t) → predicted pixels. With
    /// <c>model_output_type = "x0"</c> and a single step this output is the decode itself.</summary>
    private Tensor DiffusionStep(IBackend backend, Tensor context, Tensor noise, int frames, int height, int width, float timestep)
    {
        int stage5 = _config.StageChannels[^1];
        long tokens = (long)frames * height * width;

        Tensor x = new Tensor(new TensorShape(tokens, stage5), DType.F32);
        using (Tensor patched = PatchifyPixels(noise, frames, height, width))
            backend.Linear(x, patched, _convInXtWeight!, _convInXtBias);

        using (Tensor modulation = Modulation(backend, timestep, stage5))
        {
            foreach (LtxVideo25NaBlock block in _diffBlocks)
                block.Forward(backend, x, context, modulation, frames, height, width);
        }

        int packedChannels = _config.OutChannels * _config.PatchSize * _config.PatchSize;
        using Tensor normed = new Tensor(new TensorShape(tokens, stage5), DType.F32);
        backend.RmsNorm(normed, x, _normOutWeight!, _config.NormEps);
        x.Dispose();
        using Tensor packed = new Tensor(new TensorShape(tokens, packedChannels), DType.F32);
        backend.Linear(packed, normed, _convOutWeight!, _convOutBias);
        return UnpatchifyPixels(packed, frames, height, width);
    }

    /// <summary>Sinusoidal embedding of <c>scale·t</c> → <c>t_embedder</c> MLP → SiLU → <c>shared_adaln.proj</c>,
    /// yielding the 7 AdaLN chunks as <c>[7, stage5]</c>.</summary>
    private Tensor Modulation(IBackend backend, float timestep, int stage5)
    {
        using Tensor frequencies = new Tensor(new TensorShape(1, _config.TimestepFreqDim), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(frequencies, _config.TimestepScaleMultiplier * timestep, batch: 1,
            embDim: _config.TimestepFreqDim, maxPeriod: _config.TimestepMaxPeriod);
        using Tensor hidden = new Tensor(new TensorShape(1, _config.TimestepEmbedDim), DType.F32);
        backend.Linear(hidden, frequencies, _tEmbWeight0!, _tEmbBias0);
        backend.Silu(hidden, hidden);
        using Tensor embedding = new Tensor(new TensorShape(1, _config.TimestepEmbedDim), DType.F32);
        backend.Linear(embedding, hidden, _tEmbWeight2!, _tEmbBias2);
        backend.Silu(embedding, embedding);
        Tensor modulation = new Tensor(new TensorShape(7, stage5), DType.F32);
        using Tensor flat = modulation.Reshape(new TensorShape(1, 7L * stage5));
        backend.Linear(flat, embedding, _adaLnWeight!, _adaLnBias);
        return modulation;
    }

    private void ValidateLatent(Tensor latent)
    {
        ArgumentNullException.ThrowIfNull(latent);
        if (latent.Shape.Rank != 5)
            throw new ArgumentException($"latent must be [batch, channels, t, h, w]; got {latent.Shape}.", nameof(latent));
        if (latent.Shape[0] != 1)
            throw new ArgumentException($"only batch 1 is supported; got {latent.Shape[0]}.", nameof(latent));
        if (latent.Shape[1] != _config.InChannels)
            throw new ArgumentException($"latent channels {latent.Shape[1]} != {_config.InChannels}.", nameof(latent));
    }

    /// <summary>Un-normalizes, replicates the last frame <paramref name="pad"/> times and transposes to channels-last
    /// tokens in one pass — a separate 5-D denormalized copy of a 2048-channel-bound latent is pure waste.</summary>
    private Tensor LatentToTokens(Tensor latent, int frames, int height, int width, int pad)
    {
        Tensor source = latent.DType == DType.F32 ? latent : latent.CastTo(DType.F32);
        int channels = _config.InChannels;
        int paddedFrames = frames + pad;
        Tensor tokens = new Tensor(new TensorShape((long)paddedFrames * height * width, channels), DType.F32);
        float* src = (float*)source.DataPointer, dst = (float*)tokens.DataPointer;
        long plane = (long)height * width;
        for (int frame = 0; frame < paddedFrames; frame++)
        {
            int sourceFrame = Math.Min(frame, frames - 1);
            for (long position = 0; position < plane; position++)
            {
                long row = frame * plane + position;
                for (int channel = 0; channel < channels; channel++)
                {
                    float value = src[(channel * (long)frames + sourceFrame) * plane + position];
                    if (_latentsMean is not null && _latentsStd is not null)
                        value = value * _latentsStd[channel] + _latentsMean[channel];
                    dst[row * channels + channel] = value;
                }
            }
        }
        if (!ReferenceEquals(source, latent)) source.Dispose();
        return tokens;
    }

    /// <summary>Pixels <c>[1, 3, t, h·p, w·p]</c> → channels-last tokens <c>[t·h·w, 3·p²]</c>. The packed channel is
    /// <c>c·p² + r·p + q</c> with <c>q</c> the height sub-index and <c>r</c> the width sub-index — the reference's
    /// <c>(c p r q)</c> ordering, which puts width outside height. <see cref="IBackend.UnpatchifyVae"/> uses the same
    /// packing but a channels-first layout, so it cannot be reused here.</summary>
    private Tensor PatchifyPixels(Tensor pixels, int frames, int height, int width)
    {
        int patch = _config.PatchSize, channels = _config.OutChannels;
        int packedChannels = channels * patch * patch;
        int pixelHeight = height * patch, pixelWidth = width * patch;
        Tensor source = pixels.DType == DType.F32 ? pixels : pixels.CastTo(DType.F32);
        Tensor tokens = new Tensor(new TensorShape((long)frames * height * width, packedChannels), DType.F32);
        float* src = (float*)source.DataPointer, dst = (float*)tokens.DataPointer;
        for (int c = 0; c < channels; c++)
        for (int t = 0; t < frames; t++)
        for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
        {
            long row = ((long)t * height + h) * width + w;
            for (int q = 0; q < patch; q++)
            for (int r = 0; r < patch; r++)
            {
                long pixel = (((long)c * frames + t) * pixelHeight + h * patch + q) * pixelWidth + w * patch + r;
                dst[row * packedChannels + c * patch * patch + r * patch + q] = src[pixel];
            }
        }
        if (!ReferenceEquals(source, pixels)) source.Dispose();
        return tokens;
    }

    /// <summary>Inverse of <see cref="PatchifyPixels"/>: tokens <c>[t·h·w, 3·p²]</c> → pixels <c>[1, 3, t, h·p, w·p]</c>.</summary>
    private Tensor UnpatchifyPixels(Tensor tokens, int frames, int height, int width)
    {
        int patch = _config.PatchSize, channels = _config.OutChannels;
        int packedChannels = channels * patch * patch;
        int pixelHeight = height * patch, pixelWidth = width * patch;
        Tensor pixels = new Tensor(new TensorShape([1, channels, frames, pixelHeight, pixelWidth]), DType.F32);
        float* src = (float*)tokens.DataPointer, dst = (float*)pixels.DataPointer;
        for (int c = 0; c < channels; c++)
        for (int t = 0; t < frames; t++)
        for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
        {
            long row = ((long)t * height + h) * width + w;
            for (int q = 0; q < patch; q++)
            for (int r = 0; r < patch; r++)
            {
                long pixel = (((long)c * frames + t) * pixelHeight + h * patch + q) * pixelWidth + w * patch + r;
                dst[pixel] = src[row * packedChannels + c * patch * patch + r * patch + q];
            }
        }
        return pixels;
    }

    public void Dispose()
    {
        if (_scope is null) return;
        foreach (Tensor t in _scope.Owned) t.Dispose();
        _scope = null;
    }
}
