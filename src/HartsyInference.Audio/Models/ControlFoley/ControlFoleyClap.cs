using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>ControlFoley's reference-audio embedding (<c>FeaturesUtils.encode_audio_with_clap</c>): laion_clap's HTS-AT audio
/// tower. The clip is repeat-padded to 10 s, turned into a 64-bin log-mel, batch-normalised, stretched to a 256 x 256 Swin
/// image, encoded by four Swin stages, mean-pooled and sent through <c>audio_projection</c> and an L2 normalisation.</summary>
public sealed class ControlFoleyClap
{
    private const float BatchNormEps = 1e-5f;
    private const float NormEps = 1e-5f;

    private readonly ControlFoleyClapConfig _config;
    private readonly ControlFoleyClapFrontend _frontend;
    private ControlFoleySwinStage[]? _stages;
    private Tensor? _bnScale, _bnShift, _patchW, _patchB, _patchNormW, _patchNormB, _normW, _normB;
    private Tensor? _proj1W, _proj1B, _proj2W, _proj2B;

    /// <summary>Creates an unloaded tower; call <see cref="LoadWeights"/> before embedding.</summary>
    public ControlFoleyClap(ControlFoleyClapConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Depths.Count != config.Heads.Count || config.Depths.Count == 0)
        {
            throw new ArgumentException("Depths and Heads must be non-empty and the same length.", nameof(config));
        }

        if (config.SpecSize % config.MelBins != 0 || config.SpecSize % (config.PatchSize << (config.Depths.Count - 1)) != 0)
        {
            throw new ArgumentException("SpecSize must be a multiple of MelBins and of the final patch stride.", nameof(config));
        }

        _config = config;
        _frontend = new ControlFoleyClapFrontend(config);
    }

    /// <summary>Geometry this instance was built for.</summary>
    public ControlFoleyClapConfig Config => _config;

    /// <summary>Binds <c>audio_branch.*</c> and <c>audio_projection.*</c> of a laion_clap state dict; the STFT and mel bank
    /// buffers, classification heads and text tower are not read.</summary>
    public unsafe void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        IReadOnlyDictionary<string, Tensor> w = weights;
        ControlFoleyClapConfig c = _config;
        int e = c.EmbedDim, mels = c.MelBins;

        Tensor bnWeight = Require(w, "audio_branch.bn0.weight", mels), bnBias = Require(w, "audio_branch.bn0.bias", mels);
        Tensor mean = Require(w, "audio_branch.bn0.running_mean", mels), variance = Require(w, "audio_branch.bn0.running_var", mels);
        _bnScale = new Tensor(new TensorShape(mels), DType.F32);
        _bnShift = new Tensor(new TensorShape(mels), DType.F32);
        float* bw = (float*)bnWeight.DataPointer, bb = (float*)bnBias.DataPointer;
        float* bm = (float*)mean.DataPointer, bv = (float*)variance.DataPointer;
        float* scale = (float*)_bnScale.DataPointer, shift = (float*)_bnShift.DataPointer;
        for (int i = 0; i < mels; i++)
        {
            scale[i] = bw[i] / MathF.Sqrt(bv[i] + BatchNormEps);
            shift[i] = bb[i] - bm[i] * scale[i];
        }

        int patch = c.PatchSize;
        _patchW = Require(w, "audio_branch.patch_embed.proj.weight", e, 1, patch, patch);
        _patchB = Require(w, "audio_branch.patch_embed.proj.bias", e);
        _patchNormW = Require(w, "audio_branch.patch_embed.norm.weight", e);
        _patchNormB = Require(w, "audio_branch.patch_embed.norm.bias", e);

        int grid = c.SpecSize / patch;
        _stages = new ControlFoleySwinStage[c.Depths.Count];
        for (int i = 0; i < _stages.Length; i++)
        {
            _stages[i] = new ControlFoleySwinStage(w, $"audio_branch.layers.{i}", e << i, c.Heads[i], c.Depths[i], grid >> i,
                c.WindowSize, downsample: i < _stages.Length - 1);
        }

        _normW = Require(w, "audio_branch.norm.weight", c.FinalDim);
        _normB = Require(w, "audio_branch.norm.bias", c.FinalDim);
        _proj1W = Require(w, "audio_projection.0.weight", c.JointDim, c.FinalDim);
        _proj1B = Require(w, "audio_projection.0.bias", c.JointDim);
        _proj2W = Require(w, "audio_projection.2.weight", c.JointDim, c.JointDim);
        _proj2B = Require(w, "audio_projection.2.bias", c.JointDim);
    }

    /// <summary>Embeds a mono clip at any rate the caller chose (ControlFoley passes 16 kHz) into the L2-normalised
    /// <see cref="ControlFoleyClapConfig.JointDim"/>-vector the network consumes as <c>audio_f</c>.</summary>
    public float[] Embed(IBackend backend, ReadOnlySpan<float> audio)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (_stages is null)
        {
            throw new InvalidOperationException("Call LoadWeights before embedding.");
        }

        float[] logMel = _frontend.LogMel(_frontend.ConditionWaveform(audio));
        return EmbedLogMel(backend, logMel);
    }

    /// <summary>The conditioned log-mel <c>[frames, MelBins]</c> of a clip, before batch normalisation.</summary>
    internal float[] LogMel(ReadOnlySpan<float> audio) => _frontend.LogMel(_frontend.ConditionWaveform(audio));

    private unsafe float[] EmbedLogMel(IBackend backend, float[] logMel)
    {
        ControlFoleyClapConfig c = _config;
        int mels = c.MelBins, frames = logMel.Length / mels;
        float* scale = (float*)_bnScale!.DataPointer, shift = (float*)_bnShift!.DataPointer;
        float[] normalised = new float[logMel.Length];
        for (int t = 0; t < frames; t++)
        {
            for (int m = 0; m < mels; m++)
            {
                normalised[t * mels + m] = logMel[t * mels + m] * scale[m] + shift[m];
            }
        }

        float[] image = ToImage(StretchTime(normalised, frames, mels, c.SpecSize * (c.SpecSize / mels)), c);
        int patch = c.PatchSize, grid = c.SpecSize / patch, tokens = grid * grid, e = c.EmbedDim;
        float[] patches = new float[tokens * patch * patch];
        for (int gy = 0; gy < grid; gy++)
        {
            for (int gx = 0; gx < grid; gx++)
            {
                for (int py = 0; py < patch; py++)
                {
                    Array.Copy(image, (gy * patch + py) * c.SpecSize + gx * patch, patches, (gy * grid + gx) * patch * patch + py * patch, patch);
                }
            }
        }

        using Tensor patchWeight = _patchW!.Reshape(new TensorShape(e, patch * patch));
        float[] x = DacOps.Linear(backend, patches, patchWeight, tokens, patch * patch, e, _patchB);
        x = ControlFoleyClipTransformer.LayerNorm(backend, x, _patchNormW!, _patchNormB!, tokens, e, NormEps);
        foreach (ControlFoleySwinStage stage in _stages!)
        {
            x = stage.Forward(backend, x);
        }

        int finalTokens = x.Length / c.FinalDim;
        x = ControlFoleyClipTransformer.LayerNorm(backend, x, _normW!, _normB!, finalTokens, c.FinalDim, NormEps);
        float[] pooled = new float[c.FinalDim];
        for (int t = 0; t < finalTokens; t++)
        {
            for (int d = 0; d < c.FinalDim; d++)
            {
                pooled[d] += x[t * c.FinalDim + d];
            }
        }

        for (int d = 0; d < pooled.Length; d++)
        {
            pooled[d] /= finalTokens;
        }

        float[] hidden = DacOps.Linear(backend, pooled, _proj1W!, 1, c.FinalDim, c.JointDim, _proj1B);
        for (int i = 0; i < hidden.Length; i++)
        {
            hidden[i] = MathF.Max(hidden[i], 0f);
        }

        float[] embedding = DacOps.Linear(backend, hidden, _proj2W!, 1, c.JointDim, c.JointDim, _proj2B);
        double norm = 0.0;
        foreach (float v in embedding)
        {
            norm += (double)v * v;
        }

        float inv = 1f / MathF.Max((float)Math.Sqrt(norm), 1e-12f);
        for (int i = 0; i < embedding.Length; i++)
        {
            embedding[i] *= inv;
        }

        return embedding;
    }

    /// <summary><c>F.interpolate(mode='bicubic', align_corners=True)</c> of <c>[frames, mels]</c> along time up to
    /// <paramref name="target"/> frames (<c>reshape_wav2img</c>); a longer clip is rejected as the official assert does.</summary>
    internal static float[] StretchTime(float[] x, int frames, int mels, int target)
    {
        if (frames > target)
        {
            throw new ArgumentException($"Log-mel has {frames} frames, more than the {target} the Swin image holds.", nameof(x));
        }

        if (frames == target)
        {
            return x;
        }

        const float A = -0.75f;
        float[] result = new float[target * mels];
        float ratio = (float)(frames - 1) / (target - 1);
        for (int i = 0; i < target; i++)
        {
            float src = ratio * i;
            int i0 = (int)MathF.Floor(src);
            float t = src - i0;
            float[] wgt =
            [
                ((A * (t + 1f) - 5f * A) * (t + 1f) + 8f * A) * (t + 1f) - 4f * A,
                ((A + 2f) * t - (A + 3f)) * t * t + 1f,
                ((A + 2f) * (1f - t) - (A + 3f)) * (1f - t) * (1f - t) + 1f,
                ((A * (2f - t) - 5f * A) * (2f - t) + 8f * A) * (2f - t) - 4f * A,
            ];
            for (int m = 0; m < mels; m++)
            {
                float acc = 0f;
                for (int k = 0; k < 4; k++)
                {
                    int row = Math.Clamp(i0 - 1 + k, 0, frames - 1);
                    acc += wgt[k] * x[row * mels + m];
                }

                result[i * mels + m] = acc;
            }
        }

        return result;
    }

    /// <summary>The <c>reshape_wav2img</c> fold of a <c>[time, mels]</c> map into the square Swin image: time is cut into
    /// <c>SpecSize / MelBins</c> chunks that stack along the frequency axis.</summary>
    private static float[] ToImage(float[] stretched, ControlFoleyClapConfig c)
    {
        int mels = c.MelBins, size = c.SpecSize, chunks = size / mels;
        float[] image = new float[size * size];
        for (int q = 0; q < chunks; q++)
        {
            for (int f = 0; f < mels; f++)
            {
                for (int col = 0; col < size; col++)
                {
                    image[(q * mels + f) * size + col] = stretched[(q * size + col) * mels + f];
                }
            }
        }

        return image;
    }

    /// <summary>Fetches <paramref name="key"/> as F32 after checking its shape.</summary>
    internal static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape)
    {
        if (!weights.TryGetValue(key, out Tensor? tensor))
        {
            throw new KeyNotFoundException($"ControlFoley CLAP checkpoint is missing '{key}'.");
        }

        if (tensor.Shape.Rank != shape.Length)
        {
            throw new InvalidDataException($"'{key}' has rank {tensor.Shape.Rank}, expected {shape.Length}.");
        }

        for (int d = 0; d < shape.Length; d++)
        {
            if (tensor.Shape[d] != shape[d])
            {
                throw new InvalidDataException($"'{key}' dimension {d} is {tensor.Shape[d]}, expected {shape[d]}.");
            }
        }

        return WhisperOps.EnsureF32(tensor);
    }
}
