using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>The avatar's conditioning modes: <c>At2v</c> (audio to video, no conditioning frames), <c>Ai2v</c> (one image frame first), and
/// <c>Avc</c> (continuation: conditioning frames, one of them the reference latent at <see cref="LongCatAvatarRun.RefImgIndex"/>).</summary>
public enum LongCatAvatarMode { At2v, Ai2v, Avc }

/// <summary>One avatar denoising run. <see cref="AudioTokens"/> holds the audio tokens of every latent frame,
/// <c>[Frames·contextTokens, outputDim]</c> (the DiT's audio projection output, see <see cref="LongCatTransformer.ProjectAudio"/>).
/// <see cref="Kind"/> picks the reference's sigma path; the scheduler's static <see cref="Shift"/> then applies to it (the shipped 1.6 scheduler
/// config has shift 7).</summary>
public sealed record LongCatAvatarRun(
    int Channels, int Frames, int Height, int Width, float[] Noise,
    LongCatPrompt Positive, LongCatPrompt Negative, float[] AudioTokens, float[]? UnconditionalAudioTokens,
    int Steps, LongCatAvatarSchedule.Kind Kind, float Shift,
    float TextGuidance, float AudioGuidance,
    LongCatAvatarMode Mode, int NumCondLatents = 0, float[]? CondLatents = null,
    int RefImgIndex = 10, bool UseKvCache = true);

/// <summary>The sigmas (shifted, with the terminal zero), the timesteps (sigma·1000), and the final latents.</summary>
public sealed record LongCatAvatarResult(float[] Latents, float[] Sigmas, float[] Timesteps);

/// <summary>The avatar denoising loop (<c>generate_at2v</c>, <c>generate_ai2v</c> and <c>generate_avc</c> in
/// <c>pipeline_longcat_video_avatar.py</c> @ 58eb336), for one segment in latent space. The guidance is the reference's three-pass
/// form: <c>uncond + text·(cond − uncondText) + audio·(uncondText − uncond)</c>, where <c>cond</c> is the positive prompt with the audio,
/// <c>uncondText</c> the negative prompt with the audio, and <c>uncond</c> the negative prompt with zero audio. When both scales are 1 (the
/// distilled 1.5 path) one pass suffices. With <see cref="LongCatAvatarRun.UseKvCache"/> the conditioning frames go through the cache pass once,
/// and each step runs only the noisy frames. The conditioning frames are never stepped.</summary>
public sealed class LongCatAvatarPipeline
{
    private readonly IBackend _backend;
    private readonly LongCatTransformer _dit;

    /// <summary>Creates the pipeline over a loaded avatar transformer.</summary>
    public LongCatAvatarPipeline(IBackend backend, LongCatTransformer dit)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(dit);
        _backend = backend;
        _dit = dit;
    }

    /// <summary>Runs the denoising loop of one segment.</summary>
    public LongCatAvatarResult Denoise(LongCatAvatarRun run)
    {
        ArgumentNullException.ThrowIfNull(run);
        Validate(run);
        int channels = run.Channels, frames = run.Frames, hw = run.Height * run.Width;
        int cond = run.Mode == LongCatAvatarMode.At2v ? 0 : run.NumCondLatents;
        int noisy = frames - cond, ctx = _dit.Config.AudioContextTokens, audioWidth = _dit.Config.AudioOutputDim;
        bool useCache = run.Mode == LongCatAvatarMode.Avc && run.UseKvCache && cond > 0;
        bool guided = run.TextGuidance > 1f || run.AudioGuidance > 1f;

        // The reference's sigma path, then the scheduler's static shift, then the terminal zero.
        float[] sigmas = Shifted(LongCatAvatarSchedule.Sigmas(run.Steps, run.Kind), run.Shift);
        float[] timesteps = new float[run.Steps];
        for (int i = 0; i < run.Steps; i++)
            timesteps[i] = sigmas[i] * LongCatAvatarSchedule.NumTimesteps;
        float[] sigmasWithZero = [.. sigmas, 0f];

        LongCatConditioning conditioning = run.Mode switch
        {
            LongCatAvatarMode.At2v => new LongCatConditioning(),
            LongCatAvatarMode.Ai2v => new LongCatConditioning(NumCondLatents: 1),
            _ => LongCatConditioning.Avatar(cond, run.RefImgIndex),
        };

        float[] latents = (float[])run.Noise.Clone();
        if (cond > 0)
            WriteFrames(latents, run.CondLatents!, channels, frames, hw, cond);

        LongCatKvCache? cache = null;
        Tensor? condLatent = null;
        try
        {
            if (useCache)
            {
                condLatent = Frames(latents, channels, frames, hw, 0, cond, run.Height, run.Width);
                cache = _dit.BuildKvCache(_backend, condLatent, conditioning);
            }

            for (int i = 0; i < run.Steps; i++)
            {
                float t = timesteps[i], dt = sigmasWithZero[i + 1] - sigmasWithZero[i];
                float[] velocity = useCache
                    ? Velocity(run, cache!, latents, channels, frames, hw, cond, conditioning, t, guided)
                    : FullVelocity(run, latents, channels, frames, hw, cond, conditioning, t, guided);

                // The reference negates the model output for its scheduler and steps the noisy frames only.
                for (int c = 0; c < channels; c++)
                    for (int f = 0; f < noisy; f++)
                        for (int e = 0; e < hw; e++)
                        {
                            int index = (c * frames + cond + f) * hw + e;
                            int vIndex = (c * noisy + f) * hw + e;
                            latents[index] += dt * -velocity[vIndex];
                        }
            }
        }
        finally
        {
            cache?.Dispose();
            condLatent?.Dispose();
        }

        // The cached path steps only the noisy frames; the cond frames stay as they were written.
        return new LongCatAvatarResult(latents, sigmasWithZero, timesteps);
    }

    /// <summary>The velocity over the noisy frames with the cache: <c>[channels, noisy, hw]</c>.</summary>
    private float[] Velocity(LongCatAvatarRun run, LongCatKvCache cache, float[] latents, int channels, int frames, int hw, int cond,
        LongCatConditioning conditioning, float t, bool guided)
    {
        int noisy = frames - cond;
        float[] noisyLatents = LastFrames(latents, channels, frames, hw, noisy);
        float[] noisyAudio = AudioFor(run, cond, noisy);
        float[] timesteps = [t];
        return Guided(run, guided, (promptIsPositive, audioOn) =>
        {
            LongCatPrompt prompt = promptIsPositive ? run.Positive : run.Negative;
            return Predict(noisyLatents, channels, noisy, run.Height, run.Width, prompt, audioOn ? noisyAudio : AudioFor(run.UnconditionalAudioTokens!, cond, noisy),
                timesteps, conditioning, cache);
        });
    }

    /// <summary>The velocity over all frames without the cache: the noisy frames' rows, <c>[channels, noisy, hw]</c>.</summary>
    private float[] FullVelocity(LongCatAvatarRun run, float[] latents, int channels, int frames, int hw, int cond,
        LongCatConditioning conditioning, float t, bool guided)
    {
        int noisy = frames - cond;
        float[] frameTimesteps = conditioning.FrameTimesteps(t, frames);
        float[] all = AudioFor(run, 0, frames);
        float[] full = Guided(run, guided, (promptIsPositive, audioOn) =>
        {
            LongCatPrompt prompt = promptIsPositive ? run.Positive : run.Negative;
            return Predict(latents, channels, frames, run.Height, run.Width, prompt, audioOn ? all : AudioFor(run.UnconditionalAudioTokens!, 0, frames), frameTimesteps, conditioning,
                cache: null);
        });
        // Keep the noisy frames' rows: [channels, frames, hw] → [channels, noisy, hw].
        float[] result = new float[channels * noisy * hw];
        int k = 0;
        for (int c = 0; c < channels; c++)
            for (int f = cond; f < frames; f++)
                for (int e = 0; e < hw; e++, k++)
                    result[k] = full[(c * frames + f) * hw + e];
        return result;
    }

    /// <summary>The three-pass guidance (or one pass when unguided). <paramref name="pass"/> takes (positive prompt, audio on) flags.
    private static float[] Guided(LongCatAvatarRun run, bool guided, Func<bool, bool, float[]> pass)
    {
        if (!guided)
            return pass(true, true);
        float[] cond = pass(true, true);
        float[] uncondText = pass(false, true);
        float[] uncond = pass(false, false);
        float[] result = new float[cond.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = uncond[i] + run.TextGuidance * (cond[i] - uncondText[i]) + run.AudioGuidance * (uncondText[i] - uncond[i]);
        return result;
    }

    private float[] Predict(float[] latents, int channels, int frames, int height, int width, LongCatPrompt prompt, float[] audio,
        float[] frameTimesteps, LongCatConditioning conditioning, LongCatKvCache? cache)
    {
        using Tensor input = new(new TensorShape([1, channels, frames, height, width]), DType.F32);
        latents.AsSpan().CopyTo(input.AsSpan<float>());
        using Tensor text = new(new TensorShape(prompt.Length, _dit.Config.CaptionChannels), DType.F32);
        prompt.Embeds.AsSpan().CopyTo(text.AsSpan<float>());
        int ctx = _dit.Config.AudioContextTokens, width2 = _dit.Config.AudioOutputDim;
        using Tensor audioTokens = new(new TensorShape(frames * ctx, width2), DType.F32);
        audio.AsSpan().CopyTo(audioTokens.AsSpan<float>());
        using Tensor output = cache is null
            ? _dit.Forward(_backend, input, text, frameTimesteps, in conditioning, prompt.ValidTokens, audioTokens)
            : _dit.ForwardCached(_backend, input, text, frameTimesteps, in conditioning, cache, prompt.ValidTokens, audioTokens);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>The audio tokens of frames <paramref name="from"/>..<paramref name="from"/>+<paramref name="count"/>.</summary>
    private float[] AudioFor(LongCatAvatarRun run, int from, int count) => AudioFor(run.AudioTokens, from, count);

    private float[] AudioFor(float[] tokens, int from, int count)
    {
        int width = _dit.Config.AudioOutputDim, ctx = _dit.Config.AudioContextTokens;
        float[] result = new float[count * ctx * width];
        Array.Copy(tokens, from * ctx * width, result, 0, result.Length);
        return result;
    }


    private static float[] Shifted(float[] sigmas, float shift)
    {
        float[] result = new float[sigmas.Length];
        for (int i = 0; i < sigmas.Length; i++)
            result[i] = shift * sigmas[i] / (1f + (shift - 1f) * sigmas[i]);
        return result;
    }

    private void Validate(LongCatAvatarRun run)
    {
        if (run.Channels != _dit.Config.InChannels)
            throw new ArgumentException($"The run has {run.Channels} channels; the DiT takes {_dit.Config.InChannels}.", nameof(run));
        if (run.Frames < 1 || run.Height < 1 || run.Width < 1)
            throw new ArgumentOutOfRangeException(nameof(run), "Frames, height and width must be positive.");
        int hw = run.Height * run.Width;
        if (run.Noise.Length != run.Channels * run.Frames * hw)
            throw new ArgumentException($"The noise holds {run.Noise.Length} values; expected {run.Channels * run.Frames * hw}.", nameof(run));
        if (run.Steps < 1)
            throw new ArgumentOutOfRangeException(nameof(run), "The step count must be at least 1.");
        if (!float.IsFinite(run.Shift) || run.Shift <= 0f)
            throw new ArgumentOutOfRangeException(nameof(run), "The scheduler shift must be positive and finite.");
        if (!float.IsFinite(run.TextGuidance) || run.TextGuidance < 0f || !float.IsFinite(run.AudioGuidance) || run.AudioGuidance < 0f)
            throw new ArgumentOutOfRangeException(nameof(run), "The guidance scales must be finite and non-negative.");
        int expectedAudio = run.Frames * _dit.Config.AudioContextTokens * _dit.Config.AudioOutputDim;
        if (run.AudioTokens is null || run.AudioTokens.Length != expectedAudio)
            throw new ArgumentException($"The audio tokens hold {run.AudioTokens?.Length ?? 0} values; expected {expectedAudio}.", nameof(run));
        bool guidedAudio = run.TextGuidance > 1f || run.AudioGuidance > 1f;
        if (guidedAudio && (run.UnconditionalAudioTokens is null || run.UnconditionalAudioTokens.Length != expectedAudio))
            throw new ArgumentException("Guidance needs the unconditional audio tokens: the projection of zero audio, not zero tokens.", nameof(run));
        run.Positive.Validate(_dit.Config.CaptionChannels);
        run.Negative.Validate(_dit.Config.CaptionChannels);

        switch (run.Mode)
        {
            case LongCatAvatarMode.At2v:
                if (run.NumCondLatents != 0)
                    throw new ArgumentException("At2v has no conditioning frames.", nameof(run));
                break;
            case LongCatAvatarMode.Ai2v:
                if (run.NumCondLatents != 1)
                    throw new ArgumentException("Ai2v conditions on exactly one frame.", nameof(run));
                RequireCond(run, hw);
                break;
            default:
                if (run.NumCondLatents < 1 || run.NumCondLatents >= run.Frames)
                    throw new ArgumentException("Avc needs at least one conditioning frame and at least one noisy frame.", nameof(run));
                RequireCond(run, hw);
                break;
        }
    }

    private static void RequireCond(LongCatAvatarRun run, int hw)
    {
        if (run.CondLatents is null || run.CondLatents.Length != run.Channels * run.NumCondLatents * hw)
            throw new ArgumentException("The conditioning latents do not match the conditioning frame count.", nameof(run));
    }

    private static void WriteFrames(float[] latents, float[] source, int channels, int frames, int hw, int count)
    {
        int k = 0;
        for (int c = 0; c < channels; c++)
            for (int f = 0; f < count; f++)
                for (int e = 0; e < hw; e++, k++)
                    latents[(c * frames + f) * hw + e] = source[k];
    }

    private static float[] LastFrames(float[] latent, int channels, int frames, int hw, int count)
    {
        float[] result = new float[channels * count * hw];
        int k = 0;
        for (int c = 0; c < channels; c++)
            for (int f = frames - count; f < frames; f++)
                for (int e = 0; e < hw; e++, k++)
                    result[k] = latent[(c * frames + f) * hw + e];
        return result;
    }

    private static Tensor Frames(float[] latent, int channels, int frames, int hw, int from, int count, int height, int width)
    {
        Tensor tensor = new(new TensorShape([1, channels, count, height, width]), DType.F32);
        Span<float> destination = tensor.AsSpan<float>();
        for (int c = 0; c < channels; c++)
            for (int t = 0; t < count; t++)
                latent.AsSpan(((c * frames) + from + t) * hw, hw).CopyTo(destination.Slice(((c * count) + t) * hw, hw));
        return tensor;
    }
}
