using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Requests;

namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>The avatar's conditioning modes: <c>At2v</c> (audio to video, no conditioning frames), <c>Ai2v</c> (one image frame first), and
/// <c>Avc</c> (continuation: conditioning frames, one of them the reference latent at <see cref="LongCatAvatarRun.RefImgIndex"/>).</summary>
public enum LongCatAvatarMode { At2v, Ai2v, Avc }

/// <summary>One avatar denoising run. <see cref="Frames"/> counts every latent frame the DiT sees, the conditioning frames (and, for avc, the
/// reference frame) included. <see cref="AudioTokens"/> holds the audio tokens of every one of those frames,
/// <c>[Frames·contextTokens, outputDim]</c> (see <see cref="LongCatTransformer.ProjectAudio"/>). <see cref="UnconditionalAudioTokens"/> is the
/// projection of zero raw audio with the same shape; it is required when either guidance scale is above 1.
/// <see cref="Kind"/> picks the reference's sigma path, and the scheduler's static <see cref="Shift"/> applies to it (the shipped 1.6 scheduler
/// config has shift 7). <see cref="UseKvCache"/> applies only to <see cref="LongCatAvatarMode.Avc"/>; at2v and ai2v ignore it.</summary>
public sealed record LongCatAvatarRun(
    int Channels, int Frames, int Height, int Width, float[] Noise,
    LongCatPrompt Positive, LongCatPrompt Negative, float[] AudioTokens, float[]? UnconditionalAudioTokens,
    int Steps, LongCatAvatarSchedule.Kind Kind, float Shift,
    float TextGuidance, float AudioGuidance,
    LongCatAvatarMode Mode, int NumCondLatents = 0, float[]? CondLatents = null,
    int RefImgIndex = 10, bool UseKvCache = true, Action<GenerationProgress>? OnProgress = null, CancellationToken Cancel = default,
    AdjunctGate? LoraGate = null);

/// <summary>The sigmas (shifted, with the terminal zero), the timesteps (sigma·1000), and the final latents.</summary>
public sealed record LongCatAvatarResult(float[] Latents, float[] Sigmas, float[] Timesteps);

/// <summary>The avatar denoising loop (<c>generate_at2v</c>, <c>generate_ai2v</c> and <c>generate_avc</c> in
/// <c>pipeline_longcat_video_avatar.py</c> @ 58eb336), for one segment in latent space. The guidance is the reference's three-pass
/// form: <c>uncond + text·(cond − uncondText) + audio·(uncondText − uncond)</c>, where <c>cond</c> is the positive prompt with the audio,
/// <c>uncondText</c> the negative prompt with the audio, and <c>uncond</c> the negative prompt with zero audio. When both scales are 1 (the
/// distilled 1.5 path) one pass suffices. With the cache, the conditioning frames go through the cache pass once and each step runs only the
/// noisy frames. Without it, every step runs all frames with the conditioning frames' timesteps at zero. Either way the conditioning frames are
/// never stepped.</summary>
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
        // The distilled schedule is valid only with its LoRA live: on bare weights the 8-step path is not the distilled model.
        if (run.Kind == LongCatAvatarSchedule.Kind.Distill8 && run.LoraGate is null)
            throw new InvalidOperationException("The 8-step distilled schedule needs its DMD LoRA gate; refusing to run the bare base at 8 steps.");
        Validate(run);
        // The gate stays on for every step of this run; the stacked LoRA's strength (0.8) is its own, so the gate scale is 1.
        using IDisposable? loraScope = run.LoraGate?.Enable(1f);
        using LongCatRunTensors tensors = new(_dit.Config.CaptionChannels, _dit.Config.AudioOutputDim);
        int channels = run.Channels, frames = run.Frames, hw = run.Height * run.Width;
        int cond = run.Mode == LongCatAvatarMode.At2v ? 0 : run.NumCondLatents;
        int noisy = frames - cond;
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

        // The audio each pass reads is fixed for the whole run: copy it once, not per step.
        int audioFrom = useCache ? cond : 0, audioCount = useCache ? noisy : frames;
        float[] audioCond = AudioRows(run.AudioTokens, audioFrom, audioCount);
        float[]? audioUncond = guided ? AudioRows(run.UnconditionalAudioTokens!, audioFrom, audioCount) : null;

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

            long started = System.Diagnostics.Stopwatch.GetTimestamp();
            for (int i = 0; i < run.Steps; i++)
            {
                run.Cancel.ThrowIfCancellationRequested();
                float t = timesteps[i], dt = sigmasWithZero[i + 1] - sigmasWithZero[i];
                float[] velocity = Velocity(run, cache, latents, channels, frames, hw, cond, conditioning, t, guided, audioCond, audioUncond, tensors);

                // The reference negates the model output for its scheduler and steps the noisy frames only.
                for (int c = 0; c < channels; c++)
                    for (int f = 0; f < noisy; f++)
                        for (int e = 0; e < hw; e++)
                        {
                            int index = (c * frames + cond + f) * hw + e;
                            int vIndex = (c * noisy + f) * hw + e;
                            latents[index] += dt * -velocity[vIndex];
                        }
                run.OnProgress?.Invoke(new GenerationProgress(i + 1, run.Steps, System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds));
            }
        }
        finally
        {
            cache?.Dispose();
            condLatent?.Dispose();
        }

        return new LongCatAvatarResult(latents, sigmasWithZero, timesteps);
    }

    /// <summary>The velocity of the noisy frames, <c>[channels, noisy, hw]</c>, with the cache (the noisy frames only) or without it (all frames, the
    /// noisy rows taken from the result).</summary>
    private float[] Velocity(LongCatAvatarRun run, LongCatKvCache? cache, float[] latents, int channels, int frames, int hw, int cond,
        LongCatConditioning conditioning, float t, bool guided, float[] audioCond, float[]? audioUncond, LongCatRunTensors tensors)
    {
        int noisy = frames - cond;
        bool cached = cache is not null;
        float[] input = cached ? LastFrames(latents, channels, frames, hw, noisy) : latents;
        int inputFrames = cached ? noisy : frames;
        float[] timesteps = cached ? [t] : conditioning.FrameTimesteps(t, frames);
        float[] full = Guided(run, guided, (promptIsPositive, audioOn) =>
            Predict(input, channels, inputFrames, run.Height, run.Width, promptIsPositive ? run.Positive : run.Negative,
                audioOn ? audioCond : audioUncond!, timesteps, conditioning, cache, tensors));
        if (cached)
            return full;

        float[] result = new float[channels * noisy * hw];
        int k = 0;
        for (int c = 0; c < channels; c++)
            for (int f = cond; f < frames; f++)
                for (int e = 0; e < hw; e++, k++)
                    result[k] = full[(c * frames + f) * hw + e];
        return result;
    }

    /// <summary>The three-pass guidance, or one pass when unguided. <paramref name="pass"/> takes (positive prompt, audio on).</summary>
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
        float[] frameTimesteps, LongCatConditioning conditioning, LongCatKvCache? cache, LongCatRunTensors tensors)
    {
        using Tensor input = new(new TensorShape([1, channels, frames, height, width]), DType.F32);
        latents.AsSpan().CopyTo(input.AsSpan<float>());
        Tensor text = tensors.Text(prompt);
        Tensor audioTokens = tensors.Audio(audio, frames * _dit.Config.AudioContextTokens);
        using Tensor output = cache is null
            ? _dit.Forward(_backend, input, text, frameTimesteps, in conditioning, prompt.ValidTokens, audioTokens)
            : _dit.ForwardCached(_backend, input, text, frameTimesteps, in conditioning, cache, prompt.ValidTokens, audioTokens);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>The audio tokens of <paramref name="count"/> frames from <paramref name="from"/>, copied out of <paramref name="tokens"/>.</summary>
    private float[] AudioRows(float[] tokens, int from, int count)
    {
        int ctx = _dit.Config.AudioContextTokens, audioWidth = _dit.Config.AudioOutputDim;
        float[] result = new float[count * ctx * audioWidth];
        Array.Copy(tokens, from * ctx * audioWidth, result, 0, result.Length);
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
        // Below 1 the reference's guidance form is not what this loop computes (a scale under 1 would reduce the text or audio term), so refuse it.
        if (!float.IsFinite(run.TextGuidance) || run.TextGuidance < 1f || !float.IsFinite(run.AudioGuidance) || run.AudioGuidance < 1f)
            throw new ArgumentOutOfRangeException(nameof(run), "The guidance scales must be finite and at least 1 (1 means no guidance on that term).");
        int expectedAudio = run.Frames * _dit.Config.AudioContextTokens * _dit.Config.AudioOutputDim;
        if (run.AudioTokens is null || run.AudioTokens.Length != expectedAudio)
            throw new ArgumentException($"The audio tokens hold {run.AudioTokens?.Length ?? 0} values; expected {expectedAudio}.", nameof(run));
        bool guided = run.TextGuidance > 1f || run.AudioGuidance > 1f;
        if (guided && (run.UnconditionalAudioTokens is null || run.UnconditionalAudioTokens.Length != expectedAudio))
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

    /// <summary>The long-video chain in latent space (the reference's segment loop in <c>run_demo_avatar_*</c>). Segment 0 is
    /// <paramref name="firstLatents"/> (its frames, no reference). Each later segment conditions on the reference latent (the first frame of
    /// segment 0, fixed) and the last <paramref name="contextFrames"/> frames of the previous segment, runs <c>generate_avc</c> with the reference
    /// index, drops the reference, and contributes its frames after the context. The reference re-encodes decoded pixel frames for the context;
    /// that round trip is the VAE seam and is not in this chain. The chain is not compared with a reference chain run yet.</summary>
    /// <param name="template">The per-segment run. Its <see cref="LongCatAvatarRun.Frames"/> is one segment's latent frames plus one for the
    /// reference slot (so a segment of N frames after the context needs Frames = N + 1). Its <see cref="LongCatAvatarRun.UnconditionalAudioTokens"/>
    /// is used for every later segment, so it must be sized for the template's Frames. Mode, conditioning, audio and noise are set per segment.</param>
    /// <param name="contextFrames">Latent frames of context carried between segments (the reference's 13 pixel frames are 4 latents).</param>
    /// <param name="firstLatents">Segment 0's latents, <c>[channels, template.Frames − 1, hw]</c>.</param>
    /// <param name="audioTokens">Each later segment's audio tokens, <c>[template.Frames·contextTokens, audioOutputDim]</c>.</param>
    /// <param name="noises">Each later segment's noise, <c>[channels, template.Frames, hw]</c>.</param>
    /// <returns>Segment 0 whole, then each later segment's new frames (context dropped), each <c>[channels, frames, hw]</c>.</returns>
    public IReadOnlyList<float[]> Chain(LongCatAvatarRun template, int contextFrames, float[] firstLatents, IReadOnlyList<float[]> audioTokens,
        IReadOnlyList<float[]> noises)
    {
        ArgumentNullException.ThrowIfNull(template);
        ArgumentNullException.ThrowIfNull(firstLatents);
        ArgumentNullException.ThrowIfNull(audioTokens);
        ArgumentNullException.ThrowIfNull(noises);
        if (audioTokens.Count != noises.Count)
            throw new ArgumentException("Each later segment needs its audio tokens and its noise.", nameof(noises));
        int channels = template.Channels, hw = template.Height * template.Width, frames = template.Frames;
        int segmentFrames = frames - 1, noisy = segmentFrames - contextFrames;
        if (contextFrames < 1 || noisy < 1)
            throw new ArgumentOutOfRangeException(nameof(contextFrames), contextFrames, "The context must be shorter than a segment.");
        if (firstLatents.Length != channels * segmentFrames * hw)
            throw new ArgumentException($"The first segment holds {firstLatents.Length} values; expected {channels * segmentFrames * hw}.", nameof(firstLatents));

        // The reference latent is the first frame of segment 0 and stays fixed; the context is the last frames of the previous segment.
        float[] reference = FrameOf(firstLatents, channels, segmentFrames, hw, 0);
        float[] context = LastFrames(firstLatents, channels, segmentFrames, hw, contextFrames);
        List<float[]> outputs = [firstLatents];
        for (int s = 0; s < noises.Count; s++)
        {
            LongCatAvatarRun run = template with
            {
                Mode = LongCatAvatarMode.Avc,
                NumCondLatents = contextFrames + 1,
                CondLatents = Concat(reference, context, channels, hw, contextFrames),
                AudioTokens = audioTokens[s],
                Noise = noises[s],
            };
            float[] full = Denoise(run).Latents;
            // Drop the reference (frame 0 of the run): what remains is the context followed by this segment's frames.
            float[] withoutReference = RemoveFirstFrame(full, channels, frames, hw);
            outputs.Add(LastFrames(withoutReference, channels, segmentFrames, hw, noisy));
            context = LastFrames(withoutReference, channels, segmentFrames, hw, contextFrames);
        }
        return outputs;
    }

    private static float[] FrameOf(float[] latent, int channels, int frames, int hw, int index)
    {
        float[] result = new float[channels * hw];
        for (int c = 0; c < channels; c++)
            latent.AsSpan((c * frames + index) * hw, hw).CopyTo(result.AsSpan(c * hw, hw));
        return result;
    }

    /// <summary>[reference, context] per channel, as one run's conditioning: <c>[channels, 1 + contextFrames, hw]</c>.</summary>
    private static float[] Concat(float[] reference, float[] context, int channels, int hw, int contextFrames)
    {
        float[] result = new float[channels * (1 + contextFrames) * hw];
        for (int c = 0; c < channels; c++)
        {
            int at = c * (1 + contextFrames) * hw;
            reference.AsSpan(c * hw, hw).CopyTo(result.AsSpan(at, hw));
            context.AsSpan(c * contextFrames * hw, contextFrames * hw).CopyTo(result.AsSpan(at + hw, contextFrames * hw));
        }
        return result;
    }

    /// <summary>A <c>[channels, frames, hw]</c> latent without its first frame: <c>[channels, frames − 1, hw]</c>.</summary>
    private static float[] RemoveFirstFrame(float[] latent, int channels, int frames, int hw)
    {
        float[] result = new float[channels * (frames - 1) * hw];
        for (int c = 0; c < channels; c++)
            latent.AsSpan((c * frames + 1) * hw, (frames - 1) * hw).CopyTo(result.AsSpan(c * (frames - 1) * hw, (frames - 1) * hw));
        return result;
    }

    /// <summary>The last <paramref name="count"/> frames of a <c>[channels, frames, hw]</c> latent, as <c>[channels, count, hw]</c>.</summary>
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
