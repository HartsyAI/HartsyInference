using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Requests;

namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>One prompt's umT5 features: <paramref name="Embeds"/> is <c>[Length, captionChannels]</c> row-major, padded positions included
/// (the pipeline never zeroes them; the DiT zeroes the rows past <paramref name="ValidTokens"/> itself when the config asks it to).</summary>
public sealed record LongCatPrompt(float[] Embeds, int Length, int ValidTokens)
{
    /// <summary>Checks the shape and that the valid count is a prefix length.</summary>
    public void Validate(int captionChannels)
    {
        if (Length < 1 || Embeds.Length != (long)Length * captionChannels)
            throw new ArgumentException($"The prompt holds {Embeds.Length} values; expected {Length} x {captionChannels}.", nameof(Embeds));
        if (ValidTokens < 1 || ValidTokens > Length)
            throw new ArgumentOutOfRangeException(nameof(ValidTokens), ValidTokens, $"The valid length must lie in [1, {Length}].");
    }
}

/// <summary>What a run conditions on. <see cref="Text"/> is text-to-video; <see cref="Image"/> is image-to-video (the first latent frame is
/// the image, and the classifier-free guidance combination covers every frame, as the reference's uncached path does);
/// <see cref="Continuation"/> is the reference's <c>generate_vc</c> default (its KV cache, so the guidance combination covers the noisy frames
/// only and the context frames are never stepped).</summary>
public enum LongCatConditioningMode { Text, Image, Continuation }

/// <summary>One denoising run on a latent of shape <c>[channels, frames, height, width]</c> (batch 1). The latents are normalized
/// (the VAE's normalization is applied by the caller, outside this pipeline).</summary>
public sealed record LongCatRun(
    int Channels, int Frames, int Height, int Width,
    float[] Noise,
    LongCatPrompt Positive, LongCatPrompt Negative,
    int Steps, bool Distill, float Shift, float Guidance,
    LongCatConditioningMode Mode = LongCatConditioningMode.Text,
    int NumCondLatents = 0, float[]? CondLatents = null,
    bool EnhanceHf = false,
    bool RecordTrace = false, Action<GenerationProgress>? OnProgress = null, CancellationToken Cancel = default);

/// <summary>One step as the reference records it: the guided, negated velocity over the stepped frames, and the latent after the step.</summary>
public sealed record LongCatStepTrace(float Timestep, float Sigma, float[] Velocity, float[] SampleOut);

/// <summary>The result of a run: the final normalized latent (the context frames included), the schedule it used, and the per-step trace.</summary>
public sealed record LongCatResult(float[] Latents, LongCatSchedule Schedule, IReadOnlyList<LongCatStepTrace> Steps);

/// <summary>LongCat-Video base pipelines on a <see cref="LongCatTransformer"/>: text-to-video, image-to-video and continuation (with
/// the <c>enhance_hf</c> schedule), and the segment chain of long video. Latent math runs on the host in F32; the DiT runs on
/// <see cref="IBackend"/>. Text and VAE stay outside: the caller supplies umT5 features and normalized conditioning latents (the Wan VAE
/// encode and decode, and the umT5 numerics, are verified separately, not here).</summary>
/// <remarks>Parity with the pinned reference (<c>meituan-longcat/LongCat-Video</c> @ 58eb336) is checked by
/// <c>LongCatPipelineParityTests</c> on the tiny checkpoint. Not thread-safe: one run at a time per transformer.</remarks>
public sealed class LongCatVideoPipeline
{
    private readonly IBackend _backend;
    private readonly LongCatTransformer _dit;

    /// <summary>Creates the pipeline over a loaded transformer.</summary>
    public LongCatVideoPipeline(IBackend backend, LongCatTransformer dit)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _dit = dit ?? throw new ArgumentNullException(nameof(dit));
    }

    /// <summary>Runs the denoising loop of one run. Image and continuation runs write their conditioning latents into the leading frames
    /// before the first step, and only the stepped frames change.</summary>
    public LongCatResult Denoise(LongCatRun run)
    {
        Validate(run);
        int channels = run.Channels, frames = run.Frames, hw = run.Height * run.Width;
        int cond = run.Mode == LongCatConditioningMode.Text ? 0 : run.NumCondLatents;
        LongCatSchedule schedule = LongCatSchedule.Build(run.Steps, run.Distill, run.Shift);
        if (run.Mode == LongCatConditioningMode.Continuation && run.EnhanceHf)
            schedule = schedule.EnhanceHf();

        float[] latents = (float[])run.Noise.Clone();
        if (cond > 0)
            WriteFrames(latents, run.CondLatents!, channels, frames, hw, cond);

        LongCatConditioning conditioning = new(cond, 0, -1, 0);
        // The guidance scale covers the frames the reference's guidance sees: all of them, except in continuation's cached path.
        int guidedFrom = run.Mode == LongCatConditioningMode.Continuation ? cond : 0;
        bool guided = run.Guidance > 1f;
        List<LongCatStepTrace> trace = new(run.RecordTrace ? schedule.Steps : 0);
        long started = System.Diagnostics.Stopwatch.GetTimestamp();

        for (int i = 0; i < schedule.Steps; i++)
        {
            run.Cancel.ThrowIfCancellationRequested();
            float timestep = schedule.Timesteps[i];
            float dt = schedule.Sigmas[i + 1] - schedule.Sigmas[i];
            float[] frameTimesteps = conditioning.FrameTimesteps(timestep, frames);

            float[] velocity = guided
                ? Guide(Predict(latents, run, channels, frames, run.Positive, frameTimesteps, conditioning),
                    Predict(latents, run, channels, frames, run.Negative, frameTimesteps, conditioning),
                    guidedFrom, channels, frames, hw, run.Guidance)
                : Predict(latents, run, channels, frames, run.Positive, frameTimesteps, conditioning);

            // the reference negates the model output for its scheduler, then steps the frames after the conditioning ones.
            float[] stepped = new float[channels * (frames - cond) * hw];
            int k = 0;
            for (int c = 0; c < channels; c++)
                for (int f = cond; f < frames; f++)
                    for (int e = 0; e < hw; e++, k++)
                        stepped[k] = -velocity[(c * frames + f) * hw + e];

            float[]? sampleOut = run.RecordTrace ? new float[stepped.Length] : null;
            k = 0;
            for (int c = 0; c < channels; c++)
                for (int f = cond; f < frames; f++)
                    for (int e = 0; e < hw; e++, k++)
                    {
                        int index = (c * frames + f) * hw + e;
                        latents[index] += dt * stepped[k];
                        if (sampleOut is not null)
                            sampleOut[k] = latents[index];
                    }
            if (sampleOut is not null)
                trace.Add(new LongCatStepTrace(timestep, schedule.Sigmas[i], stepped, sampleOut));
            run.OnProgress?.Invoke(new GenerationProgress(i + 1, schedule.Steps, System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds));
        }
        return new LongCatResult(latents, schedule, trace);
    }

    /// <summary>Continuation chain (long video): each segment's context is the last <paramref name="contextFrames"/> latent frames of the
    /// previous segment's output, and a segment contributes its frames after the context. Returns every segment's full output (context
    /// included), each <c>[channels, frames, height, width]</c>. The segments are latent-space: the reference re-encodes decoded pixel frames,
    /// and that round trip is outside this pipeline.</summary>
    public IReadOnlyList<float[]> ContinuationChain(LongCatRun template, float[] firstContext, IReadOnlyList<float[]> noises, int contextFrames)
    {
        ArgumentNullException.ThrowIfNull(template);
        ArgumentNullException.ThrowIfNull(noises);
        List<float[]> outputs = [];
        float[] context = firstContext;
        foreach (float[] noise in noises)
        {
            LongCatRun segment = template with
            {
                Mode = LongCatConditioningMode.Continuation,
                NumCondLatents = contextFrames,
                CondLatents = context,
                Noise = noise,
                EnhanceHf = true,
            };
            float[] output = Denoise(segment).Latents;
            outputs.Add(output);
            context = LastFrames(output, template.Channels, template.Frames, template.Height * template.Width, contextFrames);
        }
        return outputs;
    }

    /// <summary>The last <paramref name="count"/> frames of a <c>[channels, frames, hw]</c> latent, as a <c>[channels, count, hw]</c> latent.</summary>
    public static float[] LastFrames(float[] latent, int channels, int frames, int hw, int count)
    {
        float[] result = new float[channels * count * hw];
        int k = 0;
        for (int c = 0; c < channels; c++)
            for (int f = frames - count; f < frames; f++)
                for (int e = 0; e < hw; e++, k++)
                    result[k] = latent[(c * frames + f) * hw + e];
        return result;
    }

    private float[] Predict(float[] latents, LongCatRun run, int channels, int frames, LongCatPrompt prompt, float[] frameTimesteps,
        LongCatConditioning conditioning)
    {
        using Tensor input = new(new TensorShape([1, channels, frames, run.Height, run.Width]), DType.F32);
        latents.AsSpan().CopyTo(input.AsSpan<float>());
        using Tensor text = new(new TensorShape(prompt.Length, _dit.Config.CaptionChannels), DType.F32);
        prompt.Embeds.AsSpan().CopyTo(text.AsSpan<float>());
        using Tensor output = _dit.Forward(_backend, input, text, frameTimesteps, in conditioning, prompt.ValidTokens);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>The optimized-scale guidance of the reference (CFG-zero): the unconditional prediction scaled by the projection of the
    /// conditional one onto it, over the frames from <paramref name="guidedFrom"/> on.</summary>
    private static float[] Guide(float[] positive, float[] negative, int guidedFrom, int channels, int frames, int hw, float guidance)
    {
        double dot = 0, squared = 0;
        for (int c = 0; c < channels; c++)
            for (int f = guidedFrom; f < frames; f++)
                for (int e = 0; e < hw; e++)
                {
                    int index = (c * frames + f) * hw + e;
                    dot += positive[index] * negative[index];
                    squared += negative[index] * negative[index];
                }
        float scale = (float)(dot / (squared + 1e-8));
        float[] guided = (float[])positive.Clone();
        for (int c = 0; c < channels; c++)
            for (int f = guidedFrom; f < frames; f++)
                for (int e = 0; e < hw; e++)
                {
                    int index = (c * frames + f) * hw + e;
                    float uncond = negative[index] * scale;
                    guided[index] = uncond + guidance * (positive[index] - uncond);
                }
        return guided;
    }

    private static void WriteFrames(float[] latents, float[] source, int channels, int frames, int hw, int count)
    {
        if (source.Length != channels * count * hw)
            throw new ArgumentException($"The conditioning latents hold {source.Length} values; expected {channels} x {count} x {hw}.", nameof(source));
        int k = 0;
        for (int c = 0; c < channels; c++)
            for (int f = 0; f < count; f++)
                for (int e = 0; e < hw; e++, k++)
                    latents[(c * frames + f) * hw + e] = source[k];
    }

    private void Validate(LongCatRun run)
    {
        if (run.Channels != _dit.Config.InChannels)
            throw new ArgumentException($"The run has {run.Channels} channels; the transformer takes {_dit.Config.InChannels}.", nameof(run));
        if (run.Noise.Length != (long)run.Channels * run.Frames * run.Height * run.Width)
            throw new ArgumentException("The noise does not match [channels, frames, height, width].", nameof(run));
        (int _, int patchH, int patchW) = _dit.Config.PatchSize;
        if (run.Height % patchH != 0 || run.Width % patchW != 0)
            throw new ArgumentException("The latent height and width must tile into the patch size.", nameof(run));
        if (run.Mode == LongCatConditioningMode.Text && run.NumCondLatents != 0)
            throw new ArgumentException("Text-to-video takes no conditioning latents.", nameof(run));
        if (run.Mode != LongCatConditioningMode.Text && (run.NumCondLatents < 1 || run.NumCondLatents > run.Frames))
            throw new ArgumentOutOfRangeException(nameof(run), run.NumCondLatents, $"Conditioning needs 1 to {run.Frames} latent frames.");
        if (run.Mode == LongCatConditioningMode.Image && run.NumCondLatents != 1)
            throw new ArgumentException("Image-to-video conditions on exactly one latent frame.", nameof(run));
        if (run.NumCondLatents > 0 && run.CondLatents is null)
            throw new ArgumentException("Conditioning latents are required.", nameof(run));
        if (run.Mode == LongCatConditioningMode.Continuation && run.EnhanceHf && run.Distill)
            throw new ArgumentException("The reference refuses enhance_hf with the distilled schedule.", nameof(run));
        run.Positive.Validate(_dit.Config.CaptionChannels);
        run.Negative.Validate(_dit.Config.CaptionChannels);
    }
}
