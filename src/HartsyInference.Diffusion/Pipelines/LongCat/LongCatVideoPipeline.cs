using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Reduction;
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

    /// <summary>Runs the denoising loop of one run. Image and continuation runs keep their conditioning latents fixed in the leading frames, and only
    /// the stepped frames change. The stepped frames and the conditioning frames live on the device for the whole run; the host reads the latent
    /// back once, at the end, and once per step only when a trace is recorded.</summary>
    public LongCatResult Denoise(LongCatRun run)
    {
        Validate(run);
        int channels = run.Channels, frames = run.Frames, hw = run.Height * run.Width;
        int cond = run.Mode == LongCatConditioningMode.Text ? 0 : run.NumCondLatents;
        int stepped = frames - cond;
        LongCatSchedule schedule = LongCatSchedule.Build(run.Steps, run.Distill, run.Shift);
        if (run.Mode == LongCatConditioningMode.Continuation && run.EnhanceHf)
            schedule = schedule.EnhanceHf();
        if (cond > 0 && run.CondLatents!.Length != channels * cond * hw)
            throw new ArgumentException($"The conditioning latents hold {run.CondLatents.Length} values; expected {channels} x {cond} x {hw}.", nameof(run));

        LongCatConditioning conditioning = new(cond, 0, -1, 0);
        // The guidance scale covers the frames the reference's guidance sees: all of them, except in continuation's cached path. Every stepped
        // frame lies in that range, because it starts at the conditioning frame count.
        int guidedFrom = run.Mode == LongCatConditioningMode.Continuation ? cond : 0;
        bool guided = run.Guidance > 1f;
        List<LongCatStepTrace> trace = new(run.RecordTrace ? schedule.Steps : 0);
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        using LongCatRunTensors tensors = new(_dit.Config.CaptionChannels, _dit.Config.AudioOutputDim);
        using Tensor? condState = cond > 0 ? LongCatFrameOps.FromHost(run.CondLatents!, new TensorShape(channels * cond, hw)) : null;
        Tensor state = LongCatFrameOps.FromHost(LongCatFrameOps.SteppedFrames(run.Noise, channels, frames, hw, cond),
            new TensorShape(channels * stepped, hw));
        try
        {
            for (int i = 0; i < schedule.Steps; i++)
            {
                run.Cancel.ThrowIfCancellationRequested();
                float timestep = schedule.Timesteps[i];
                float dt = schedule.Sigmas[i + 1] - schedule.Sigmas[i];
                float[] frameTimesteps = conditioning.FrameTimesteps(timestep, frames);

                using Tensor clip = LongCatFrameOps.Clip(_backend, state, condState, channels, cond, stepped, hw);
                using Tensor input = LongCatFrameOps.Reshape(_backend, clip, new TensorShape([1, channels, frames, run.Height, run.Width]));
                using Tensor velocity = Velocity(run, input, frameTimesteps, conditioning, tensors, guided, guidedFrom, channels, frames, cond, hw);

                // The reference negates the model output for its scheduler, then steps the stepped frames: state += dt · (−velocity).
                using Tensor step = new(state.Shape, DType.F32);
                _backend.Scale(step, velocity, -dt);
                Tensor next = new(state.Shape, DType.F32);
                try
                {
                    _backend.Add(next, state, step);
                }
                catch
                {
                    next.Dispose();
                    throw;
                }
                if (run.RecordTrace)
                {
                    float[] negated = velocity.AsReadOnlySpan<float>().ToArray();
                    for (int k = 0; k < negated.Length; k++)
                        negated[k] = -negated[k];
                    trace.Add(new LongCatStepTrace(timestep, schedule.Sigmas[i], negated, next.AsReadOnlySpan<float>().ToArray()));
                }
                state.Dispose();
                state = next;
                run.OnProgress?.Invoke(new GenerationProgress(i + 1, schedule.Steps, System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds));
            }
            float[] latents = LongCatFrameOps.JoinFrames(run.CondLatents, state.AsReadOnlySpan<float>().ToArray(), channels, frames, hw, cond);
            return new LongCatResult(latents, schedule, trace);
        }
        finally
        {
            state.Dispose();
        }
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

    /// <summary>One forward pass over the whole clip; the velocity stays on the device, owned by the caller.</summary>
    private Tensor Predict(Tensor input, LongCatRun run, LongCatPrompt prompt, float[] frameTimesteps, LongCatConditioning conditioning,
        LongCatRunTensors tensors)
    {
        Tensor text = tensors.Text(prompt);
        return _dit.Forward(_backend, input, text, frameTimesteps, in conditioning, prompt.ValidTokens);
    }

    /// <summary>The stepped frames' velocity, <c>[channels·stepped, hw]</c>, on the device. Guided runs take the CFG-zero combine.</summary>
    private Tensor Velocity(LongCatRun run, Tensor input, float[] frameTimesteps, LongCatConditioning conditioning, LongCatRunTensors tensors,
        bool guided, int guidedFrom, int channels, int frames, int cond, int hw)
    {
        if (guided)
        {
            // Both velocities stay live until the reduction, which reads both. The positive one is one latent-sized tensor
            // (about 9.6 MB at 832 x 480 x 93), so keeping it through the negative forward costs little next to the activations.
            using Tensor positive = Predict(input, run, run.Positive, frameTimesteps, conditioning, tensors);
            using Tensor negative = Predict(input, run, run.Negative, frameTimesteps, conditioning, tensors);
            return Guide(positive, negative, guidedFrom, channels, frames, cond, hw, run.Guidance);
        }
        using Tensor output = Predict(input, run, run.Positive, frameTimesteps, conditioning, tensors);
        using Tensor flat = LongCatFrameOps.Reshape(_backend, output, new TensorShape(channels * frames, hw));
        return LongCatFrameOps.Rows(_backend, flat, channels, frames, cond, frames - cond, hw);
    }

    /// <summary>The optimized-scale guidance (CFG-zero) on the device: the unconditional prediction scaled by the projection of the conditional one
    /// onto it, over the frames from <paramref name="guidedFrom"/> on. The two sums come from one device reduction over that frame range; the
    /// combine and the stepped frames' rows stay on the device. Every stepped frame is at or after <paramref name="guidedFrom"/>, so the combine
    /// covers all of them.</summary>
    private Tensor Guide(Tensor positive, Tensor negative, int guidedFrom, int channels, int frames, int cond, int hw, float guidance)
    {
        if (guidedFrom > cond)
            throw new InvalidOperationException("The guided frames must start at or before the first stepped frame.");
        StridedView region = StridedView.Of([channels, frames, hw]).Slice(1, guidedFrom, frames - guidedFrom);
        // One reduction reads both velocities for the two sums; the host reads two doubles back, and nothing else.
        (double dot, double squared) = ProductSums.DotAndSquaredB(_backend, positive, negative, region);
        float scale = (float)(dot / (squared + 1e-8));

        using Tensor uncond = new(negative.Shape, DType.F32);
        _backend.Scale(uncond, negative, scale);
        using Tensor difference = Difference(positive, uncond);
        using Tensor weighted = new(difference.Shape, DType.F32);
        _backend.Scale(weighted, difference, guidance);
        Tensor guidedAll = new(uncond.Shape, DType.F32);
        try
        {
            _backend.Add(guidedAll, uncond, weighted);
        }
        catch
        {
            guidedAll.Dispose();
            throw;
        }
        using (guidedAll)
        {
            using Tensor flat = LongCatFrameOps.Reshape(_backend, guidedAll, new TensorShape(channels * frames, hw));
            return LongCatFrameOps.Rows(_backend, flat, channels, frames, cond, frames - cond, hw);
        }
    }

    /// <summary>a − b, computed as a + (−1)·b, which is the same float result.</summary>
    private Tensor Difference(Tensor a, Tensor b)
    {
        using Tensor negated = new(b.Shape, DType.F32);
        _backend.Scale(negated, b, -1f);
        Tensor difference = new(a.Shape, DType.F32);
        try
        {
            _backend.Add(difference, a, negated);
            return difference;
        }
        catch
        {
            difference.Dispose();
            throw;
        }
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
