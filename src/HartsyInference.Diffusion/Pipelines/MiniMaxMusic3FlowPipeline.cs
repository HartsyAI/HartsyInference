using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>MiniMax Music 3's flow-matching stage: turns the autoregressive stage's per-frame hidden states into
/// Flow-VAE latents, one 200-frame window at a time with a 100-frame hop, blending each window into the previous
/// one over their overlap so neighbouring windows share a boundary.
///
/// <para>The integrator is written out rather than routed through <see cref="Schedulers.FlowMatchEulerDiscreteScheduler"/>:
/// the checkpoint's scheduler config sets <c>invert_sigmas</c> with <c>num_train_timesteps = 1</c> over
/// <c>linspace(1, 1/N, N)</c>, which reduces to <c>sigma_i = i/N</c> with a uniform <c>1/N</c> step — a plain Euler
/// walk from t=0 (noise) to t=1 (data). The shared scheduler exposes neither the injected sigmas nor the
/// inversion.</para></summary>
public sealed unsafe class MiniMaxMusic3FlowPipeline : DiffusionPipelineBase
{
    /// <summary>Autoregressive frames per denoising window.</summary>
    public const int ChunkFrames = 200;

    /// <summary>Frames advanced between windows.</summary>
    public const int ChunkHop = 100;

    /// <summary>Latent frames of a window that are blended toward the previous window.</summary>
    public const int OverlapLatentLength = 172;

    /// <summary>Leading latents every window after the first drops when its waveform is stitched.</summary>
    public const int CropLeftLatents = 86;

    /// <summary>Trailing latents every window before the last drops when its waveform is stitched.</summary>
    public const int CropRightLatents = (2 * OverlapLatentLength) - CropLeftLatents;

    /// <summary>The reference's guidance scale for this stage.</summary>
    public const float DefaultCfgScale = 1.7f;

    /// <summary>The reference's step count.</summary>
    public const int DefaultSteps = 30;

    private const int LatentChannels = 128;

    private readonly MiniMaxMusic3ConditionEncoder _conditionEncoder;
    private readonly MiniMaxMusic3Dit _dit;

    public MiniMaxMusic3FlowPipeline(IBackend backend, MiniMaxMusic3ConditionEncoder conditionEncoder, MiniMaxMusic3Dit dit)
        : base(backend)
    {
        ArgumentNullException.ThrowIfNull(conditionEncoder);
        ArgumentNullException.ThrowIfNull(dit);
        _conditionEncoder = conditionEncoder;
        _dit = dit;
    }

    /// <summary>Frame index at which each denoising window starts.</summary>
    public static int[] ChunkStarts(int frames)
    {
        if (frames <= ChunkFrames)
        {
            return [0];
        }
        int count = ((frames - ChunkHop - 1) / ChunkHop) + 1;
        int[] starts = new int[count];
        for (int i = 0; i < count; i++)
        {
            starts[i] = i * ChunkHop;
        }
        return starts;
    }

    /// <summary>Crops the per-window waveforms so the kept spans tile the song exactly, and concatenates them into
    /// one stereo pair. Neighbouring windows overlap by <c>2 · <see cref="OverlapLatentLength"/></c> latents: every
    /// window after the first drops its leading <see cref="CropLeftLatents"/> and every window before the last drops
    /// its trailing <see cref="CropRightLatents"/>.</summary>
    /// <param name="waveforms">One <c>[1, 2, samples]</c> waveform per window, in order.</param>
    /// <param name="hopLength">Waveform samples per latent frame.</param>
    public static (float[] Left, float[] Right) Stitch(IReadOnlyList<Tensor> waveforms, int hopLength)
    {
        ArgumentNullException.ThrowIfNull(waveforms);
        if (waveforms.Count == 0)
        {
            throw new ArgumentException("at least one window is required.", nameof(waveforms));
        }
        int total = 0;
        int[] offsets = new int[waveforms.Count];
        int[] lengths = new int[waveforms.Count];
        for (int i = 0; i < waveforms.Count; i++)
        {
            int samples = (int)waveforms[i].Shape[2];
            offsets[i] = i == 0 ? 0 : CropLeftLatents * hopLength;
            int right = i == waveforms.Count - 1 ? 0 : CropRightLatents * hopLength;
            lengths[i] = Math.Max(0, samples - offsets[i] - right);
            total += lengths[i];
        }

        float[] left = new float[total];
        float[] right2 = new float[total];
        int written = 0;
        for (int i = 0; i < waveforms.Count; i++)
        {
            int samples = (int)waveforms[i].Shape[2];
            ReadOnlySpan<float> values = waveforms[i].AsReadOnlySpan<float>();
            values.Slice(offsets[i], lengths[i]).CopyTo(left.AsSpan(written, lengths[i]));
            values.Slice(samples + offsets[i], lengths[i]).CopyTo(right2.AsSpan(written, lengths[i]));
            written += lengths[i];
        }
        for (int i = 0; i < total; i++)
        {
            left[i] = Math.Clamp(left[i], -1f, 1f);
            right2[i] = Math.Clamp(right2[i], -1f, 1f);
        }
        return (left, right2);
    }

    /// <summary>Denoises every window of <paramref name="frameHiddens"/> (flat
    /// <c>[frames · 8 · 4096]</c>) and returns the uncropped per-window latents. Caller disposes them.
    ///
    /// <para><paramref name="forcedNoise"/> substitutes the reference's saved noise draws so a parity run compares
    /// the integrator rather than two incompatible RNGs; pass null for real generation.</para></summary>
    public Tensor[] Denoise(
        ReadOnlySpan<float> frameHiddens,
        int frames,
        int steps,
        float cfgScale,
        int seed,
        Action<int, int>? onStep = null,
        CancellationToken cancel = default,
        IReadOnlyList<float[]>? forcedNoise = null)
    {
        ThrowIfDisposed();
        if (frames <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(frames), frames, "frames must be positive.");
        }
        int stepCount = Math.Max(1, steps);
        int[] starts = ChunkStarts(frames);
        Tensor[] chunks = new Tensor[starts.Length];
        int totalSteps = starts.Length * stepCount;
        int completedSteps = 0;

        Tensor? previousLatent = null;
        Tensor? previousCondition = null;
        try
        {
            for (int chunk = 0; chunk < starts.Length; chunk++)
            {
                cancel.ThrowIfCancellationRequested();
                int start = starts[chunk];
                int count = Math.Min(start + ChunkFrames, frames) - start;
                Tensor condition = _conditionEncoder.Encode(Backend, frameHiddens, start, count);
                int length = (int)condition.Shape[1];
                int overlap = previousLatent is null ? 0 : Math.Min((int)previousLatent.Shape[2], length);
                if (overlap > 0)
                {
                    SpliceCondition(condition, previousCondition!, overlap);
                }

                Tensor latents = DrawNoise(length, seed, forcedNoise, chunk);
                using Tensor noisePrompt = SliceLeading(latents, overlap);
                using Tensor zeros = new Tensor(condition.Shape, DType.F32);

                for (int step = 0; step < stepCount; step++)
                {
                    cancel.ThrowIfCancellationRequested();
                    float time = (float)step / stepCount;
                    if (overlap > 0)
                    {
                        BlendOverlap(latents, noisePrompt, previousLatent!, overlap, time);
                    }
                    using Tensor conditional = _dit.Forward(Backend, latents, time, condition);
                    using Tensor unconditional = _dit.Forward(Backend, latents, time, zeros);
                    using Tensor velocity = CfgHelper.ApplyCfg(unconditional, conditional, cfgScale);
                    Advance(latents, velocity, 1f / stepCount);
                    onStep?.Invoke(++completedSteps, totalSteps);
                }

                if (overlap > 0)
                {
                    RestoreOverlap(latents, previousLatent!, overlap);
                }
                int carryStart = Math.Max(0, length - (2 * OverlapLatentLength));
                int carryEnd = Math.Max(carryStart, length - OverlapLatentLength);
                previousLatent?.Dispose();
                previousCondition?.Dispose();
                previousLatent = SliceLatents(latents, carryStart, carryEnd);
                previousCondition = SliceCondition(condition, carryStart, carryEnd);
                condition.Dispose();
                chunks[chunk] = latents;
            }
        }
        catch
        {
            foreach (Tensor? built in chunks)
            {
                built?.Dispose();
            }
            throw;
        }
        finally
        {
            previousLatent?.Dispose();
            previousCondition?.Dispose();
        }
        return chunks;
    }

    /// <summary>Each window draws its own noise from a per-window seed. The reference draws sequentially from one
    /// generator instead; either way the windows are independent and the run reproduces from <paramref name="seed"/>.</summary>
    private static Tensor DrawNoise(int length, int seed, IReadOnlyList<float[]>? forcedNoise, int chunk)
    {
        TensorShape shape = new TensorShape(1, LatentChannels, length);
        if (forcedNoise is null || chunk >= forcedNoise.Count)
        {
            return SeedGenerator.CreateNoise(shape, unchecked(seed + (chunk * 7919)));
        }
        Tensor latents = new Tensor(shape, DType.F32);
        Span<float> values = new Span<float>((float*)latents.DataPointer, LatentChannels * length);
        forcedNoise[chunk].AsSpan(0, values.Length).CopyTo(values);
        return latents;
    }

    /// <summary>Overwrites the window's leading <paramref name="overlap"/> latents with a blend of its own initial
    /// noise and the previous window's trailing latents, weighted by the flow-matching time.</summary>
    private static void BlendOverlap(Tensor latents, Tensor noisePrompt, Tensor previousLatent, int overlap, float time)
    {
        int length = (int)latents.Shape[2];
        int previousLength = (int)previousLatent.Shape[2];
        float noiseWeight = 1f - ((1f - 1e-6f) * time);
        float* target = (float*)latents.DataPointer;
        float* noise = (float*)noisePrompt.DataPointer;
        float* previous = (float*)previousLatent.DataPointer;
        for (int channel = 0; channel < LatentChannels; channel++)
        {
            long targetRow = (long)channel * length;
            long noiseRow = (long)channel * overlap;
            long previousRow = (long)channel * previousLength;
            for (int i = 0; i < overlap; i++)
            {
                target[targetRow + i] = (noiseWeight * noise[noiseRow + i]) + (time * previous[previousRow + i]);
            }
        }
    }

    private static void RestoreOverlap(Tensor latents, Tensor previousLatent, int overlap)
    {
        int length = (int)latents.Shape[2];
        int previousLength = (int)previousLatent.Shape[2];
        float* target = (float*)latents.DataPointer;
        float* previous = (float*)previousLatent.DataPointer;
        for (int channel = 0; channel < LatentChannels; channel++)
        {
            new ReadOnlySpan<float>(previous + ((long)channel * previousLength), overlap)
                .CopyTo(new Span<float>(target + ((long)channel * length), overlap));
        }
    }

    private static void Advance(Tensor latents, Tensor velocity, float dt)
    {
        Span<float> target = new Span<float>((float*)latents.DataPointer, (int)latents.Shape.ElementCount);
        ReadOnlySpan<float> step = new ReadOnlySpan<float>((float*)velocity.DataPointer, target.Length);
        for (int i = 0; i < target.Length; i++)
        {
            target[i] += dt * step[i];
        }
    }

    private static Tensor SliceLeading(Tensor latents, int overlap)
    {
        int length = (int)latents.Shape[2];
        Tensor slice = new Tensor(new TensorShape(1, LatentChannels, Math.Max(1, overlap)), DType.F32);
        if (overlap <= 0)
        {
            return slice;
        }
        float* source = (float*)latents.DataPointer;
        float* target = (float*)slice.DataPointer;
        for (int channel = 0; channel < LatentChannels; channel++)
        {
            new ReadOnlySpan<float>(source + ((long)channel * length), overlap)
                .CopyTo(new Span<float>(target + ((long)channel * overlap), overlap));
        }
        return slice;
    }

    private static Tensor SliceLatents(Tensor latents, int start, int end)
    {
        int length = (int)latents.Shape[2];
        int count = Math.Max(0, end - start);
        Tensor slice = new Tensor(new TensorShape(1, LatentChannels, Math.Max(1, count)), DType.F32);
        if (count == 0)
        {
            return slice;
        }
        float* source = (float*)latents.DataPointer;
        float* target = (float*)slice.DataPointer;
        for (int channel = 0; channel < LatentChannels; channel++)
        {
            new ReadOnlySpan<float>(source + ((long)channel * length) + start, count)
                .CopyTo(new Span<float>(target + ((long)channel * count), count));
        }
        return slice;
    }

    private static Tensor SliceCondition(Tensor condition, int start, int end)
    {
        int width = (int)condition.Shape[2];
        int count = Math.Max(0, end - start);
        Tensor slice = new Tensor(new TensorShape(1, Math.Max(1, count), width), DType.F32);
        if (count == 0)
        {
            return slice;
        }
        condition.AsReadOnlySpan<float>().Slice(start * width, count * width)
            .CopyTo(new Span<float>((float*)slice.DataPointer, count * width));
        return slice;
    }

    private static void SpliceCondition(Tensor condition, Tensor previousCondition, int overlap)
    {
        int width = (int)condition.Shape[2];
        previousCondition.AsReadOnlySpan<float>()[..(overlap * width)]
            .CopyTo(new Span<float>((float*)condition.DataPointer, overlap * width));
    }
}
