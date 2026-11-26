using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Pipelines.MovieRoll;

/// <summary>AnnealCFG's scale range: the scale runs from <see cref="Max"/> at the highest timestep to <see cref="Min"/> at zero.</summary>
public sealed record MovieRollCfg(float Min, float Max);

/// <summary>One sampling run's policy. The chunk layout is fixed for the run; the rest follow MovieRoll's streaming config.</summary>
public sealed record MovieRollSamplerOptions
{
    /// <summary>Latent frames in each chunk.</summary>
    public required int ChunkFrames { get; init; }

    /// <summary>Latent frames in the chunks that drain the queue after the last real chunk is enqueued; 0 means <see cref="ChunkFrames"/>.</summary>
    public int TailChunkFrames { get; init; }

    /// <summary>Latent frames in the first chunk only; 0 means <see cref="ChunkFrames"/>. Text-to-video uses <c>1 + ChunkFrames</c>
    /// (upstream <c>check_and_get_chunk_sizes</c>, mode <c>default</c>, as <c>movieroll_dmd_pipeline.py:740-765</c> and
    /// <c>movieroll_pipeline.py:991-992</c> call it). A value between 0 and <see cref="ChunkFrames"/> is rejected: that is the image-to-video
    /// layout, which needs a conditioning path this sampler does not have.</summary>
    public int FirstChunkFrames { get; init; }

    /// <summary>Real output chunks. The queue runs <c>NumChunks + window</c> rounds, so the drain chunks are denoised but never emitted.</summary>
    public required int NumChunks { get; init; }

    /// <summary>Blocks below this index keep their whole history; the rest are sink-recent.</summary>
    public int FullHistoryBlocks { get; init; }

    /// <summary>The sigma of the context noise on each clean chunk before it enters the cache, in [0, 1] (0 leaves the chunk clean).</summary>
    public float ContextNoise { get; init; }

    /// <summary>Whether a KV recompute may fire when the cache reaches <see cref="KvRecomputeThreshold"/> chunks.</summary>
    public bool KvRecomputeEnabled { get; init; }

    /// <summary>The cached chunk count that triggers a recompute; positive when recompute is enabled.</summary>
    public int KvRecomputeThreshold { get; init; } = 18;

    /// <summary>The most recent chunks a recompute keeps, besides the first and the middle chunk; not negative.</summary>
    public int KvRecomputeRecentK { get; init; } = 3;

    /// <summary>Classifier-free guidance; null runs the distilled chain (no guidance, one forward per step).</summary>
    public MovieRollCfg? Cfg { get; init; }
}

/// <summary>MovieRoll's rolling sampler for one prompt: a queue of up to <c>window</c> chunks, each at its own step of the ladder. Every round enqueues
/// a fresh noisy chunk, runs <c>StepsPerWindow</c> steps over the whole queue, and then, once the queue is full, emits the head: its clean latents are
/// cloned to the output, context-noised and encoded into the cache, or recomputed when the cache reaches its threshold. Per-chunk arithmetic keeps
/// one step index per chunk, so every chunk takes its own Euler step. This is the single-event path (one prompt, no prompt switch).</summary>
public sealed class MovieRollSampler
{
    private readonly IBackend _backend;
    private readonly HunyuanVideo15Dit _dit;
    private readonly MovieRollSchedule _schedule;

    public MovieRollSampler(IBackend backend, HunyuanVideo15Dit dit, MovieRollSchedule schedule)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _dit = dit ?? throw new ArgumentNullException(nameof(dit));
        _schedule = schedule ?? throw new ArgumentNullException(nameof(schedule));
    }

    /// <summary>Runs the rolling loop and returns the clean latents <c>[1, C, T, H, W]</c>, owned by the caller. <paramref name="noise"/> draws a
    /// noise tensor <c>[1, C, frames, H, W]</c>; the cache must be initialised for the latent grid and is left holding the run's history.</summary>
    public Tensor Sample(MovieRollKvCache cache, MovieRollSamplerOptions options, Tensor qwen, Tensor? byt5, Tensor? negativeQwen,
        Tensor? negativeByt5, Func<int, Tensor> noise)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(noise);
        if (!cache.IsInit) throw new InvalidOperationException("Init the cache with the DiT's stream layer names before sampling.");
        if (options.Cfg is not null && negativeQwen is null)
            throw new ArgumentException("Classifier-free guidance needs the negative prompt's text.", nameof(negativeQwen));
        if (options.ChunkFrames <= 0 || options.NumChunks <= 0 || options.TailChunkFrames < 0 || options.FirstChunkFrames < 0)
            throw new ArgumentException("Chunk and chunk count must be positive.", nameof(options));
        if (options.FirstChunkFrames > 0 && options.FirstChunkFrames < options.ChunkFrames)
            throw new ArgumentException("A first chunk shorter than the chunk is the image-to-video layout, which this sampler does not condition.", nameof(options));
        if (options.ContextNoise < 0f || options.ContextNoise > 1f)
            throw new ArgumentOutOfRangeException(nameof(options), options.ContextNoise, "The context-noise sigma lies in [0, 1].");
        if (options.KvRecomputeRecentK < 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.KvRecomputeRecentK, "The recent-chunk count is not negative.");
        if (options.KvRecomputeEnabled && options.KvRecomputeThreshold <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.KvRecomputeThreshold, "The recompute threshold is positive when recompute is enabled.");

        int window = _schedule.WindowSize;
        int tail = options.TailChunkFrames == 0 ? options.ChunkFrames : options.TailChunkFrames;
        int stepsPerWindow = _schedule.StepsPerWindow;
        List<Tensor> queue = new();
        List<Tensor> outputs = new();
        int currentFrame = 0;
        int decoded = 0;
        try
        {
            for (int round = 0; round < options.NumChunks + window; round++)
            {
                if (round >= window)
                {
                    Tensor clean = queue[0];
                    queue.RemoveAt(0);
                    try
                    {
                        outputs.Add(CopyOf(clean));
                        bool globalLast = decoded == options.NumChunks - 1;
                        if (!globalLast)
                            currentFrame = Emit(cache, options, outputs, clean, currentFrame, noise);
                    }
                    finally
                    {
                        clean.Dispose();
                    }
                    decoded++;
                    if (decoded >= options.NumChunks) break;
                }

                int frames = round == 0 && options.FirstChunkFrames > 0 ? options.FirstChunkFrames
                    : round < options.NumChunks ? options.ChunkFrames : tail;
                queue.Add(noise(frames));
                for (int step = 0; step < stepsPerWindow; step++)
                    StepQueue(cache, options, queue, round * stepsPerWindow + step, qwen, byt5, negativeQwen, negativeByt5);
            }
            return Concat(outputs);
        }
        finally
        {
            foreach (Tensor chunk in queue) chunk.Dispose();
            foreach (Tensor output in outputs) output.Dispose();
        }
    }

    /// <summary>Encodes the clean head into the cache, or recomputes the cache from the selected outputs, and returns the new current frame.</summary>
    private int Emit(MovieRollKvCache cache, MovieRollSamplerOptions options, List<Tensor> outputs, Tensor clean, int currentFrame,
        Func<int, Tensor> noise)
    {
        bool recompute = options.KvRecomputeEnabled && cache.ShouldRecompute(options.KvRecomputeThreshold);
        List<int> selected = recompute ? SelectRecomputeChunks(outputs.Count, options.KvRecomputeRecentK) : new List<int>();
        if (recompute && selected.Count < outputs.Count)
        {
            List<Tensor> picked = selected.Select(index => outputs[index]).ToList();
            int[] sizes = picked.Select(part => (int)part.Shape[2]).ToArray();
            using Tensor joined = Concat(picked);
            using Tensor noisy = ContextNoise(joined, options.ContextNoise, noise);
            return _dit.StreamRecompute(_backend, noisy, sizes, cache, options.FullHistoryBlocks);
        }

        int frames = (int)clean.Shape[2];
        using Tensor kvLatent = ContextNoise(clean, options.ContextNoise, noise);
        _dit.StreamEncode(_backend, kvLatent, new float[frames], cache, options.FullHistoryBlocks, currentFrame);
        return currentFrame + frames;
    }

    /// <summary>One global step over the queue: every chunk takes its own Euler step from its own ladder entry, with CFG when configured.</summary>
    private void StepQueue(MovieRollKvCache cache, MovieRollSamplerOptions options, List<Tensor> queue, int globalStep, Tensor qwen, Tensor? byt5,
        Tensor? negativeQwen, Tensor? negativeByt5)
    {
        int[] indices = _schedule.StepIndices(globalStep).Take(queue.Count).ToArray();
        int chunks = queue.Count;
        int[] frames = queue.Select(chunk => (int)chunk.Shape[2]).ToArray();
        float[] frameTimesteps = new float[frames.Sum()];
        int at = 0;
        for (int c = 0; c < chunks; c++)
            for (int f = 0; f < frames[c]; f++) frameTimesteps[at++] = _schedule.Timesteps[indices[c]];

        // The text stream takes the first frame's timestep, which is the queue's head (slot 0), as MovieRoll's rolling forward does.
        using Tensor window = Concat(queue);
        using Tensor velocity = _dit.StreamDenoise(_backend, window, frameTimesteps, frameTimesteps[0], qwen, byt5, cache, options.FullHistoryBlocks);
        Tensor? unconditional = options.Cfg is null
            ? null
            : _dit.StreamDenoise(_backend, window, frameTimesteps, frameTimesteps[0], negativeQwen!, negativeByt5, cache, options.FullHistoryBlocks);
        try
        {
            for (int c = 0; c < chunks; c++)
            {
                int start = FramesBefore(frames, c);
                Tensor conditional = Slice(velocity, start, frames[c]);
                Tensor? guided = null;
                Tensor? updated = null;
                try
                {
                    if (options.Cfg is null)
                    {
                        guided = conditional;
                    }
                    else
                    {
                        using Tensor uncond = Slice(unconditional!, start, frames[c]);
                        guided = Guide(uncond, conditional, CfgScale(options.Cfg, _schedule.Timesteps[indices[c]]));
                    }
                    float sigma = _schedule.Sigmas[indices[c]];
                    float next = _schedule.Sigmas[indices[c] + 1];
                    updated = new Tensor(queue[c].Shape, DType.F32);
                    _backend.AffineMix(updated, queue[c], guided, 1f, next - sigma);
                }
                catch
                {
                    updated?.Dispose();
                    throw;
                }
                finally
                {
                    conditional.Dispose();
                    guided?.Dispose();
                }
                queue[c].Dispose();
                queue[c] = updated!;
            }
        }
        finally
        {
            unconditional?.Dispose();
        }
    }

    /// <summary>AnnealCFG: the scale at a timestep, <c>Min + (Max − Min) · t / N_train</c>, with N_train 1000.</summary>
    private static float CfgScale(MovieRollCfg cfg, float timestep) => cfg.Min + (cfg.Max - cfg.Min) * timestep / 1000f;

    /// <summary>The guided velocity <c>uncond + s · (cond − uncond)</c>, as one affine mix.</summary>
    private Tensor Guide(Tensor uncond, Tensor cond, float scale)
    {
        Tensor output = new(cond.Shape, DType.F32);
        _backend.AffineMix(output, cond, uncond, scale, 1f - scale);
        return output;
    }

    /// <summary>The context-noised chunk <c>(1 − σ) · clean + σ · eps</c>. A sigma of 0 is a copy of the clean chunk, and draws no noise.</summary>
    private Tensor ContextNoise(Tensor clean, float sigma, Func<int, Tensor> noise)
    {
        if (sigma <= 0f) return CopyOf(clean);
        int frames = (int)clean.Shape[2];
        using Tensor eps = noise(frames);
        Tensor output = new(clean.Shape, DType.F32);
        _backend.AffineMix(output, clean, eps, 1f - sigma, sigma);
        return output;
    }

    /// <summary>MovieRoll's recompute selection with one event: the first chunk, the middle of the run, and the most recent <paramref name="recentK"/>.</summary>
    private static List<int> SelectRecomputeChunks(int total, int recentK)
    {
        SortedSet<int> indices = new() { 0, (total - 1) / 2 };
        for (int i = Math.Max(0, total - recentK); i < total; i++) indices.Add(i);
        return indices.ToList();
    }

    private static int FramesBefore(int[] frames, int chunk)
    {
        int total = 0;
        for (int c = 0; c < chunk; c++) total += frames[c];
        return total;
    }

    private Tensor Slice(Tensor source, int start, int frames)
    {
        Tensor output = new(new TensorShape([1L, source.Shape[1], frames, source.Shape[3], source.Shape[4]]), DType.F32);
        StridedCopies.SliceAxis(_backend, output, source, 2, start, frames);
        return output;
    }

    private Tensor CopyOf(Tensor source)
    {
        // A device-side copy of the whole tensor: CopyTo would read a device source back to the host.
        Tensor copy = new(source.Shape, DType.F32);
        StridedCopies.SliceAxis(_backend, copy, source, 0, 0, source.Shape[0]);
        return copy;
    }

    private Tensor Concat(IReadOnlyList<Tensor> parts)
    {
        long frames = 0;
        foreach (Tensor part in parts) frames += part.Shape[2];
        Tensor first = parts[0];
        Tensor joined = new(new TensorShape([1L, first.Shape[1], frames, first.Shape[3], first.Shape[4]]), DType.F32);
        _backend.Concat(joined, parts.ToArray(), 2);
        return joined;
    }
}
