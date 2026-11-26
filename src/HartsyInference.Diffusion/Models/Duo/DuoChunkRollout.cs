using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Chunk-by-chunk Duo generation on the H3 stack, one conversation. The reference and the shared Intent rows are
/// committed once; each block is then denoised against the committed history and its clean result is committed. The denoise
/// is the shared Euler loop, so the cached path and a full forward over the same history take the same schedule and noise.
/// CPU reference: it allocates per call and is for tiny-config self-consistency tests, not production.</summary>
internal static class DuoChunkRollout
{
    /// <summary>Commits the reference and the shared Intent rows. <paramref name="sharedText"/> holds the shared rows (description
    /// then cue, <c>TextDim</c> values each); <paramref name="referenceVideo"/> holds the reference latent rows.</summary>
    public static void Prefill(MiniMaxH3Transformer transformer, IBackend backend, DuoSampleInput sample, DuoRopePolicy rope,
        DuoStreamCache cache, float[] sharedText, float[] referenceVideo)
    {
        DuoLayout prefix = DuoLayoutBuilder.BuildPrefix(sample, rope);
        IReadOnlyList<DuoGroup> refiner = [SharedGroup(SharedRows(sample))];
        Forward(transformer, backend, prefix, refiner, sharedText, referenceVideo, null, [1f, 1f], cache, commit: true)
            .Dispose();
    }

    /// <summary>Denoises block <paramref name="block"/> against the committed history, then commits its clean media.
    /// <paramref name="bodyText"/> holds this block's body Intent rows. Returns the denoised video and audio that were committed.</summary>
    public static (float[] Video, float[] Audio) GenerateBlock(MiniMaxH3Transformer transformer, IBackend backend, DuoSampleInput sample,
        int block, DuoRopePolicy rope, DuoStreamCache cache, float[] sharedText, float[] bodyText, float[] noiseVideo, float[] noiseAudio,
        int steps, double shift)
    {
        float[] text = Concat(sharedText, bodyText);
        IReadOnlyList<DuoGroup> refiner = ChunkRefiner(sample, block);
        DuoLayout noisy = DuoLayoutBuilder.BuildChunk(sample, block, rope, clean: false);
        (float[] video, float[] audio) = DuoSampling.Sample(noiseVideo, noiseAudio, steps, shift, DuoSolver.Euler,
            (v, a, time) => Velocity(transformer, backend, noisy, refiner, text, v, a, time, cache), renoise: null);

        DuoLayout clean = DuoLayoutBuilder.BuildChunk(sample, block, rope, clean: true);
        Forward(transformer, backend, clean, refiner, text, video, audio, [1f, 1f], cache, commit: true).Dispose();
        return (video, audio);
    }

    /// <summary>The cache-free reference for <see cref="GenerateBlock"/>: one full forward over reference, shared rows, clean blocks
    /// <c>0..block-1</c> and block <paramref name="block"/>, with the same Euler loop. <paramref name="cleanVideo"/> and
    /// <paramref name="cleanAudio"/> hold the clean media of blocks <c>0..block-1</c> in layout order.</summary>
    public static (float[] Video, float[] Audio) FullBlock(MiniMaxH3Transformer transformer, IBackend backend, DuoSampleInput sample,
        int block, DuoRopePolicy rope, float[] sharedText, float[][] bodyTexts, float[] referenceVideo, float[] cleanVideo,
        float[] cleanAudio, float[] noiseVideo, float[] noiseAudio, int steps, double shift)
    {
        DuoLayout full = DuoLayoutBuilder.Build([Truncate(sample, block)], rope);
        int shared = SharedRows(sample);
        List<DuoGroup> refiner = [SharedGroup(shared)];
        List<float> text = [.. sharedText];
        int cursor = shared;
        for (int b = 0; b <= block; b++)
        {
            int n = sample.Blocks[b].BodyTextRows;
            if (n > 0)
            {
                refiner.Add(new DuoGroup(0, b, DuoGroupKind.Text, DuoModality.None, cursor, cursor + n));
            }
            text.AddRange(bodyTexts[b]);
            cursor += n;
        }
        // Layout order is reference, clean blocks 0..block-1, then the noisy block.
        float[] fixedVideo = Concat(referenceVideo, cleanVideo);
        float[] fixedAudio = cleanAudio;
        float[] textAll = [.. text];
        return DuoSampling.Sample(noiseVideo, noiseAudio, steps, shift, DuoSolver.Euler,
            (v, a, time) => FullVelocity(transformer, backend, full, refiner, textAll, fixedVideo, fixedAudio, v, a, time),
            renoise: null);
    }

    private static (float[] Video, float[] Audio) Velocity(MiniMaxH3Transformer transformer, IBackend backend, DuoLayout layout,
        IReadOnlyList<DuoGroup> refiner, float[] text, float[] video, float[] audio, float time, DuoStreamCache cache)
    {
        using Forwarded result = Forward(transformer, backend, layout, refiner, text, video, audio, [1f, time], cache, commit: false);
        return result.Read();
    }

    private static (float[] Video, float[] Audio) FullVelocity(MiniMaxH3Transformer transformer, IBackend backend, DuoLayout layout,
        IReadOnlyList<DuoGroup> refiner, float[] text, float[] fixedVideo, float[] fixedAudio, float[] video, float[] audio, float time)
    {
        float[] allVideo = Concat(fixedVideo, video);
        float[] allAudio = Concat(fixedAudio, audio);
        using Forwarded result = Forward(transformer, backend, layout, refiner, text, allVideo, allAudio, [1f, time], null, commit: false);
        (float[] v, float[] a) = result.Read();
        return (v[^video.Length..], a[^audio.Length..]);
    }

    private static Forwarded Forward(MiniMaxH3Transformer transformer, IBackend backend, DuoLayout layout,
        IReadOnlyList<DuoGroup> refiner, float[] text, float[] video, float[]? audio, float[] times, DuoStreamCache? cache, bool commit)
    {
        int textRows = refiner.Count == 0 ? 0 : refiner[^1].End;
        Tensor? textTensor = textRows == 0 ? null : HostTensor(textRows, text.Length / textRows, text);
        Tensor videoTensor = HostTensor(layout.VideoRows.Length, video.Length / layout.VideoRows.Length, video);
        Tensor? audioTensor = audio is null || layout.AudioRows.Length == 0
            ? null
            : HostTensor(layout.AudioRows.Length, audio.Length / layout.AudioRows.Length, audio);
        try
        {
            if (textTensor is null)
            {
                throw new ArgumentException("A Duo forward needs text rows: the shared Intent rows are always present.", nameof(refiner));
            }
            (Tensor v, Tensor? a) = cache is null
                ? transformer.ForwardDuo(backend, layout, DuoAttentionMode.Causal, refiner, textTensor, videoTensor, audioTensor, times)
                : transformer.ForwardDuoCached(backend, layout, DuoAttentionMode.Causal, refiner, textTensor, videoTensor, audioTensor,
                    times, cache, commit);
            return new Forwarded(v, a);
        }
        finally
        {
            textTensor?.Dispose();
            videoTensor.Dispose();
            audioTensor?.Dispose();
        }
    }

    private sealed class Forwarded(Tensor video, Tensor? audio) : IDisposable
    {
        public (float[] Video, float[] Audio) Read()
        {
            float[] v = video.AsSpan<float>().ToArray();
            float[] a = audio is null ? [] : audio.AsSpan<float>().ToArray();
            return (v, a);
        }

        public void Dispose()
        {
            video.Dispose();
            audio?.Dispose();
        }
    }

    private static Tensor HostTensor(int rows, int width, float[] values)
    {
        Tensor tensor = new(new TensorShape(rows, width), DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static DuoSampleInput Truncate(DuoSampleInput sample, int block) => sample with
    {
        Blocks = sample.Blocks.Take(block + 1).ToArray(),
        CleanBlocks = block,
        NoisyBlocks = [block],
    };

    private static int SharedRows(DuoSampleInput sample) => sample.SharedDescriptionRows + sample.SharedCueRows;

    private static DuoGroup SharedGroup(int rows) => new(0, -1, DuoGroupKind.Shared, DuoModality.None, 0, rows);

    private static IReadOnlyList<DuoGroup> ChunkRefiner(DuoSampleInput sample, int block)
    {
        int shared = SharedRows(sample), n = sample.Blocks[block].BodyTextRows;
        return n == 0
            ? [SharedGroup(shared)]
            : [SharedGroup(shared), new DuoGroup(0, block, DuoGroupKind.Text, DuoModality.None, shared, shared + n)];
    }

    private static float[] Concat(float[] a, float[] b)
    {
        float[] result = new float[a.Length + b.Length];
        a.CopyTo(result, 0);
        b.CopyTo(result, a.Length);
        return result;
    }
}
