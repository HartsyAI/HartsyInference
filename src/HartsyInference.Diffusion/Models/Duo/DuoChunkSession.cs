using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.Duo;

namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Chunk-by-chunk Duo generation on the H3 stack for one conversation. <see cref="Prefill"/> commits the reference and the shared
/// Intent rows once; <see cref="GenerateBlock"/> then denoises the next block against the committed history and commits its clean result.
/// Blocks must come in order from 0, and every input is length-checked against the layout before anything is computed. The denoise runs on
/// the backend with the Euler schedule, so the cached path and a full forward over the same history take the same schedule and noise.
/// The state stays device-resident across the Euler steps, and the clean block is read back once, before the eviction and the commit.
/// <see cref="FullBlock"/> is the dense host reference that the tests compare against.</summary>
internal sealed class DuoChunkSession
{
    private readonly MiniMaxH3Transformer _transformer;
    private readonly IBackend _backend;
    private readonly DuoSampleInput _sample;
    private readonly DuoRopePolicy _rope;
    private readonly DuoStreamCache _cache;
    private int _nextBlock = -1;
    private float[] _sharedText = [];
    private int _textWidth;
    private int _videoWidth;

    public DuoChunkSession(MiniMaxH3Transformer transformer, IBackend backend, DuoSampleInput sample, DuoRopePolicy rope, DuoStreamCache cache)
    {
        _transformer = transformer ?? throw new ArgumentNullException(nameof(transformer));
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _sample = sample ?? throw new ArgumentNullException(nameof(sample));
        _rope = rope;
        _cache = cache ?? throw new ArgumentNullException(nameof(cache));
    }

    /// <summary>The next block to generate, or -1 before <see cref="Prefill"/>.</summary>
    public int NextBlock => _nextBlock;

    /// <summary>Commits the reference and the shared Intent rows. <paramref name="sharedText"/> holds the shared rows (description then cue,
    /// one row of the text width each); <paramref name="referenceVideo"/> holds the reference latent rows. Runs once.</summary>
    public void Prefill(float[] sharedText, float[] referenceVideo)
    {
        ArgumentNullException.ThrowIfNull(sharedText);
        ArgumentNullException.ThrowIfNull(referenceVideo);
        if (_nextBlock >= 0)
        {
            throw new InvalidOperationException("The session is already prefilled.");
        }
        int sharedRows = SharedRows(_sample);
        int textWidth = RowWidth(sharedText, sharedRows, "shared text");
        int videoWidth = RowWidth(referenceVideo, _sample.Reference.VideoRows, "reference video");

        DuoLayout prefix = DuoLayoutBuilder.BuildPrefix(_sample, _rope);
        Forward(prefix, [SharedGroup(sharedRows)], sharedText, referenceVideo, null, [1f, 1f], commit: true).Dispose();
        _sharedText = (float[])sharedText.Clone();
        _textWidth = textWidth;
        _videoWidth = videoWidth;
        _nextBlock = 0;
    }

    /// <summary>Denoises block <paramref name="block"/> against the committed history, then commits its clean media. It must be the next
    /// block in order. <paramref name="bodyText"/> holds this block's body Intent rows. Returns the committed clean video and audio.</summary>
    public (float[] Video, float[] Audio) GenerateBlock(int block, float[] bodyText, float[] noiseVideo, float[] noiseAudio, int steps, double shift)
    {
        ArgumentNullException.ThrowIfNull(bodyText);
        ArgumentNullException.ThrowIfNull(noiseVideo);
        ArgumentNullException.ThrowIfNull(noiseAudio);
        if (_nextBlock < 0)
        {
            throw new InvalidOperationException("Prefill the session before generating a block.");
        }
        if (block != _nextBlock)
        {
            throw new InvalidOperationException($"Block {block} is out of order: the next block is {_nextBlock}.");
        }
        if ((uint)block >= (uint)_sample.Blocks.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(block), block, "The block is not in the sample.");
        }

        DuoBlockInput input = _sample.Blocks[block];
        int shared = SharedRows(_sample);
        int bodyRows = input.BodyTextRows;
        if (bodyText.Length != bodyRows * _textWidth)
        {
            throw new ArgumentException($"Block {block} body text has {bodyText.Length} values; expected {bodyRows} rows of {_textWidth}.", nameof(bodyText));
        }
        RequireRows(noiseVideo, input.Video.VideoRows, _videoWidth, "noise video", block);
        int audioRows = 2 * input.AudioSteps;
        bool audioOk = audioRows == 0 ? noiseAudio.Length == 0 : noiseAudio.Length > 0 && noiseAudio.Length % audioRows == 0;
        if (!audioOk)
        {
            throw new ArgumentException($"Block {block} noise audio has {noiseAudio.Length} values; expected {audioRows} rows.", nameof(noiseAudio));
        }

        // Refuse a block that cannot fit even after every clean block is dropped, before any work is done or anything is evicted.
        int committedRows = input.Video.VideoRows + audioRows;
        int evictable = _cache.Groups.Where(g => g.Kind == DuoGroupKind.Clean).Sum(g => g.End - g.Start);
        if (_cache.Length - evictable + committedRows > _cache.CapacityRows)
        {
            throw new InvalidOperationException(
                $"Block {block} commits {committedRows} rows; the reference and shared rows leave room for {_cache.CapacityRows - (_cache.Length - evictable)}.");
        }

        float[] text = Concat(_sharedText, bodyText);
        IReadOnlyList<DuoGroup> refiner = ChunkRefiner(block, shared, bodyRows);
        DuoLayout noisy = DuoLayoutBuilder.BuildChunk(_sample, block, _rope, clean: false);
        DuoLayout clean = DuoLayoutBuilder.BuildChunk(_sample, block, _rope, clean: true);

        // The denoise stays on the backend: text and noise are uploaded once, each step's velocity comes back as device tensors, and the
        // Euler update is two device ops. The host reads the clean block once, at the end.
        int textRows = refiner.Count == 0 ? 0 : refiner[^1].End;
        using Tensor textTensor = DuoDeviceWindow.Upload(new TensorShape(textRows, RowWidth(text, textRows, "text")), text);
        Tensor video = DuoDeviceWindow.Upload(new TensorShape(noisy.VideoRows.Length, RowWidth(noiseVideo, noisy.VideoRows.Length, "noise video")), noiseVideo);
        Tensor? audio = noisy.AudioRows.Length == 0 || noiseAudio.Length == 0
            ? null
            : DuoDeviceWindow.Upload(new TensorShape(noisy.AudioRows.Length, RowWidth(noiseAudio, noisy.AudioRows.Length, "noise audio")), noiseAudio);
        float[] schedule = DuoSchedule.Grid(steps, shift);
        try
        {
            for (int index = 0; index < steps; index++)
            {
                float t = schedule[index];
                float remaining = schedule[index + 1] - t;
                using Forwarded predicted = ForwardTensors(_transformer, _backend, noisy, refiner, textTensor, video, audio, [1f, t], _cache,
                    commit: false);
                Tensor nextVideo = Euler(video, predicted.Video, remaining);
                Tensor? nextAudio;
                try
                {
                    if ((audio is null) != (predicted.Audio is null))
                    {
                        throw new InvalidOperationException("The forward returned audio for one stream of the block and not the other.");
                    }
                    nextAudio = audio is null || predicted.Audio is null ? null : Euler(audio, predicted.Audio, remaining);
                }
                catch
                {
                    nextVideo.Dispose();
                    throw;
                }
                video.Dispose();
                audio?.Dispose();
                video = nextVideo;
                audio = nextAudio;
            }

            // The check comes before any change to the cache: a non-finite block is refused with the history exactly as it was.
            float[] outVideo = video.AsSpan<float>().ToArray();
            float[] outAudio = audio is null ? [] : audio.AsSpan<float>().ToArray();
            if (!IsFinite(outVideo) || !IsFinite(outAudio))
            {
                throw new InvalidOperationException("Nonfinite generated Duo latent.");
            }

            // The window slides: the oldest clean blocks go, reference and shared rows stay, so this block's rows fit.
            while (_cache.Length + committedRows > _cache.CapacityRows)
            {
                _cache.EvictOldestClean();
            }
            ForwardTensors(_transformer, _backend, clean, refiner, textTensor, video, audio, [1f, 1f], _cache, commit: true).Dispose();
            _nextBlock++;
            return (outVideo, outAudio);
        }
        finally
        {
            video.Dispose();
            audio?.Dispose();
        }
    }

    /// <summary>One Euler step on the device: <c>state + remaining · velocity</c>, the same arithmetic as <see cref="DuoSampler.Euler"/>.</summary>
    private Tensor Euler(Tensor state, Tensor velocity, float remaining)
    {
        Tensor next = new(state.Shape, DType.F32);
        Tensor scaled = new(velocity.Shape, DType.F32);
        try
        {
            _backend.Scale(scaled, velocity, remaining);
            _backend.Add(next, state, scaled);
        }
        catch
        {
            next.Dispose();
            throw;
        }
        finally
        {
            scaled.Dispose();
        }
        return next;
    }

    private static bool IsFinite(float[] values)
    {
        foreach (float value in values)
        {
            if (!float.IsFinite(value))
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>Reference for the session: one full forward over reference, shared rows, clean blocks <c>0..block-1</c> and block
    /// <paramref name="block"/>, with the same Euler loop and no cache. <paramref name="cleanVideo"/> and <paramref name="cleanAudio"/> hold the
    /// clean media of blocks <c>0..block-1</c> in layout order. Static, so the tests can run it against any history, right or wrong.</summary>
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

    private static (float[] Video, float[] Audio) FullVelocity(MiniMaxH3Transformer transformer, IBackend backend, DuoLayout layout,
        IReadOnlyList<DuoGroup> refiner, float[] text, float[] fixedVideo, float[] fixedAudio, float[] video, float[] audio, float time)
    {
        float[] allVideo = Concat(fixedVideo, video);
        float[] allAudio = Concat(fixedAudio, audio);
        using Forwarded result = Forward(transformer, backend, layout, refiner, text, allVideo, allAudio, [1f, time], cache: null, commit: false);
        (float[] v, float[] a) = result.Read();
        return (v[^video.Length..], a[^audio.Length..]);
    }

    private Forwarded Forward(DuoLayout layout, IReadOnlyList<DuoGroup> refiner, float[] text, float[] video, float[]? audio, float[] times,
        bool commit) => Forward(_transformer, _backend, layout, refiner, text, video, audio, times, _cache, commit);

    private static Forwarded Forward(MiniMaxH3Transformer transformer, IBackend backend, DuoLayout layout,
        IReadOnlyList<DuoGroup> refiner, float[] text, float[] video, float[]? audio, float[] times, DuoStreamCache? cache, bool commit)
    {
        int textRows = refiner.Count == 0 ? 0 : refiner[^1].End;
        if (textRows == 0)
        {
            throw new ArgumentException("A Duo forward needs text rows: the shared Intent rows are always present.", nameof(refiner));
        }
        Tensor textTensor = DuoDeviceWindow.Upload(new TensorShape(textRows, RowWidth(text, textRows, "text")), text);
        Tensor videoTensor = DuoDeviceWindow.Upload(new TensorShape(layout.VideoRows.Length, RowWidth(video, layout.VideoRows.Length, "video")), video);
        Tensor? audioTensor = audio is null || layout.AudioRows.Length == 0
            ? null
            : DuoDeviceWindow.Upload(new TensorShape(layout.AudioRows.Length, RowWidth(audio, layout.AudioRows.Length, "audio")), audio);
        try
        {
            return ForwardTensors(transformer, backend, layout, refiner, textTensor, videoTensor, audioTensor, times, cache, commit);
        }
        finally
        {
            textTensor.Dispose();
            videoTensor.Dispose();
            audioTensor?.Dispose();
        }
    }

    /// <summary>The forward over device tensors the caller owns. The outputs are new tensors, owned by the returned <see cref="Forwarded"/>.</summary>
    private static Forwarded ForwardTensors(MiniMaxH3Transformer transformer, IBackend backend, DuoLayout layout,
        IReadOnlyList<DuoGroup> refiner, Tensor text, Tensor video, Tensor? audio, float[] times, DuoStreamCache? cache, bool commit)
    {
        (Tensor v, Tensor? a) = cache is null
            ? transformer.ForwardDuo(backend, layout, DuoAttentionMode.Causal, refiner, text, video, audio, times)
            : transformer.ForwardDuoCached(backend, layout, DuoAttentionMode.Causal, refiner, text, video, audio, times, cache, commit);
        return new Forwarded(v, a);
    }

    private sealed class Forwarded(Tensor video, Tensor? audio) : IDisposable
    {
        public Tensor Video => video;

        public Tensor? Audio => audio;

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

    private static int RowWidth(float[] values, int rows, string what)
    {
        if (rows <= 0 || values.Length == 0 || values.Length % rows != 0)
        {
            throw new ArgumentException($"{what} has {values.Length} values; expected a positive multiple of {rows} rows.", what);
        }
        return values.Length / rows;
    }

    private static void RequireRows(float[] values, int rows, int width, string what, int block)
    {
        if (values.Length != rows * width)
        {
            throw new ArgumentException($"Block {block} {what} has {values.Length} values; expected {rows} rows of {width}.", what);
        }
    }


    private static DuoSampleInput Truncate(DuoSampleInput sample, int block) => sample with
    {
        Blocks = sample.Blocks.Take(block + 1).ToArray(),
        CleanBlocks = block,
        NoisyBlocks = [block],
    };

    private static int SharedRows(DuoSampleInput sample) => sample.SharedDescriptionRows + sample.SharedCueRows;

    private static DuoGroup SharedGroup(int rows) => new(0, -1, DuoGroupKind.Shared, DuoModality.None, 0, rows);

    private static IReadOnlyList<DuoGroup> ChunkRefiner(int block, int shared, int bodyRows) =>
        bodyRows == 0
            ? [SharedGroup(shared)]
            : [SharedGroup(shared), new DuoGroup(0, block, DuoGroupKind.Text, DuoModality.None, shared, shared + bodyRows)];

    private static float[] Concat(float[] a, float[] b)
    {
        float[] result = new float[a.Length + b.Length];
        a.CopyTo(result, 0);
        b.CopyTo(result, a.Length);
        return result;
    }
}
