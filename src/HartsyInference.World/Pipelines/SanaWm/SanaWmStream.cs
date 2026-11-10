using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Streaming;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Video;
using HartsyInference.World.Camera;
using HartsyInference.World.Sessions;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>A running SANA-WM stream: the resident streaming DiT and causal VAE decoder, the conditioning, and the stream's
/// position. Each <see cref="GenerateChunk"/> is one autoregressive step of upstream's <c>SelfForcingFlowEulerCamCtrl</c>:
/// the chunk's noise is denoised over the distilled schedule against the state of the earlier chunks, one more forward over
/// the clean chunk at timestep 0 writes the state the next chunk reads, and the causal VAE decodes the chunk's frames. The
/// seed frame's latent is frame 0 of chunk 0 and stays fixed. Values cross every forward rounded to BF16, as upstream's
/// BF16 sampler stores them.</summary>
/// <remarks><para>With a refiner, the chunk's latents after the seed frame pass through it between the DiT and the decoder
/// (upstream's stage 1 → refiner → decode, one chunk in each); the DiT's own state keeps the unrefined latents, and the seed
/// frame's latent, the refiner's sink, is decoded as it is.</para>
/// <para>Chunk 0 delivers <c>8·(T−1)+1</c> frames, the seed frame first; every later chunk delivers <c>8·T</c>. Owns the
/// component stages, the caption and the seed latent; disposing releases them.</para></remarks>
public sealed class SanaWmStream : IChunkedWorldStream
{
    private const int NoiseSeedStride = 0x61C88647;
    private const int LatentPlane = SanaWmPipeline.LatentHeight * SanaWmPipeline.LatentWidth;

    private readonly SanaWmStreamRequest _request;
    private readonly SelfForcingFlowSchedule _schedule;
    private readonly ChunkPlanner _planner;
    private readonly SanaWmStreamCamera _camera;
    private readonly SanaWmTextConditioning _text;
    private readonly Tensor _firstLatent;
    private readonly SanaWmStage<ISanaWmChunkDenoiser> _denoiser;
    private readonly SanaWmStage<IStreamingVideoDecoder> _decoder;
    private readonly SanaWmStage<IStreamingLatentRefiner>? _refiner;
    private readonly bool _guided;
    private bool _faulted;
    private int _disposed;

    internal SanaWmStream(SanaWmStreamRequest request, SelfForcingFlowSchedule schedule, ChunkPlanner planner, SanaWmStreamCamera camera,
        SanaWmTextConditioning text, Tensor firstLatent, SanaWmStage<ISanaWmChunkDenoiser> denoiser,
        SanaWmStage<IStreamingVideoDecoder> decoder, SanaWmStage<IStreamingLatentRefiner>? refiner = null)
    {
        _request = request;
        _schedule = schedule;
        _planner = planner;
        _camera = camera;
        _text = text;
        _firstLatent = firstLatent;
        _denoiser = denoiser;
        _decoder = decoder;
        _refiner = refiner;
        _guided = request.Guidance > 1.0;
    }

    /// <inheritdoc/>
    public int FrameWidth => SanaWmPipeline.Width;

    /// <inheritdoc/>
    public int FrameHeight => SanaWmPipeline.Height;

    /// <inheritdoc/>
    public int ChunksGenerated { get; private set; }

    /// <inheritdoc/>
    public int FramesGenerated { get; private set; }

    /// <summary>Latent frames generated so far.</summary>
    public int LatentFramesGenerated { get; private set; }

    /// <inheritdoc/>
    public int NextChunkFrames
    {
        get
        {
            if (Completed)
                return 0;
            int latentFrames = _planner.Span(ChunksGenerated).Length;
            int stride = SanaWmCameraConditioning.TemporalStride;
            return ChunksGenerated == 0 ? (latentFrames - 1) * stride + 1 : latentFrames * stride;
        }
    }

    /// <inheritdoc/>
    public bool Completed => _planner.ChunkCount is int chunks && ChunksGenerated >= chunks;

    /// <summary>Model forwards per chunk for one guidance branch: the DiT's denoising steps and its state-writing pass, then
    /// the refiner's passes when the stream has one.</summary>
    public int ForwardsPerChunk => _schedule.Steps + 1 + (_refiner?.Component.PassesPerBlock ?? 0);

    /// <summary>Whether the stream refines its latents before decoding them.</summary>
    public bool Refines => _refiner is not null;

    /// <inheritdoc/>
    public void SendAction(string action)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _camera.Send(action);
    }

    /// <inheritdoc/>
    public IReadOnlyList<VideoFrame> GenerateChunk(CancellationToken cancel, Action<GenerationProgress>? progress = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_faulted)
            throw new InvalidOperationException("This SANA-WM stream failed part-way through a chunk and cannot continue; open a new one.");
        if (Completed)
            throw new InvalidOperationException("This SANA-WM stream has generated every chunk it was opened for.");

        int chunk = ChunksGenerated;
        ChunkSpan span = _planner.Span(chunk);
        cancel.ThrowIfCancellationRequested();
        (float[] conditions, Tensor plucker) = _camera.Chunk(span.Start, span.Length);
        using (plucker)
        using (Tensor latents = InitialLatents(chunk, span.Length))
        {
            bool[] clean = new bool[span.Length];
            clean[0] = chunk == 0;
            ISanaWmChunkDenoiser denoiser = _denoiser.Component;
            IStreamingLatentRefiner? refiner = _refiner?.Component;
            Tensor? refined = null;
            try
            {
                try
                {
                    Denoise(denoiser, latents, conditions, plucker, clean, cancel, progress);
                    _request.Tap?.Invoke($"chunk{chunk}_latent", latents);
                    WriteState(denoiser, latents, conditions, plucker, progress);
                    // The refiner reads only the clean latents, so it runs before either model advances: a chunk cancelled
                    // here leaves both where they were.
                    if (refiner is not null)
                        refined = Refine(refiner, latents, chunk, cancel, progress);
                }
                catch
                {
                    denoiser.Discard();
                    refiner?.Discard();
                    throw;
                }

                // From here the models stand after this chunk: a failure leaves the decoder behind them, so the stream
                // cannot continue.
                try
                {
                    denoiser.Commit(span.Length);
                    if (refined is not null)
                        refiner!.Commit();
                    IReadOnlyList<VideoFrame> frames = Decode(refined ?? latents);
                    ChunksGenerated++;
                    LatentFramesGenerated += span.Length;
                    FramesGenerated += frames.Count;
                    return frames;
                }
                catch
                {
                    _faulted = true;
                    throw;
                }
            }
            finally
            {
                refined?.Dispose();
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        try
        {
            _decoder.Dispose();
        }
        finally
        {
            try
            {
                _refiner?.Dispose();
            }
            finally
            {
                try
                {
                    _denoiser.Dispose();
                }
                finally
                {
                    _firstLatent.Dispose();
                    _text.Dispose();
                }
            }
        }
    }

    // Seeded noise for the chunk, rounded to BF16; chunk 0 carries the seed frame's latent as its frame 0.
    private Tensor InitialLatents(int chunk, int latentFrames)
    {
        TensorShape shape = new([1, SanaWmPipeline.LatentChannels, latentFrames, SanaWmPipeline.LatentHeight, SanaWmPipeline.LatentWidth]);
        Tensor latents = new(shape, DType.F32, DeviceKind.Cpu);
        bool built = false;
        try
        {
            Span<float> values = latents.AsSpan<float>();
            if (_request.ChunkNoise?.Invoke(chunk) is Tensor injected)
            {
                if (injected.DType != DType.F32 || injected.Shape != shape)
                    throw new ArgumentException($"The injected noise of chunk {chunk} must be F32 {shape}; got {injected.DType} "
                        + $"{injected.Shape}.");
                injected.AsReadOnlySpan<float>().CopyTo(values);
            }
            else
            {
                using Tensor noise = SeedGenerator.CreateNoise(shape, unchecked((int)_request.Seed + chunk * NoiseSeedStride));
                noise.AsReadOnlySpan<float>().CopyTo(values);
            }

            SanaWmPipeline.RoundToBf16(values);
            if (chunk == 0)
            {
                ReadOnlySpan<float> first = _firstLatent.AsReadOnlySpan<float>();
                for (int c = 0; c < SanaWmPipeline.LatentChannels; c++)
                    first.Slice(c * LatentPlane, LatentPlane).CopyTo(values.Slice(c * latentFrames * LatentPlane, LatentPlane));
            }

            _request.Tap?.Invoke($"chunk{chunk}_noise", latents);
            built = true;
            return latents;
        }
        finally
        {
            if (!built)
                latents.Dispose();
        }
    }

    // The distilled schedule over the chunk: every pass reads the committed stream state and writes none.
    private void Denoise(ISanaWmChunkDenoiser denoiser, Tensor latents, float[] conditions, Tensor plucker, bool[] clean,
        CancellationToken cancel, Action<GenerationProgress>? progress)
    {
        float[] timesteps = new float[clean.Length];
        Stopwatch clock = Stopwatch.StartNew();
        for (int step = 0; step < _schedule.Steps; step++)
        {
            cancel.ThrowIfCancellationRequested();
            double started = clock.Elapsed.TotalMilliseconds;
            _schedule.FrameTimesteps(step, clean, timesteps);
            using Tensor velocity = Velocity(denoiser, latents, timesteps, conditions, plucker, commit: false);
            _schedule.Step(latents.AsSpan<float>(), velocity.AsReadOnlySpan<float>(), step, SanaWmPipeline.LatentChannels, clean,
                roundToBf16: true);
            progress?.Invoke(new GenerationProgress(step, ForwardsPerChunk, clock.Elapsed.TotalMilliseconds - started));
        }

        cancel.ThrowIfCancellationRequested();
    }

    // The forward over the clean chunk at timestep 0 that stages the state the next chunk reads; its prediction is unused.
    private void WriteState(ISanaWmChunkDenoiser denoiser, Tensor latents, float[] conditions, Tensor plucker,
        Action<GenerationProgress>? progress)
    {
        Stopwatch clock = Stopwatch.StartNew();
        float[] timesteps = new float[(int)latents.Shape[2]];
        Velocity(denoiser, latents, timesteps, conditions, plucker, commit: true).Dispose();
        progress?.Invoke(new GenerationProgress(_schedule.Steps, ForwardsPerChunk, clock.Elapsed.TotalMilliseconds));
    }

    // One forward per guidance branch in use, rounded to BF16 as the BF16 DiT emits it, combined when guided.
    private Tensor Velocity(ISanaWmChunkDenoiser denoiser, Tensor latents, float[] timesteps, float[] conditions, Tensor plucker,
        bool commit)
    {
        Tensor conditional = denoiser.Predict(latents, timesteps, SanaWmGuidanceBranch.Conditional, conditions, plucker, commit);
        try
        {
            Require(conditional, latents.Shape);
            if (!_guided)
            {
                SanaWmPipeline.RoundToBf16(conditional.AsSpan<float>());
                return conditional;
            }

            Tensor unconditional = denoiser.Predict(latents, timesteps, SanaWmGuidanceBranch.Unconditional, conditions, plucker, commit);
            try
            {
                Require(unconditional, latents.Shape);
                SanaWmPipeline.CombineGuidance(unconditional.AsSpan<float>(), conditional.AsReadOnlySpan<float>(), _request.Guidance);
                conditional.Dispose();
                return unconditional;
            }
            catch
            {
                unconditional.Dispose();
                throw;
            }
        }
        catch
        {
            conditional.Dispose();
            throw;
        }
    }

    // The chunk's latents with every frame after the refiner's sink refined: what the decoder receives. Returns null when
    // the chunk holds nothing to refine. The refiner's state is staged, not committed.
    private Tensor? Refine(IStreamingLatentRefiner refiner, Tensor latents, int chunk, CancellationToken cancel,
        Action<GenerationProgress>? progress)
    {
        int frames = (int)latents.Shape[2];
        // The stream's first frames are the refiner's sink: it holds them already and they pass through as they are.
        int kept = chunk == 0 ? Math.Min(refiner.SinkFrames, frames) : 0;
        if (chunk == 0 && kept != refiner.SinkFrames)
            throw new InvalidOperationException($"The first chunk holds {frames} latent frames; the refiner's sink takes "
                + $"{refiner.SinkFrames}.");
        if (frames == kept)
            return null;
        cancel.ThrowIfCancellationRequested();
        Stopwatch clock = Stopwatch.StartNew();
        double started = 0;
        int first = _schedule.Steps + 1;
        using Tensor block = FrameRange(latents, kept, frames - kept);
        using Tensor refinedBlock = refiner.Refine(block, cancel, pass =>
        {
            double now = clock.Elapsed.TotalMilliseconds;
            progress?.Invoke(new GenerationProgress(first + pass, ForwardsPerChunk, now - started));
            started = now;
        });
        if (refinedBlock.DType != DType.F32 || refinedBlock.Shape != block.Shape)
            throw new InvalidOperationException($"The refiner returned {refinedBlock.DType} {refinedBlock.Shape}; expected F32 "
                + $"{block.Shape}.");
        Tensor joined = new(latents.Shape, DType.F32, DeviceKind.Cpu);
        try
        {
            ReadOnlySpan<float> source = latents.AsReadOnlySpan<float>(), refinedValues = refinedBlock.AsReadOnlySpan<float>();
            Span<float> destination = joined.AsSpan<float>();
            int refinedFrames = frames - kept;
            for (int c = 0; c < SanaWmPipeline.LatentChannels; c++)
            {
                source.Slice(c * frames * LatentPlane, kept * LatentPlane).CopyTo(destination.Slice(c * frames * LatentPlane));
                refinedValues.Slice(c * refinedFrames * LatentPlane, refinedFrames * LatentPlane)
                    .CopyTo(destination.Slice((c * frames + kept) * LatentPlane));
            }

            _request.Tap?.Invoke($"chunk{chunk}_refined", joined);
            return joined;
        }
        catch
        {
            joined.Dispose();
            throw;
        }
    }

    // Frames [start, start + count) of [1, C, T, H, W] latents as a host tensor.
    private static Tensor FrameRange(Tensor latents, int start, int count)
    {
        int frames = (int)latents.Shape[2];
        Tensor range = new(new TensorShape([1, SanaWmPipeline.LatentChannels, count, SanaWmPipeline.LatentHeight, SanaWmPipeline.LatentWidth]),
            DType.F32, DeviceKind.Cpu);
        ReadOnlySpan<float> source = latents.AsReadOnlySpan<float>();
        Span<float> destination = range.AsSpan<float>();
        for (int c = 0; c < SanaWmPipeline.LatentChannels; c++)
            source.Slice((c * frames + start) * LatentPlane, count * LatentPlane).CopyTo(destination.Slice(c * count * LatentPlane));
        return range;
    }

    private IReadOnlyList<VideoFrame> Decode(Tensor latents)
    {
        IStreamingVideoDecoder decoder = _decoder.Component;
        int expected = decoder.PixelFrames((int)latents.Shape[2]);
        using Tensor rgb = decoder.DecodeNext(latents);
        TensorShape shape = new([1, 3, expected, FrameHeight, FrameWidth]);
        if (rgb.DType != DType.F32 || rgb.Shape != shape)
            throw new InvalidOperationException($"The streaming decoder returned {rgb.DType} {rgb.Shape}; expected F32 {shape}.");
        ReadOnlySpan<float> values = rgb.AsReadOnlySpan<float>();
        VideoFrame[] frames = new VideoFrame[expected];
        for (int f = 0; f < expected; f++)
            frames[f] = new VideoFrame(FramesGenerated + f, FrameWidth, FrameHeight, ToRgb24(values, expected, f, FrameHeight, FrameWidth));
        return frames;
    }

    /// <summary>Upstream's streaming conversion <c>clamp(127.5·x + 127.5, 0, 255).to(uint8)</c> in single precision: frame
    /// <paramref name="frame"/> of RGB <c>[1, 3, F, H, W]</c> as an RGB24 buffer.</summary>
    public static byte[] ToRgb24(ReadOnlySpan<float> rgb, int frames, int frame, int height, int width)
    {
        int plane = height * width;
        byte[] pixels = new byte[plane * 3];
        for (int c = 0; c < 3; c++)
        {
            ReadOnlySpan<float> channel = rgb.Slice((c * frames + frame) * plane, plane);
            for (int p = 0; p < plane; p++)
            {
                float scaled = 127.5f * channel[p] + 127.5f;
                pixels[p * 3 + c] = float.IsNaN(scaled) ? (byte)0 : (byte)MathF.Min(MathF.Max(scaled, 0f), 255f);
            }
        }
        return pixels;
    }

    private static void Require(Tensor velocity, TensorShape expected)
    {
        if (velocity.DType != DType.F32 || velocity.Shape != expected)
            throw new InvalidOperationException($"The denoiser returned {velocity.DType} {velocity.Shape}; expected F32 {expected}.");
    }
}
