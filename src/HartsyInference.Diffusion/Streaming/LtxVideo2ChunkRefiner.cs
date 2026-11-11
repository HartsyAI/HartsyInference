using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Streaming;

/// <summary>How an <see cref="LtxVideo2ChunkRefiner"/> runs.</summary>
public sealed record LtxVideo2ChunkRefinerOptions
{
    /// <summary>Latent rows of a frame.</summary>
    public required int LatentHeight { get; init; }

    /// <summary>Latent columns of a frame.</summary>
    public required int LatentWidth { get; init; }

    /// <summary>The video's frame rate; it scales the temporal rotation.</summary>
    public required double Fps { get; init; }

    /// <summary>The distilled sigma schedule, descending to 0.</summary>
    public IReadOnlyList<float> Sigmas { get; init; } = FlowRefineSchedule.Ltx2Stage2Sigmas;

    /// <summary>Leading frames kept as the permanent anchor.</summary>
    public int SinkFrames { get; init; } = 1;

    /// <summary>The self-attention window.</summary>
    public LtxVideo2ChunkWindow Window { get; init; } = new();

    /// <summary>Seed of the per-block noise.</summary>
    public long Seed { get; init; } = 42;

    /// <summary>Replaces the seeded noise of a block, by block index, with a host F32 <c>[1, C, T, H, W]</c> tensor borrowed for
    /// the call (for parity against a recorded run); a null result draws the seeded noise.</summary>
    public Func<int, Tensor?>? Noise { get; init; }

    /// <summary>Rounds the noise, the model's velocity and every stepped latent to BF16, as upstream's BF16 sampler holds them.</summary>
    public bool RoundToBf16 { get; init; } = true;

    /// <summary>Device bytes the block weights must leave free beyond the refiner's own carried keys and values: its
    /// activations, and whatever else shares the device while it is resident.</summary>
    public long HeadroomBytes { get; init; }

    /// <summary>Observes intermediates by name: per block <c>refine{k}_noise</c>, <c>refine{k}_start</c>,
    /// <c>refine{k}_velocity{step}</c> and <c>refine{k}_latent</c>, all host F32 <c>[tokens, C]</c> in (frame, row, column)
    /// order, borrowed for the call.</summary>
    public Action<string, Tensor>? Tap { get; init; }
}

/// <summary>The chunk-causal LTX-2 refiner (upstream SANA-WM <c>RefinerChunkRunner</c>) as an
/// <see cref="IStreamingLatentRefiner"/>. Each block of clean first-stage latents is re-noised to the schedule's first sigma
/// and walked back in a few distilled steps; every step's forward attends to the stream's sink and to the keys and values
/// of the frames already refined, and one more forward over the refined block at timestep 0 stages that block's keys and
/// values for the blocks after it.</summary>
/// <remarks>Owns the stream's <see cref="LtxVideo2ChunkState"/>, the projected caption and the placement of the
/// transformer's weights (<see cref="BlockStreamingScope"/>: whatever does not fit beside the carried state and the
/// headroom is streamed per forward). Borrows the transformer. Noise is seeded per block, so a block refined again after
/// a discard reproduces.</remarks>
public sealed class LtxVideo2ChunkRefiner : IStreamingLatentRefiner
{
    private const int NoiseSeedStride = 0x61C88647;

    private readonly IBackend _backend;
    private readonly LtxVideo2ChunkTransformer _transformer;
    private readonly LtxVideo2ChunkRefinerOptions _options;
    private readonly FlowRefineSchedule _schedule;
    private readonly LtxVideo2ChunkState _state;
    private readonly BlockStreamingScope _placement;
    private readonly Tensor _caption;
    private readonly int _channels, _height, _width;
    private int _blocks;
    private int _stagedFrames;
    private int _disposed;

    /// <summary>Places <paramref name="transformer"/> on <paramref name="backend"/> and binds it to one stream's caption.</summary>
    /// <param name="promptEmbeds">The text connector's output for the prompt, host F32 <c>[L, captionChannels]</c>, borrowed
    /// for the call.</param>
    public LtxVideo2ChunkRefiner(IBackend backend, LtxVideo2ChunkTransformer transformer, Tensor promptEmbeds,
        LtxVideo2ChunkRefinerOptions options)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(promptEmbeds);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(options.Sigmas);
        ArgumentNullException.ThrowIfNull(options.Window);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.LatentHeight, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.LatentWidth, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(options.HeadroomBytes);
        if (!(options.Fps > 0))
            throw new ArgumentOutOfRangeException(nameof(options), options.Fps, "The frame rate must be positive.");
        _backend = backend;
        _transformer = transformer;
        _options = options;
        _channels = transformer.Config.InChannels;
        _height = options.LatentHeight;
        _width = options.LatentWidth;
        _schedule = new FlowRefineSchedule(options.Sigmas.ToArray());
        int tokensPerFrame = _height * _width;
        LtxVideo2ChunkState? state = null;
        BlockStreamingScope? placement = null;
        Tensor? caption = null;
        try
        {
            state = new LtxVideo2ChunkState(backend, transformer.Config, tokensPerFrame, options.SinkFrames, options.Window);
            // The carried keys and values arrive over the first blocks, after the weights are placed: reserve them now.
            long carried = LtxVideo2ChunkState.EstimateBytes(transformer.Config, tokensPerFrame, options.SinkFrames, options.Window,
                LtxVideo2ChunkTransformer.ActivationDType(backend));
            RequireRoom(backend, carried, options.Window);
            placement = BlockStreamingScope.Open(new BlockStreamingOptions
            {
                Backend = backend,
                Denoiser = transformer,
                ModelName = "LTX-2 refiner",
                HeadroomBytes = checked(carried + options.HeadroomBytes),
            });
            using (Tensor embeds = Upload(promptEmbeds))
                caption = transformer.ProjectCaption(backend, embeds);
            backend.PinActivation(caption);
        }
        catch
        {
            caption?.Dispose();
            if (placement is not null)
            {
                backend.FreeWeights(placement.EnumerateResidentWeights());
                placement.Dispose();
            }

            state?.Dispose();
            throw;
        }

        _state = state;
        _placement = placement;
        _caption = caption;
    }

    /// <inheritdoc />
    public int SinkFrames => _state.SinkFrames;

    /// <inheritdoc />
    public bool HasSink => _state.SinkCaptured;

    /// <inheritdoc />
    public int RefinedFrames => _state.CommittedFrames;

    /// <inheritdoc />
    public int PassesPerBlock => _schedule.Steps + (_state.LastHistoryLayer >= 0 ? 1 : 0);

    /// <summary>Blocks refined and committed so far.</summary>
    public int RefinedBlocks => _blocks;

    /// <summary>Transformer blocks kept on the device for the whole stream; the rest are streamed per forward.</summary>
    public int ResidentBlocks => _placement.ResidentPrefixBlocks;

    /// <inheritdoc />
    public void SetSink(Tensor sinkLatents, CancellationToken cancel)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(sinkLatents);
        if (_state.SinkCaptured)
            throw new InvalidOperationException("The refiner's sink is already set.");
        int frames = RequireLatents(sinkLatents, nameof(sinkLatents));
        if (frames != _state.SinkFrames)
            throw new ArgumentException($"The sink is {_state.SinkFrames} latent frames; got {frames}.", nameof(sinkLatents));
        cancel.ThrowIfCancellationRequested();
        using Tensor host = Pack(sinkLatents, frames);
        using Tensor tokens = Upload(host);
        using LtxVideo2ChunkScope chunk = _transformer.OpenChunk(_backend, frames, _height, _width, _options.Fps, firstFrame: 0);
        try
        {
            _transformer.CaptureSink(_backend, tokens, _caption, chunk, _state);
            _placement.EndStep();
            _state.CommitSink();
        }
        catch
        {
            _state.Discard();
            throw;
        }
    }

    /// <inheritdoc />
    public Tensor Refine(Tensor latents, CancellationToken cancel, Action<int>? onPass = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(latents);
        if (!_state.SinkCaptured)
            throw new InvalidOperationException("The refiner needs its sink before the first block.");
        int frames = RequireLatents(latents, nameof(latents));
        Discard();
        cancel.ThrowIfCancellationRequested();
        int block = _blocks;
        bool round = _options.RoundToBf16;
        using Tensor x = Renoised(latents, frames, block);
        using LtxVideo2ChunkScope chunk = _transformer.OpenChunk(_backend, frames, _height, _width, _options.Fps, _state.NextFrame, _state);
        try
        {
            int pass = 0;
            float scale = _transformer.Config.TimestepScaleMultiplier;
            for (int step = 0; step < _schedule.Steps; step++)
            {
                cancel.ThrowIfCancellationRequested();
                using Tensor velocity = Velocity(x, _schedule.Timestep(step, scale), chunk);
                if (round)
                    RoundToBf16(velocity.AsSpan<float>());
                _options.Tap?.Invoke($"refine{block}_velocity{step}", velocity);
                _schedule.Step(x.AsSpan<float>(), velocity.AsReadOnlySpan<float>(), step, round);
                onPass?.Invoke(pass++);
            }

            _options.Tap?.Invoke($"refine{block}_latent", x);
            if (_state.LastHistoryLayer >= 0)
            {
                cancel.ThrowIfCancellationRequested();
                using Tensor tokens = Upload(x);
                _transformer.CaptureChunk(_backend, tokens, _caption, chunk, _state);
                _placement.EndStep();
                onPass?.Invoke(pass);
            }

            Tensor refined = Unpack(x, frames);
            _stagedFrames = frames;
            return refined;
        }
        catch
        {
            _state.Discard();
            throw;
        }
    }

    /// <inheritdoc />
    public void Commit()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_stagedFrames == 0)
            throw new InvalidOperationException("No refined block is waiting to be committed.");
        _state.Commit(_stagedFrames);
        _stagedFrames = 0;
        _blocks++;
    }

    /// <inheritdoc />
    public void Discard()
    {
        _state.Discard();
        _stagedFrames = 0;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        try
        {
            _backend.UnpinActivation(_caption);
            _caption.Dispose();
            _state.Dispose();
        }
        finally
        {
            try
            {
                // The scope leaves its resident blocks parked; this stream is their only reader.
                _backend.FreeWeights(_placement.EnumerateResidentWeights());
            }
            finally
            {
                _placement.Dispose();
            }
        }
    }

    // The block's starting point in token layout: (1 − σ₀)·clean + σ₀·noise.
    private Tensor Renoised(Tensor latents, int frames, int block)
    {
        TensorShape shape = latents.Shape;
        using Tensor clean = Pack(latents, frames);
        Tensor? drawn = null;
        try
        {
            Tensor noise;
            if (_options.Noise?.Invoke(block) is Tensor injected)
            {
                if (injected.DType != DType.F32 || injected.Shape != shape)
                    throw new ArgumentException($"The injected noise of refiner block {block} must be F32 {shape}; got "
                        + $"{injected.DType} {injected.Shape}.");
                noise = injected;
            }
            else
            {
                drawn = SeedGenerator.CreateNoise(shape, unchecked((int)_options.Seed + block * NoiseSeedStride));
                noise = drawn;
            }

            using Tensor packed = Pack(noise, frames);
            if (_options.RoundToBf16)
                RoundToBf16(packed.AsSpan<float>());
            _options.Tap?.Invoke($"refine{block}_noise", packed);
            Tensor x = new(packed.Shape, DType.F32, DeviceKind.Cpu);
            try
            {
                _schedule.Renoise(x.AsSpan<float>(), clean.AsReadOnlySpan<float>(), packed.AsReadOnlySpan<float>(), _options.RoundToBf16);
                _options.Tap?.Invoke($"refine{block}_start", x);
                return x;
            }
            catch
            {
                x.Dispose();
                throw;
            }
        }
        finally
        {
            drawn?.Dispose();
        }
    }

    // One prediction pass: the host tokens go up, the velocity comes back as a host tensor.
    private Tensor Velocity(Tensor x, float timestep, LtxVideo2ChunkScope chunk)
    {
        using Tensor tokens = Upload(x);
        using Tensor velocity = _transformer.Predict(_backend, tokens, timestep, _caption, chunk, _state);
        _placement.EndStep();
        Tensor host = new(velocity.Shape, DType.F32, DeviceKind.Cpu);
        try
        {
            _backend.CopyTo(host, velocity);
            return host;
        }
        catch
        {
            host.Dispose();
            throw;
        }
    }

    private int RequireLatents(Tensor latents, string name)
    {
        TensorShape shape = latents.Shape;
        if (latents.DType != DType.F32 || shape.Rank != 5 || shape[0] != 1 || shape[1] != _channels || shape[2] < 1
            || shape[3] != _height || shape[4] != _width)
            throw new ArgumentException($"The refiner takes F32 latents [1, {_channels}, T, {_height}, {_width}]; got "
                + $"{latents.DType} {shape}.", name);
        return (int)shape[2];
    }

    // [1, C, T, H, W] to token rows [T·H·W, C] in (frame, row, column) order (patch size 1).
    private Tensor Pack(Tensor latents, int frames)
    {
        int positions = frames * _height * _width;
        Tensor tokens = new(new TensorShape(positions, _channels), DType.F32, DeviceKind.Cpu);
        ReadOnlySpan<float> source = latents.AsReadOnlySpan<float>();
        Span<float> rows = tokens.AsSpan<float>();
        for (int c = 0; c < _channels; c++)
        {
            ReadOnlySpan<float> channel = source.Slice(c * positions, positions);
            for (int p = 0; p < positions; p++)
                rows[p * _channels + c] = channel[p];
        }

        return tokens;
    }

    private Tensor Unpack(Tensor tokens, int frames)
    {
        int positions = frames * _height * _width;
        Tensor latents = new(new TensorShape([1, _channels, frames, _height, _width]), DType.F32, DeviceKind.Cpu);
        ReadOnlySpan<float> rows = tokens.AsReadOnlySpan<float>();
        Span<float> destination = latents.AsSpan<float>();
        for (int c = 0; c < _channels; c++)
        {
            Span<float> channel = destination.Slice(c * positions, positions);
            for (int p = 0; p < positions; p++)
                channel[p] = rows[p * _channels + c];
        }

        return latents;
    }

    private Tensor Upload(Tensor host)
    {
        Tensor device = new(host.Shape, DType.F32, _backend.Device);
        try
        {
            _backend.CopyTo(device, host);
            return device;
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    // Weights can be streamed; the carried state cannot. A window the device cannot hold fails here, not several blocks in.
    private static void RequireRoom(IBackend backend, long carried, LtxVideo2ChunkWindow window)
    {
        if (backend.GetVramInfo().TotalBytes <= 0)
            return;
        backend.TrimMemoryPool();
        long free = backend.FreeMemoryBytes();
        if (carried > free)
            throw new InvalidOperationException($"The refiner's attention window of {window.MaxFrames} frames needs "
                + $"{carried / (double)(1L << 30):F1} GB of device memory once full and {free / (double)(1L << 30):F1} GB is free. "
                + "Use a smaller window, keep the history on fewer blocks, or run without the refiner.");
    }

    private static void RoundToBf16(Span<float> values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(values[i]));
    }
}
