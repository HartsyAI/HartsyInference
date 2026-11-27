using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.FastFrames;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>The numerical recipe of one released FastFrames model (arXiv 2611.23226; <c>FastFramesConfig</c> in
/// <c>fastframes/pipeline.py</c>, ml-fastframes a65c592): the timesteps of each frame's few denoising steps, the timestep the
/// KV refinement network is conditioned on, the flow shift and the latent geometry.</summary>
/// <param name="Name">Display name (<c>FF-S</c>, <c>FF-L</c>).</param>
/// <param name="FirstFrameTimesteps">Timesteps of the first frame's denoising steps (four of them: nothing precedes it).</param>
/// <param name="Timesteps">Timesteps of every later frame's denoising steps (two).</param>
/// <param name="RefineTimestep">The timestep the refinement network's embedding is built for.</param>
public sealed record FastFramesConfig(string Name, float[] FirstFrameTimesteps, float[] Timesteps, float RefineTimestep,
    int LatentChannels = 16, int LatentHeight = 60, int LatentWidth = 104, float Shift = 5f)
{
    /// <summary>FastFrames-S: the Causal Forcing++ two-step frame-wise generator (shifted timesteps of nominal 1000, 750, 500, 250).</summary>
    public static FastFramesConfig S { get; } = new("FF-S", [1000f, 937.5f, 833.3333129882812f, 625f], [1000f, 833.3333129882812f], 833.3333129882812f);

    /// <summary>FastFrames-L: the two-step model distilled from Wan2.1-14B.</summary>
    public static FastFramesConfig L { get; } = new("FF-L", [1000f, 937f, 833f, 625f], [1000f, 833f], 833f);

    /// <summary>Latent values in one frame (<c>channels · height · width</c>).</summary>
    public int FrameValues => LatentChannels * LatentHeight * LatentWidth;
}

/// <summary>The random inputs of a rollout, as plain <c>[channels · height · width]</c> arrays. Upstream draws them from a CUDA
/// generator in bf16, which no other implementation reproduces, so the pipeline takes them as inputs: a seeded stream for
/// ordinary generation (<see cref="FastFramesSeededNoise"/>), saved arrays for a parity run.</summary>
public interface IFastFramesNoise
{
    /// <summary>The noise frame <paramref name="frame"/> starts from.</summary>
    float[] Initial(int frame);

    /// <summary>The noise that re-noises frame <paramref name="frame"/> after denoising step <paramref name="step"/> (before the next step).</summary>
    float[] Transition(int step, int frame);
}

/// <summary>Seeded standard-normal noise from the engine's own generator: one stream per (kind, step, frame), so a frame's noise
/// does not depend on how many frames are generated.</summary>
public sealed class FastFramesSeededNoise(int seed, FastFramesConfig config) : IFastFramesNoise
{
    /// <inheritdoc/>
    public float[] Initial(int frame) => Draw(0, 0, frame);

    /// <inheritdoc/>
    public float[] Transition(int step, int frame) => Draw(1, step, frame);

    private float[] Draw(int kind, int step, int frame)
    {
        int derived = unchecked((((seed * 31 + kind) * 31 + step) * 1000003) + frame);
        using Tensor noise = SeedGenerator.CreateNoise(new TensorShape(config.FrameValues), derived);
        return noise.AsReadOnlySpan<float>().ToArray();
    }
}

/// <summary>One finished latent frame of a rollout.</summary>
/// <param name="Index">The latent frame's index.</param>
/// <param name="Latent">The final clean-latent estimate, <c>[channels, height, width]</c>.</param>
/// <param name="Pixels">The decoder's RGB frames for it, <c>[patchT, 3, 16·gridH, 16·gridW]</c> in about <c>[-1, 1]</c>, unclamped.</param>
public sealed record FastFramesFrame(int Index, float[] Latent, float[] Pixels, int PixelFrames, int PixelHeight, int PixelWidth);

/// <summary>FastFrames text-to-video generation (Sec. 3 of arXiv 2611.23226; <c>fastframes/pipeline.py</c>): each latent frame is
/// denoised in a few steps by the frame-wise causal generator against the per-layer KV cache of the frames before it; the pixel
/// decoder turns the last step's hidden states into RGB frames; then the KV refinement network replaces the frame's noisy cache
/// entries with clean-context ones (the first frame instead gets one clean generator pass at timestep 0, since nothing precedes it
/// to refine from). A frame is yielded as soon as it is decoded, so a consumer can stream; the cache work for that frame runs when
/// the consumer asks for the next one.
/// <para>The pipeline does not own its components, and the constructor checks that they fit together. Numerics are the engine's
/// (F32 residual stream, the Wan block's branch dtype); they differ from upstream's BF16 run by BF16 rounding, which is why parity
/// is judged against an FP32 run of the upstream code.</para></summary>
public sealed class FastFramesPipeline : IDisposable
{
    private readonly IBackend _backend;
    private bool _disposed;
    private readonly WanVideoTransformer _transformer;
    private readonly FastFramesKvRefiner _refiner;
    private readonly FastFramesPixelDecoder _decoder;
    private readonly FastFramesConfig _config;
    private readonly FastFramesFlowSchedule _schedule;
    private readonly int _gridHeight, _gridWidth;
    private readonly ResidentPrefixPin _generatorPin = new ResidentPrefixPin();
    private readonly ResidentWeights _refinerWeights;
    private readonly ResidentWeights _decoderWeights;

    /// <summary>Pairs a transformer, a refiner, a decoder and a recipe. Refuses a combination that cannot run, naming both sides, so
    /// the mismatch shows here rather than in the middle of a clip.</summary>
    /// <exception cref="ArgumentException">The components and the recipe disagree on channels, patching, layers or width.</exception>
    public FastFramesPipeline(IBackend backend, WanVideoTransformer transformer, FastFramesKvRefiner refiner,
        FastFramesPixelDecoder decoder, FastFramesConfig config)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(refiner);
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(config);
        WanVideoConfig wan = transformer.Config;
        if (wan.PatchSize.T != 1)
            throw new ArgumentException($"FastFrames denoises one latent frame per pass; the transformer's temporal patch is {wan.PatchSize.T}.", nameof(transformer));
        if (config.LatentChannels != wan.InChannels)
            throw new ArgumentException($"{config.Name} has {config.LatentChannels} latent channels; the transformer takes {wan.InChannels}.", nameof(config));
        if (config.LatentHeight % wan.PatchSize.H != 0 || config.LatentWidth % wan.PatchSize.W != 0)
            throw new ArgumentException($"{config.Name}'s latent {config.LatentHeight}x{config.LatentWidth} is not a whole number of {wan.PatchSize.H}x{wan.PatchSize.W} patches.", nameof(config));
        if (refiner.Config.Layers != transformer.BlockCount)
            throw new ArgumentException($"The refiner has {refiner.Config.Layers} layers; the transformer has {transformer.BlockCount}.", nameof(refiner));
        if (refiner.Config.Dim != wan.InnerDim)
            throw new ArgumentException($"The refiner is {refiner.Config.Dim} wide; the transformer is {wan.InnerDim}.", nameof(refiner));
        if (decoder.Config.InputDim != wan.InnerDim)
            throw new ArgumentException($"The decoder reads {decoder.Config.InputDim}-wide hidden states; the transformer is {wan.InnerDim} wide.", nameof(decoder));

        _backend = backend;
        _transformer = transformer;
        _refiner = refiner;
        _decoder = decoder;
        _config = config;
        _schedule = new FastFramesFlowSchedule(config.Shift);
        _gridHeight = config.LatentHeight / wan.PatchSize.H;
        _gridWidth = config.LatentWidth / wan.PatchSize.W;
        _refinerWeights = new ResidentWeights(backend, refiner.EnumerateWeights);
        _decoderWeights = new ResidentWeights(backend, decoder.EnumerateWeights);
    }

    /// <summary>The first <c>patchT − 1</c> pixel frames of the clip are dropped (upstream's <c>align_frames</c>: 84 decoded frames to 81).</summary>
    public int LeadingPixelFramesDropped => _decoder.Config.PatchT - 1;

    /// <summary>Pixel frames a clip of <paramref name="latentFrames"/> latent frames has after the leading ones are dropped.</summary>
    public int PixelFrameCount(int latentFrames) => latentFrames * _decoder.Config.PatchT - LeadingPixelFramesDropped;

    /// <summary>Runs the rollout and yields each latent frame once it is decoded. <paramref name="promptEmbeds"/> is the umT5
    /// output zero-padded to the generator's text length (<c>[512, 4096]</c> for Wan), as <c>WanVideoTransformer</c> takes it.
    /// The arguments are checked when this is called, not when the first frame is requested.</summary>
    public IEnumerable<FastFramesFrame> Rollout(Tensor promptEmbeds, IFastFramesNoise noise, int latentFrames, Action<int, int>? progress = null,
        CancellationToken cancel = default)
    {
        ValidateRequest(promptEmbeds, noise, latentFrames);
        return RolloutCore(promptEmbeds, noise, latentFrames, progress, cancel);
    }

    /// <summary>Generates a clip and yields its frames as 8-bit RGB (<c>[H, W, 3]</c>, top row first), the leading frames dropped, as
    /// the frames decode. The arguments are checked when this is called.</summary>
    public IEnumerable<VideoFrame> GenerateFrames(Tensor promptEmbeds, IFastFramesNoise noise, int latentFrames, Action<int, int>? progress = null,
        CancellationToken cancel = default)
    {
        ValidateRequest(promptEmbeds, noise, latentFrames);
        return GenerateCore(promptEmbeds, noise, latentFrames, progress, cancel);
    }

    private void ValidateRequest(Tensor promptEmbeds, IFastFramesNoise noise, int latentFrames)
    {
        ArgumentNullException.ThrowIfNull(promptEmbeds);
        ArgumentNullException.ThrowIfNull(noise);
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfLessThan(latentFrames, 1);
        // A pinned prefix from an earlier clip is still on the device, so the free VRAM read here does not show what the placement
        // will have free. The placement releases or keeps that prefix; the check after it decides, so skip this one.
        // Auto (null) decides the mode in the rollout and checks that mode there; only a forced resident cache is refused at admission.
        if (!_generatorPin.Resident && OffloadFrameCache == false)
            CheckKvFitsNow(latentFrames, offload: false);
    }

    /// <summary>The frame-cache need of a clip of <paramref name="latentFrames"/> on this pipeline: the cache in the slot dtype, plus one
    /// block's joined prefix. Used by both checks and by the block placement's headroom, so they agree.</summary>
    private long ClipKvNeedBytes(int latentFrames) =>
        KvNeedBytes(_transformer.BlockCount, latentFrames, _gridHeight * _gridWidth, _transformer.Config.InnerDim,
            _transformer.BranchActivation(_backend, singleGroup: true));

    /// <summary>The device bytes of a clip whose frame cache is offloaded: one block's cache and its join, the most a block's attention holds.</summary>
    private long OffloadedKvNeedBytes(int latentFrames) =>
        KvNeedBytes(1, latentFrames, _gridHeight * _gridWidth, _transformer.Config.InnerDim, _transformer.BranchActivation(_backend, singleGroup: true));

    /// <summary>The frame cache's device need in the mode the clip runs in: the whole cache, or one block's share when offloaded.</summary>
    private long KvNeedFor(int latentFrames, bool offload) => offload ? OffloadedKvNeedBytes(latentFrames) : ClipKvNeedBytes(latentFrames);

    /// <summary>Whether this clip's frame cache lives on the host. Forced by <see cref="OffloadFrameCache"/>; in auto mode, when the whole cache
    /// does not fit the free VRAM now. A backend that reports no VRAM is not offloaded.</summary>
    private bool ResolveFrameCacheOffload(int latentFrames)
    {
        if (OffloadFrameCache is bool forced)
            return forced;
        (long freeBytes, long totalBytes) = _backend.GetVramInfo();
        return totalBytes > 0 && ClipKvNeedBytes(latentFrames) > freeBytes;
    }

    /// <summary>Checks the clip's frame cache, in the mode it runs in, against the free VRAM the backend reports now. Called at admission and
    /// again once the block placement has run, so the weights it uploaded are counted as used.</summary>
    private void CheckKvFitsNow(int latentFrames, bool offload)
    {
        (long freeBytes, long totalBytes) = _backend.GetVramInfo();
        DType slot = _transformer.BranchActivation(_backend, singleGroup: true);
        CheckFits(_config.Name, KvNeedFor(latentFrames, offload), latentFrames, slot, freeBytes, totalBytes);
    }

    /// <summary>Refuses a clip whose frame-wise cache cannot fit in the free VRAM, before any frame is denoised. The cache keeps every
    /// frame's keys and values for every block until the clip ends, in the dtype the slots take (<paramref name="slot"/>: the attention
    /// branch's activation, F32 when F16 branches are off or the backend lacks F16 activations). It also reserves one block's joined
    /// prefix, the copy the rewrite makes per block. A backend that reports no VRAM (CPU) is not checked. This is a lower bound: the
    /// attention scratch, the decoder and the refiner come on top.</summary>
    /// <exception cref="InvalidOperationException">The cache for <paramref name="latentFrames"/> does not fit in <paramref name="freeBytes"/>.</exception>
    public static void CheckKvFits(string name, int blocks, int latentFrames, int tokensPerFrame, int hidden, DType slot,
        long freeBytes, long totalBytes) =>
        CheckFits(name, KvNeedBytes(blocks, latentFrames, tokensPerFrame, hidden, slot), latentFrames, slot, freeBytes, totalBytes);

    /// <summary>The frame cache of a clip in <paramref name="slot"/> plus one block's joined prefix (the copy the rewrite makes per
    /// block): the bytes the clip needs before attention scratch, the decoder and the refiner.</summary>
    public static long KvNeedBytes(int blocks, int latentFrames, int tokensPerFrame, int hidden, DType slot) =>
        FrameKvCache.EstimateBytes(blocks, latentFrames, tokensPerFrame, hidden, slot)
        + FrameKvCache.EstimateBytes(1, latentFrames, tokensPerFrame, hidden, slot);

    private static void CheckFits(string name, long need, int latentFrames, DType slot, long freeBytes, long totalBytes)
    {
        if (totalBytes <= 0)
            return;
        if (need > freeBytes)
        {
            throw new InvalidOperationException(
                $"{name} needs {need >> 20} MiB of frame KV cache ({slot}) for {latentFrames} latent frames; the device has {freeBytes >> 20} MiB free "
                + $"of {totalBytes >> 20} MiB. Generate fewer frames, or free VRAM first.");
        }
    }

    private IEnumerable<FastFramesFrame> RolloutCore(Tensor promptEmbeds, IFastFramesNoise noise, int latentFrames,
        Action<int, int>? progress, CancellationToken cancel)
    {
        // The body runs on the first MoveNext, so a rollout enumerated after Dispose is refused here.
        ObjectDisposedException.ThrowIf(_disposed, this);
        // The mode is decided here, before the placement: an offloaded cache keeps only the slots a block's join stages on the device.
        bool offload = ResolveFrameCacheOffload(latentFrames);
        using FrameKvCache cache = new(_backend, _transformer.BlockCount, latentFrames, offload);
        using FastFramesPixelDecoder.State decoderState = _decoder.CreateState();
        using FastFramesKvRefiner.Conditioning conditioning = _refiner.Prepare(_backend, _config.RefineTimestep);
        // The block placement reserves this clip's frame-cache need, in the slot dtype, so the planner keeps it free before it decides how
        // many blocks stay resident. The same estimate backs the check after the placement. On a CPU backend the scope is resident-only.
        long kvNeed = KvNeedFor(latentFrames, offload);
        // The refiner and the decoder stay on the device for the clip. Uploading them before the placement lets the planner size the
        // prefix beside them, so the headroom only has to cover activations. A failure from here to the first frame releases them and
        // the generator, so a refused clip leaves nothing it uploaded on the device.
        BlockStreamingScope? placed = null;
        try
        {
            _refinerWeights.Preload();
            _decoderWeights.Preload();
            // The pin keeps the resident prefix across clips on this pipeline. A longer clip needs a bigger cache reservation, so the
            // prefix is sized again then; a shorter or equal one reuses it. TokenLoad is the clip's token count, as Wan's pin takes it.
            placed = BlockStreamingScope.Open(new BlockStreamingOptions
            {
                Backend = _backend,
                Denoiser = _transformer,
                ModelName = _config.Name,
                HeadroomBytes = StreamingHeadroomOverride ?? kvNeed + DefaultStreamingHeadroomBytes,
                Mode = StreamingMode,
                Pin = _generatorPin,
                TokenLoad = (long)latentFrames * _gridHeight * _gridWidth,
            });
            LastStreamedBlocks = placed.StreamedBlocks;
            // Free VRAM is read again now that the placement has uploaded its prefix; the admission check ran before that upload.
            CheckKvFitsNow(latentFrames, offload);
        }
        catch
        {
            placed?.Dispose();
            FreeGenerator();
            _refinerWeights.Release(force: true);
            _decoderWeights.Release(force: true);
            throw;
        }
        BlockStreamingScope streaming = placed!;
        bool streamed = streaming.StreamedBlocks > 0;
        try
        {
            for (int frame = 0; frame < latentFrames; frame++)
            {
                cancel.ThrowIfCancellationRequested();
                // Refuses a frame started after Dispose. A Dispose while the consumer held the previous frame is caught after the yield below.
                ObjectDisposedException.ThrowIf(_disposed, this);
                (float[] latent, Tensor pixels) = DenoiseAndDecode(promptEmbeds, noise, frame, cache, decoderState);
                FastFramesFrame output;
                using (pixels)
                    output = new FastFramesFrame(frame, latent, pixels.AsReadOnlySpan<float>().ToArray(), _decoder.Config.PatchT,
                        (int)pixels.Shape[2], (int)pixels.Shape[3]);
                yield return output;

                // Disposed while the consumer held the frame: stop before any cache or weight work runs on freed weights. The throw
                // unwinds through the finally above, which frees the generator, so GeneratorResident is false afterwards.
                ObjectDisposedException.ThrowIf(_disposed, this);
                // The next frame reads this frame's cache slot, so the clean pass or the refinement runs before it. It runs after the
                // consumer has the frame, so a frame is not held back by the cache work of the one after it.
                if (frame == 0)
                    CleanPass(promptEmbeds, latent, frame, cache);
                else
                    Refine(latent, frame, cache, conditioning);
                streaming.EndStep();
                progress?.Invoke(frame + 1, latentFrames);
            }
        }
        finally
        {
            // Disposing the scope evicts the streamed window; the release below then decides what the next clip finds on the device.
            streaming.Dispose();
            ReleaseOrKeepGenerator(streamed);
        }
    }

    /// <summary>After a clip. A streamed clip frees the generator's device weights, the resident prefix included, so the next clip and the
    /// decoder get the VRAM back. The pin keeps the count it was sized to, so the next placement uploads that prefix only as far as it
    /// fits. A clip that stayed resident keeps the weights, and the pin says so, which lets the next clip skip the admission check the
    /// weights already passed. A disposed pipeline frees instead of keeping, so nothing stays on the device after Dispose.
    /// Mirrors WanVideoPipeline.ReleaseOrKeepTransformer.</summary>
    private void ReleaseOrKeepGenerator(bool streamed)
    {
        // After a streamed clip the refiner and the decoder follow the VRAM profile, as the generator's own release does not.
        if (streamed)
        {
            _refinerWeights.Release();
            _decoderWeights.Release();
        }
        if (streamed || _disposed)
        {
            FreeGenerator();
            return;
        }
        _generatorPin.Resident = true;
    }

    /// <summary>Bytes the block placement keeps free beyond the clip's frame cache: activations, the refiner and the decoder. When
    /// null, <see cref="DefaultStreamingHeadroomBytes"/> is used.</summary>
    public long? StreamingHeadroomOverride { get; set; }

    /// <summary>The low-VRAM policy the placement runs under; null resolves from the process setting. ForceOn streams even when everything fits
    /// (for tests); ForceOff keeps every block resident.</summary>
    public LowVramMode? StreamingMode { get; set; }

    /// <summary>Where a clip's frame cache lives: null (auto) offloads it to host memory when the whole cache does not fit the free VRAM
    /// (each written slot leaves the device; a block's join stages back what it reads), true always offloads, false keeps it resident and refuses
    /// a clip that does not fit. The output is the same in every mode.</summary>
    public bool? OffloadFrameCache { get; set; }

    /// <summary>Blocks the last clip streamed (the generator's block count when every block streams). Zero before the first clip.</summary>
    public int LastStreamedBlocks { get; private set; }

    /// <summary>True while the generator's weights stay on the device from the last clip, the resident path that <c>ReleaseOrKeepGenerator</c> keeps.</summary>
    internal bool GeneratorResident => _generatorPin.Resident;

    /// <summary>Frees the device copies of the generator, the refiner and the decoder, so a dropped pipeline gives the VRAM back. The pipeline
    /// does not own the backend or the transformer, refiner and decoder objects; their owner disposes those. Not thread-safe:
    /// call it on the thread that enumerates the clips, or after they finish. To abort a clip, cancel its token; a Dispose during a clip
    /// stops it at the next frame boundary.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        if (_generatorPin.Resident)
        {
            FreeGenerator();
        }
        _refinerWeights.Release(force: true);
        _decoderWeights.Release(force: true);
        _backend.TrimMemoryPool();
    }

    /// <summary>Frees every generator weight on the device and marks the pin not resident.</summary>
    private void FreeGenerator()
    {
        _backend.FreeWeights(_transformer.EnumerateWeights());
        _backend.TrimMemoryPool();
        _generatorPin.Resident = false;
    }

    /// <summary>The activation allowance on top of the frame cache when the placement is planned. The refiner and the decoder are not in it:
    /// they are uploaded before the placement, so the planner counts them as used.</summary>
    public const long DefaultStreamingHeadroomBytes = 1L << 30;

    private IEnumerable<VideoFrame> GenerateCore(Tensor promptEmbeds, IFastFramesNoise noise, int latentFrames,
        Action<int, int>? progress, CancellationToken cancel)
    {
        int outIndex = 0;
        int skip = LeadingPixelFramesDropped;
        foreach (FastFramesFrame frame in RolloutCore(promptEmbeds, noise, latentFrames, progress, cancel))
        {
            for (int p = 0; p < frame.PixelFrames; p++)
            {
                if (skip > 0)
                {
                    skip--;
                    continue;
                }
                yield return new VideoFrame(outIndex++, frame.PixelWidth, frame.PixelHeight, ToRgb8(frame, p));
            }
        }
    }

    /// <summary>One latent frame's denoising steps, then its decode. Returns the final clean-latent estimate and the decoded pixels,
    /// which the caller owns. The last step's hidden states are released on every path, including a throw.</summary>
    private (float[] Latent, Tensor Pixels) DenoiseAndDecode(Tensor promptEmbeds, IFastFramesNoise noise, int frame,
        FrameKvCache cache, FastFramesPixelDecoder.State decoderState)
    {
        float[] schedule = frame == 0 ? _config.FirstFrameTimesteps : _config.Timesteps;
        float[] x = (float[])noise.Initial(frame).Clone();   // x is re-noised in place below
        float[] x0 = new float[x.Length];
        for (int step = 0; step < schedule.Length; step++)
        {
            bool last = step + 1 == schedule.Length;
            WanVideoTransformer.FrameForward forward;
            using (Tensor input = FrameTensor(x))
                forward = _transformer.ForwardFrame(_backend, input, promptEmbeds, schedule[step], frame, cache);
            try
            {
                double sigma = _schedule.Sigma(schedule[step]);
                using (forward.Velocity)
                {
                    ReadOnlySpan<float> velocity = forward.Velocity.AsReadOnlySpan<float>();
                    for (int i = 0; i < x.Length; i++)
                        x0[i] = (float)(x[i] - sigma * velocity[i]);   // x0 = x_t - sigma · flow, in double
                }
                if (last)
                    return (x0, _decoder.DecodeFrame(_backend, forward.Hidden, forward.Temb, _gridHeight, _gridWidth, decoderState));
                double next = _schedule.Sigma(schedule[step + 1]);
                float[] eps = noise.Transition(step, frame);
                for (int i = 0; i < x.Length; i++)
                    x[i] = (float)((1 - next) * x0[i] + next * eps[i]);
            }
            finally
            {
                forward.Hidden.Dispose();
                forward.Temb.Dispose();
            }
        }
        throw new InvalidOperationException("A frame schedule has no steps.");
    }

    /// <summary>One generator pass over the clean frame at timestep 0, which leaves the frame's clean keys and values in its slot.</summary>
    private void CleanPass(Tensor promptEmbeds, float[] latent, int frame, FrameKvCache cache)
    {
        using Tensor input = FrameTensor(latent);
        WanVideoTransformer.FrameForward forward = _transformer.ForwardFrame(_backend, input, promptEmbeds, 0f, frame, cache);
        forward.Velocity.Dispose();
        forward.Hidden.Dispose();
        forward.Temb.Dispose();
    }

    /// <summary>Replaces the frame's noisy cache entries in every layer with the refinement network's clean-context ones.</summary>
    private void Refine(float[] latent, int frame, FrameKvCache cache, FastFramesKvRefiner.Conditioning conditioning)
    {
        using Tensor input = FrameTensor(latent);
        using Tensor stem = _transformer.PatchEmbedFrame(_backend, input);
        using Tensor phi = _refiner.ComputePhi(_backend, stem, conditioning);
        using Tensor phiV = _refiner.ProjectPhi(_backend, phi);
        for (int layer = 0; layer < cache.Layers; layer++)
            RefineLayer(layer, frame, cache, phiV, conditioning);
    }

    /// <summary>One layer's slot: the refinement in F32, converted back to the slot's dtype, then stored. Whatever is owned here is
    /// released on every path; <see cref="FrameKvCache.Set"/> takes the refined pair once it is stored.</summary>
    private void RefineLayer(int layer, int frame, FrameKvCache cache, Tensor phiV, FastFramesKvRefiner.Conditioning conditioning)
    {
        Tensor noisyK = cache.Keys(layer, frame), noisyV = cache.Values(layer, frame);
        DType slotType = noisyK.DType;
        bool converted = slotType != DType.F32;
        Tensor? k32 = converted ? noisyK.CastTo(DType.F32) : null;
        Tensor? v32 = converted ? noisyV.CastTo(DType.F32) : null;
        Tensor? refinedK = null, refinedV = null;
        try
        {
            (refinedK, refinedV) = _refiner.RepairLayer(_backend, layer, k32 ?? noisyK, v32 ?? noisyV, phiV, conditioning);
            if (converted)
            {
                Tensor k16 = refinedK.CastTo(slotType);
                refinedK.Dispose();
                refinedK = k16;
                Tensor v16 = refinedV.CastTo(slotType);
                refinedV.Dispose();
                refinedV = v16;
            }
            cache.Set(layer, frame, refinedK, refinedV);
            refinedK = refinedV = null;
        }
        finally
        {
            k32?.Dispose();
            v32?.Dispose();
            refinedK?.Dispose();
            refinedV?.Dispose();
        }
    }

    private Tensor FrameTensor(float[] values)
    {
        if (values.Length != _config.FrameValues)
            throw new ArgumentException($"A latent frame has {_config.FrameValues} values; the noise gave {values.Length}.", nameof(values));
        Tensor t = new(new TensorShape([1L, _config.LatentChannels, 1, _config.LatentHeight, _config.LatentWidth]), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    /// <summary>One decoded frame <c>[3, H, W]</c> as 8-bit RGB: <c>round((clamp(x, -1, 1) + 1) · 127.5)</c>, channels last. A non-finite
    /// pixel means the rollout diverged, so it is refused rather than rendered as black.</summary>
    private static byte[] ToRgb8(FastFramesFrame frame, int pixelFrame)
    {
        int h = frame.PixelHeight, w = frame.PixelWidth;
        byte[] rgb = new byte[h * w * 3];
        ReadOnlySpan<float> source = frame.Pixels.AsSpan((pixelFrame * 3) * h * w, 3 * h * w);
        for (int c = 0; c < 3; c++)
        {
            for (int i = 0; i < h * w; i++)
            {
                float raw = source[c * h * w + i];
                if (!float.IsFinite(raw))
                    throw new InvalidOperationException($"Latent frame {frame.Index} decoded a non-finite pixel; the rollout diverged.");
                float v = Math.Clamp(raw, -1f, 1f);
                rgb[i * 3 + c] = (byte)MathF.Round((v + 1f) * 127.5f);
            }
        }
        return rgb;
    }
}
