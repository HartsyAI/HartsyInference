using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.FastFrames;
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
public sealed class FastFramesPipeline
{
    private readonly IBackend _backend;
    private readonly WanVideoTransformer _transformer;
    private readonly FastFramesKvRefiner _refiner;
    private readonly FastFramesPixelDecoder _decoder;
    private readonly FastFramesConfig _config;
    private readonly FastFramesFlowSchedule _schedule;
    private readonly int _gridHeight, _gridWidth;

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

    private static void ValidateRequest(Tensor promptEmbeds, IFastFramesNoise noise, int latentFrames)
    {
        ArgumentNullException.ThrowIfNull(promptEmbeds);
        ArgumentNullException.ThrowIfNull(noise);
        ArgumentOutOfRangeException.ThrowIfLessThan(latentFrames, 1);
    }

    private IEnumerable<FastFramesFrame> RolloutCore(Tensor promptEmbeds, IFastFramesNoise noise, int latentFrames,
        Action<int, int>? progress, CancellationToken cancel)
    {
        using FrameKvCache cache = new(_backend, _transformer.BlockCount, latentFrames);
        using FastFramesPixelDecoder.State decoderState = _decoder.CreateState();
        using FastFramesKvRefiner.Conditioning conditioning = _refiner.Prepare(_backend, _config.RefineTimestep);
        for (int frame = 0; frame < latentFrames; frame++)
        {
            cancel.ThrowIfCancellationRequested();
            (float[] latent, Tensor pixels) = DenoiseAndDecode(promptEmbeds, noise, frame, cache, decoderState);
            FastFramesFrame output;
            using (pixels)
                output = new FastFramesFrame(frame, latent, pixels.AsReadOnlySpan<float>().ToArray(), _decoder.Config.PatchT,
                    (int)pixels.Shape[2], (int)pixels.Shape[3]);
            yield return output;

            // The next frame reads this frame's cache slot, so the clean pass or the refinement runs before it. It runs after the
            // consumer has the frame, so a frame is not held back by the cache work of the one after it.
            if (frame == 0)
                CleanPass(promptEmbeds, latent, frame, cache);
            else
                Refine(latent, frame, cache, conditioning);
            progress?.Invoke(frame + 1, latentFrames);
        }
    }

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
