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
    int LatentChannels = 16, int LatentHeight = 60, int LatentWidth = 104, int LatentFrames = 21, float Shift = 5f)
{
    /// <summary>FastFrames-S: the Causal Forcing++ two-step frame-wise generator (shifted timesteps of nominal 1000, 750, 500, 250).</summary>
    public static FastFramesConfig S { get; } = new("FF-S", [1000f, 937.5f, 833.3333129882812f, 625f], [1000f, 833.3333129882812f], 833.3333129882812f);

    /// <summary>FastFrames-L: the two-step model distilled from Wan2.1-14B.</summary>
    public static FastFramesConfig L { get; } = new("FF-L", [1000f, 937f, 833f, 625f], [1000f, 833f], 833f);

    /// <summary>Latent values in one frame (<c>channels · height · width</c>).</summary>
    public int FrameValues => LatentChannels * LatentHeight * LatentWidth;

    /// <summary>The generator's token grid height (latent height over the 2×2 patch).</summary>
    public int GridHeight => LatentHeight / 2;

    /// <summary>The generator's token grid width.</summary>
    public int GridWidth => LatentWidth / 2;
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
/// to refine from). A frame is yielded as soon as it is decoded, so a consumer can stream.
/// <para>The pipeline does not own its components. Numerics are the engine's (F32 residual stream, the Wan block's branch dtype);
/// they differ from upstream's BF16 run by BF16 rounding, which is why parity is judged against an FP32 run of the upstream code.</para></summary>
public sealed class FastFramesPipeline(IBackend backend, WanVideoTransformer transformer, FastFramesKvRefiner refiner,
    FastFramesPixelDecoder decoder, FastFramesConfig config)
{
    private readonly FastFramesConfig _config = config;
    private readonly FastFramesFlowSchedule _schedule = new(config.Shift);

    /// <summary>The first <c>patchT − 1</c> pixel frames of the clip are dropped (upstream's <c>align_frames</c>: 84 decoded frames to 81).</summary>
    public int LeadingPixelFramesDropped => decoder.Config.PatchT - 1;

    /// <summary>Pixel frames a clip of <paramref name="latentFrames"/> latent frames has after the leading ones are dropped.</summary>
    public int PixelFrameCount(int latentFrames) => latentFrames * decoder.Config.PatchT - LeadingPixelFramesDropped;

    /// <summary>Runs the rollout and yields each latent frame once it is decoded. <paramref name="promptEmbeds"/> is the umT5
    /// output zero-padded to the generator's text length (<c>[512, 4096]</c> for Wan), as <c>WanVideoTransformer</c> takes it.</summary>
    public IEnumerable<FastFramesFrame> Rollout(Tensor promptEmbeds, IFastFramesNoise noise, int latentFrames, Action<int, int>? progress = null,
        CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(promptEmbeds);
        ArgumentNullException.ThrowIfNull(noise);
        ArgumentOutOfRangeException.ThrowIfLessThan(latentFrames, 1);
        int layers = transformer.BlockCount;
        using FrameKvCache cache = new(backend, layers, latentFrames);
        using FastFramesPixelDecoder.State decoderState = decoder.CreateState();
        using FastFramesKvRefiner.Conditioning conditioning = refiner.Prepare(backend, _config.RefineTimestep);
        int gridH = _config.GridHeight, gridW = _config.GridWidth;

        for (int frame = 0; frame < latentFrames; frame++)
        {
            cancel.ThrowIfCancellationRequested();
            float[] latent = Denoise(promptEmbeds, noise, frame, cache, out Tensor hidden, out Tensor temb);
            Tensor pixels;
            using (hidden)
            using (temb)
                pixels = decoder.DecodeFrame(backend, hidden, temb, gridH, gridW, decoderState);
            if (frame == 0)
                CleanPass(promptEmbeds, latent, frame, cache);
            else
                Refine(latent, frame, cache, conditioning);
            using (pixels)
            {
                yield return new FastFramesFrame(frame, latent, pixels.AsReadOnlySpan<float>().ToArray(), (int)pixels.Shape[0],
                    (int)pixels.Shape[2], (int)pixels.Shape[3]);
            }
            progress?.Invoke(frame + 1, latentFrames);
        }
    }

    /// <summary>Generates a clip and yields its frames as 8-bit RGB (<c>[H, W, 3]</c>, top row first), the leading frames dropped, as
    /// the frames decode.</summary>
    public IEnumerable<VideoFrame> GenerateFrames(Tensor promptEmbeds, IFastFramesNoise noise, int latentFrames, Action<int, int>? progress = null,
        CancellationToken cancel = default)
    {
        int outIndex = 0;
        int skip = LeadingPixelFramesDropped;
        foreach (FastFramesFrame frame in Rollout(promptEmbeds, noise, latentFrames, progress, cancel))
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

    /// <summary>One latent frame's denoising steps. Returns the final clean-latent estimate; <paramref name="hidden"/> and
    /// <paramref name="temb"/> are the last step's pre-head hidden states and head time embedding, which the decoder reads.</summary>
    private float[] Denoise(Tensor promptEmbeds, IFastFramesNoise noise, int frame, FrameKvCache cache, out Tensor hidden, out Tensor temb)
    {
        float[] schedule = frame == 0 ? _config.FirstFrameTimesteps : _config.Timesteps;
        float[] x = (float[])noise.Initial(frame).Clone();
        float[] x0 = new float[x.Length];
        hidden = null!;
        temb = null!;
        for (int step = 0; step < schedule.Length; step++)
        {
            using Tensor input = FrameTensor(x);
            WanVideoTransformer.FrameForward forward = transformer.ForwardFrame(backend, input, promptEmbeds, schedule[step], frame, cache);
            double sigma = _schedule.Sigma(schedule[step]);
            using (forward.Velocity)
            {
                ReadOnlySpan<float> velocity = forward.Velocity.AsReadOnlySpan<float>();
                for (int i = 0; i < x.Length; i++)
                    x0[i] = (float)(x[i] - sigma * velocity[i]);   // x0 = x_t - sigma · flow, in double
            }
            if (step + 1 < schedule.Length)
            {
                forward.Hidden.Dispose();
                forward.Temb.Dispose();
                double next = _schedule.Sigma(schedule[step + 1]);
                float[] eps = noise.Transition(step, frame);
                for (int i = 0; i < x.Length; i++)
                    x[i] = (float)((1 - next) * x0[i] + next * eps[i]);
            }
            else
            {
                hidden = forward.Hidden;
                temb = forward.Temb;
            }
        }
        return x0;
    }

    /// <summary>One generator pass over the clean frame at timestep 0, which leaves the frame's clean keys and values in its slot.</summary>
    private void CleanPass(Tensor promptEmbeds, float[] latent, int frame, FrameKvCache cache)
    {
        using Tensor input = FrameTensor(latent);
        WanVideoTransformer.FrameForward forward = transformer.ForwardFrame(backend, input, promptEmbeds, 0f, frame, cache);
        forward.Velocity.Dispose();
        forward.Hidden.Dispose();
        forward.Temb.Dispose();
    }

    /// <summary>Replaces the frame's noisy cache entries in every layer with the refinement network's clean-context ones.</summary>
    private void Refine(float[] latent, int frame, FrameKvCache cache, FastFramesKvRefiner.Conditioning conditioning)
    {
        using Tensor input = FrameTensor(latent);
        using Tensor stem = transformer.PatchEmbedFrame(backend, input);
        using Tensor phi = refiner.ComputePhi(backend, stem, conditioning);
        using Tensor phiV = refiner.ProjectPhi(backend, phi);
        for (int layer = 0; layer < cache.Layers; layer++)
        {
            Tensor noisyK = cache.Keys(layer, frame), noisyV = cache.Values(layer, frame);
            DType slotType = noisyK.DType;
            Tensor k32 = slotType == DType.F32 ? noisyK : noisyK.CastTo(DType.F32);
            Tensor v32 = slotType == DType.F32 ? noisyV : noisyV.CastTo(DType.F32);
            (Tensor refinedK, Tensor refinedV) = refiner.RepairLayer(backend, layer, k32, v32, phiV, conditioning);
            if (slotType != DType.F32)
            {
                k32.Dispose();
                v32.Dispose();
                Tensor k16 = refinedK.CastTo(slotType), v16 = refinedV.CastTo(slotType);
                refinedK.Dispose();
                refinedV.Dispose();
                (refinedK, refinedV) = (k16, v16);
            }
            cache.Set(layer, frame, refinedK, refinedV);
        }
    }

    private Tensor FrameTensor(float[] values)
    {
        Tensor t = new(new TensorShape([1L, _config.LatentChannels, 1, _config.LatentHeight, _config.LatentWidth]), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    /// <summary>One decoded frame <c>[3, H, W]</c> as 8-bit RGB: <c>round((clamp(x, -1, 1) + 1) · 127.5)</c>, channels last.</summary>
    private static byte[] ToRgb8(FastFramesFrame frame, int pixelFrame)
    {
        int h = frame.PixelHeight, w = frame.PixelWidth;
        byte[] rgb = new byte[h * w * 3];
        ReadOnlySpan<float> source = frame.Pixels.AsSpan((pixelFrame * 3) * h * w, 3 * h * w);
        for (int c = 0; c < 3; c++)
        {
            for (int i = 0; i < h * w; i++)
            {
                float v = Math.Clamp(source[c * h * w + i], -1f, 1f);
                rgb[i * 3 + c] = (byte)MathF.Round((v + 1f) * 127.5f);
            }
        }
        return rgb;
    }
}
