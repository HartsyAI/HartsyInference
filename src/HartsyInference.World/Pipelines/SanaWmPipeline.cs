using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.World.Camera;
using HartsyInference.World.Pipelines.SanaWm;

namespace HartsyInference.World.Pipelines;

/// <summary>SANA-WM stage-1 clip generation (<c>flow_euler_ltx</c>, upstream <c>_sample_stage1</c> and
/// <c>_decode_with_sana_vae</c>): the seed image is cover-fit and center-cropped to 704×1280, its normalized VAE latent
/// becomes latent frame 0 of seeded noise, the DiT denoises every other frame under classifier-free guidance while frame 0
/// stays hard-conditioned, and the LTX-2 VAE decodes the result in temporal windows. Components are resident one stage at a
/// time (text encoder, VAE encoder, DiT, VAE decoder) through <see cref="ISanaWmComponents"/>. Values cross every stage
/// boundary rounded to BF16, and the guidance combine and latent update follow upstream's BF16 arithmetic, so a run fed the
/// same component outputs reproduces upstream's sampler bit for bit. Returns frames; writing a container is the caller's.</summary>
public sealed class SanaWmPipeline
{
    /// <summary>Output width: SANA-WM is trained at this single resolution.</summary>
    public const int Width = 1280;

    /// <summary>Output height.</summary>
    public const int Height = 704;

    /// <summary>Upstream's default step count.</summary>
    public const int DefaultSteps = 60;

    /// <summary>Upstream's default guidance scale.</summary>
    public const double DefaultGuidance = 5.0;

    /// <summary>Upstream's default schedule shift (<c>inference_flow_shift</c>).</summary>
    public const double DefaultFlowShift = 9.8;

    /// <summary>Latent channels of the LTX-2 VAE.</summary>
    public const int LatentChannels = 128;

    /// <summary>Upstream's temporal decode window (<c>tile_sample_min_num_frames</c>), in decoded frames.</summary>
    public const int DecodeWindowFrames = 96;

    /// <summary>Upstream's temporal decode stride (<c>tile_sample_stride_num_frames</c>), in decoded frames.</summary>
    public const int DecodeStrideFrames = 64;

    private const float TrainTimesteps = 1000f;
    private const float SigmaEpsilon = 1e-6f;
    internal const int LatentHeight = Height / SanaWmCameraConditioning.SpatialStride;
    internal const int LatentWidth = Width / SanaWmCameraConditioning.SpatialStride;
    private const int LatentPlane = LatentHeight * LatentWidth;

    private readonly ISanaWmComponents _components;

    /// <summary>Creates the pipeline over <paramref name="components"/>, which it opens one stage at a time per clip.</summary>
    public SanaWmPipeline(ISanaWmComponents components)
    {
        ArgumentNullException.ThrowIfNull(components);
        _components = components;
    }

    /// <summary>Renders one clip. Checks <paramref name="cancel"/> before each stage and each denoising step; a stage that is
    /// open when the run fails or is cancelled is released before the exception leaves.</summary>
    public SanaWmClip Generate(SanaWmClipRequest request, CancellationToken cancel, Action<GenerationProgress>? progress = null)
    {
        Validate(request);
        int latentFrames = (request.Frames - 1) / SanaWmCameraConditioning.TemporalStride + 1;
        CoverCropMapping crop = CoverCropMapping.Create(request.ImageWidth, request.ImageHeight, Width, Height);
        using SanaWmCameraConditioning camera = SanaWmCameraConditioning.Prepare(request.CameraToWorld, request.Intrinsics, crop,
            Width, Height);
        RoundToBf16(camera.Conditions);
        RoundToBf16(camera.ChunkPlucker.AsSpan<float>());
        cancel.ThrowIfCancellationRequested();

        using SanaWmTextConditioning text = EncodeText(request);
        cancel.ThrowIfCancellationRequested();
        using Tensor latents = InitialLatents(request, latentFrames);
        cancel.ThrowIfCancellationRequested();
        Sample(request, text, camera, latents, cancel, progress);
        request.Tap?.Invoke("final_latent", latents);
        cancel.ThrowIfCancellationRequested();
        IReadOnlyList<byte[]> frames = Decode(latents, request.Frames);
        return new SanaWmClip
        {
            Frames = frames,
            Width = Width,
            Height = Height,
            Fps = SanaWmAction.Fps,
            LatentFrames = latentFrames,
            Steps = request.Steps,
            Guidance = request.Guidance,
            FlowShift = request.FlowShift,
            Seed = request.Seed,
            InitialLatentInjected = request.InitialLatent is not null,
        };
    }

    /// <summary>Upstream <c>resize_and_center_crop</c> then <c>ToTensor() * 2 − 1</c>: an aspect-preserving PIL Lanczos resize
    /// that covers 704×1280, a center crop, and 8-bit pixels mapped to [-1, 1]. Returns a host F32 <c>[1, 3, 1, 704, 1280]</c>
    /// tensor owned by the caller.</summary>
    public static Tensor PrepareImage(ReadOnlySpan<byte> rgb, int width, int height)
    {
        CoverCropMapping crop = CoverCropMapping.Create(width, height, Width, Height);
        if (rgb.Length != width * height * 3)
            throw new ArgumentException($"The seed image holds {rgb.Length} bytes; {width}x{height} RGB24 needs {width * height * 3}.", nameof(rgb));

        int resizedWidth = crop.ResizedWidth;
        byte[] resized = new byte[resizedWidth * crop.ResizedHeight * 3];
        Resample.LanczosPil8(rgb, width, height, 3, resized, resizedWidth, crop.ResizedHeight);

        Tensor video = new(new TensorShape([1, 3, 1, Height, Width]), DType.F32, DeviceKind.Cpu);
        Span<float> destination = video.AsSpan<float>();
        const int Plane = Width * Height;
        for (int y = 0; y < Height; y++)
        {
            int sourceRow = (y + crop.Top) * resizedWidth;
            for (int x = 0; x < Width; x++)
            {
                int source = (sourceRow + x + crop.Left) * 3;
                for (int c = 0; c < 3; c++)
                    destination[c * Plane + y * Width + x] = resized[source + c] / 255f * 2f - 1f;
            }
        }
        return video;
    }

    /// <summary>Upstream <c>clamp(127.5·x + 127.5, 0, 255).to(uint8)</c> on a BF16 decode: each product and sum rounded to BF16,
    /// then truncated. Converts frame <paramref name="frame"/> of RGB <c>[1, 3, F, H, W]</c> into an RGB24 buffer.</summary>
    public static byte[] ToRgb24(ReadOnlySpan<float> rgb, int frames, int frame, int height, int width)
    {
        int plane = height * width;
        byte[] pixels = new byte[plane * 3];
        for (int c = 0; c < 3; c++)
        {
            ReadOnlySpan<float> channel = rgb.Slice((c * frames + frame) * plane, plane);
            for (int p = 0; p < plane; p++)
            {
                float scaled = Bf16(Bf16(127.5f * channel[p]) + 127.5f);
                pixels[p * 3 + c] = float.IsNaN(scaled) ? (byte)0 : (byte)MathF.Min(MathF.Max(scaled, 0f), 255f);
            }
        }
        return pixels;
    }

    /// <summary>One <c>flow_euler_ltx</c> update in place, as diffusers' per-token <c>FlowMatchEulerDiscreteScheduler.step</c>
    /// computes it: each latent frame moves by <c>(σ − σ_next)·(−v)</c>, where σ is its timestep over 1000 and σ_next the
    /// largest schedule sigma below it, then frame 0 is restored from <paramref name="conditionFrame"/> and the result is
    /// rounded to BF16.</summary>
    /// <param name="latents">Host F32 <c>[1, C, T, H, W]</c>.</param>
    /// <param name="velocity">The guided velocity, shaped like <paramref name="latents"/>.</param>
    /// <param name="timestep">This step's timestep (σ·1000); frame 0 always runs at 0.</param>
    /// <param name="sigmas">The schedule, terminal 0 included.</param>
    /// <param name="conditionFrame">Latent frame 0 as conditioned, <c>C·H·W</c> floats.</param>
    public static void EulerStep(Span<float> latents, ReadOnlySpan<float> velocity, float timestep, ReadOnlySpan<float> sigmas,
        ReadOnlySpan<float> conditionFrame)
    {
        int framePlane = conditionFrame.Length / LatentChannels;
        int frames = latents.Length / (LatentChannels * framePlane);
        if (velocity.Length != latents.Length || framePlane * LatentChannels != conditionFrame.Length || frames < 1)
            throw new ArgumentException("Latents, velocity and the condition frame disagree in shape.");
        float sigma = timestep / TrainTimesteps;
        float dt = sigma - NextSigma(sigma, sigmas);
        for (int c = 0; c < LatentChannels; c++)
        {
            int channel = c * frames * framePlane;
            conditionFrame.Slice(c * framePlane, framePlane).CopyTo(latents.Slice(channel, framePlane));
            for (int i = channel + framePlane; i < channel + frames * framePlane; i++)
                latents[i] = Bf16(latents[i] + dt * -velocity[i]);
        }
    }

    /// <summary>Upstream's guidance combine in BF16: <c>u + s·(c − u)</c> with each operation rounded, after both branch outputs are
    /// rounded to BF16 as the BF16 DiT emits them. Writes into <paramref name="unconditional"/>.</summary>
    public static void CombineGuidance(Span<float> unconditional, ReadOnlySpan<float> conditional, double scale)
    {
        if (unconditional.Length != conditional.Length)
            throw new ArgumentException("Guidance branches differ in length.", nameof(conditional));
        float s = (float)scale;
        for (int i = 0; i < unconditional.Length; i++)
        {
            float u = Bf16(unconditional[i]);
            float difference = Bf16(Bf16(conditional[i]) - u);
            unconditional[i] = Bf16(u + Bf16(s * difference));
        }
    }

    /// <summary>Rounds every value to the nearest BF16, ties to even.</summary>
    public static void RoundToBf16(Span<float> values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = Bf16(values[i]);
    }

    private SanaWmTextConditioning EncodeText(SanaWmClipRequest request) =>
        SanaWmEncoding.EncodeText(_components.OpenTextEncoder, request.Prompt, request.NegativePrompt);

    private Tensor InitialLatents(SanaWmClipRequest request, int latentFrames)
    {
        TensorShape shape = new([1, LatentChannels, latentFrames, LatentHeight, LatentWidth]);
        Tensor latents = new(shape, DType.F32, DeviceKind.Cpu);
        bool built = false;
        try
        {
            Span<float> values = latents.AsSpan<float>();
            if (request.InitialLatent is Tensor injected)
            {
                if (injected.DType != DType.F32 || injected.Shape != shape)
                    throw new ArgumentException($"The injected initial latent must be F32 {shape}; got {injected.DType} {injected.Shape}.",
                        nameof(request));
                injected.AsReadOnlySpan<float>().CopyTo(values);
                RoundToBf16(values);
            }
            else
            {
                using Tensor first = EncodeFirstFrame(request);
                using Tensor noise = SeedGenerator.CreateNoise(shape, unchecked((int)request.Seed));
                noise.AsReadOnlySpan<float>().CopyTo(values);
                RoundToBf16(values);
                ReadOnlySpan<float> firstValues = first.AsReadOnlySpan<float>();
                for (int c = 0; c < LatentChannels; c++)
                    firstValues.Slice(c * LatentPlane, LatentPlane).CopyTo(values.Slice(c * latentFrames * LatentPlane, LatentPlane));
            }
            request.Tap?.Invoke("z0", latents);
            built = true;
            return latents;
        }
        finally
        {
            if (!built)
                latents.Dispose();
        }
    }

    private Tensor EncodeFirstFrame(SanaWmClipRequest request) =>
        SanaWmEncoding.EncodeFirstFrame(_components.OpenImageEncoder, request.ImageRgb, request.ImageWidth, request.ImageHeight,
            request.Tap);

    private void Sample(SanaWmClipRequest request, SanaWmTextConditioning text, SanaWmCameraConditioning camera, Tensor latents,
        CancellationToken cancel, Action<GenerationProgress>? progress)
    {
        (float[] sigmas, float[] timesteps) = SanaWmSchedule.Build(request.Steps, request.FlowShift);
        int latentFrames = (int)latents.Shape[2];
        float[] conditionFrame = new float[LatentChannels * LatentPlane];
        ReadOnlySpan<float> initial = latents.AsReadOnlySpan<float>();
        for (int c = 0; c < LatentChannels; c++)
            initial.Slice(c * latentFrames * LatentPlane, LatentPlane).CopyTo(conditionFrame.AsSpan(c * LatentPlane));

        bool guided = request.Guidance > 1.0;
        using SanaWmStage<ISanaWmDenoiser> stage = _components.OpenDenoiser(text, camera);
        ISanaWmDenoiser denoiser = stage.Component;
        if (guided && request.SequentialCfg && !denoiser.SupportsSingleBranch)
            throw new NotSupportedException("The SANA-WM denoiser cannot run one guidance branch per forward; sequential CFG is unavailable.");

        float[] frameTimesteps = new float[latentFrames];
        Stopwatch clock = Stopwatch.StartNew();
        for (int step = 0; step < request.Steps; step++)
        {
            cancel.ThrowIfCancellationRequested();
            double started = clock.Elapsed.TotalMilliseconds;
            // Frame 0 is hard-conditioned: min(t, (1 − mask)·1000) gives it timestep 0.
            Array.Fill(frameTimesteps, timesteps[step]);
            frameTimesteps[0] = 0f;
            using Tensor velocity = Velocity(denoiser, latents, frameTimesteps, guided, request.SequentialCfg, request.Guidance);
            EulerStep(latents.AsSpan<float>(), velocity.AsReadOnlySpan<float>(), timesteps[step], sigmas, conditionFrame);
            progress?.Invoke(new GenerationProgress(step, request.Steps, clock.Elapsed.TotalMilliseconds - started));
        }
    }

    private static Tensor Velocity(ISanaWmDenoiser denoiser, Tensor latents, float[] frameTimesteps, bool guided, bool sequential,
        double guidance)
    {
        if (!guided)
        {
            Tensor conditional = Predict(denoiser, latents, frameTimesteps, [SanaWmGuidanceBranch.Conditional], latents.Shape);
            RoundToBf16(conditional.AsSpan<float>());
            return conditional;
        }
        if (sequential)
        {
            Tensor unconditional = Predict(denoiser, latents, frameTimesteps, [SanaWmGuidanceBranch.Unconditional], latents.Shape);
            try
            {
                using Tensor conditional = Predict(denoiser, latents, frameTimesteps, [SanaWmGuidanceBranch.Conditional], latents.Shape);
                CombineGuidance(unconditional.AsSpan<float>(), conditional.AsReadOnlySpan<float>(), guidance);
                return unconditional;
            }
            catch
            {
                unconditional.Dispose();
                throw;
            }
        }

        long count = latents.Shape.ElementCount;
        TensorShape batchShape = new([2, latents.Shape[1], latents.Shape[2], latents.Shape[3], latents.Shape[4]]);
        using Tensor batch = new(batchShape, DType.F32, DeviceKind.Cpu);
        Span<float> batchValues = batch.AsSpan<float>();
        latents.AsReadOnlySpan<float>().CopyTo(batchValues);
        latents.AsReadOnlySpan<float>().CopyTo(batchValues[(int)count..]);
        float[] batchTimesteps = new float[frameTimesteps.Length * 2];
        frameTimesteps.CopyTo(batchTimesteps, 0);
        frameTimesteps.CopyTo(batchTimesteps, frameTimesteps.Length);
        using Tensor both = Predict(denoiser, batch, batchTimesteps,
            [SanaWmGuidanceBranch.Unconditional, SanaWmGuidanceBranch.Conditional], batchShape);
        Tensor combined = new(latents.Shape, DType.F32, DeviceKind.Cpu);
        ReadOnlySpan<float> bothValues = both.AsReadOnlySpan<float>();
        bothValues[..(int)count].CopyTo(combined.AsSpan<float>());
        CombineGuidance(combined.AsSpan<float>(), bothValues[(int)count..], guidance);
        return combined;
    }

    private static Tensor Predict(ISanaWmDenoiser denoiser, Tensor latents, ReadOnlySpan<float> timesteps,
        ReadOnlySpan<SanaWmGuidanceBranch> branches, TensorShape expected)
    {
        Tensor velocity = denoiser.Predict(latents, timesteps, branches);
        if (velocity.DType != DType.F32 || velocity.Shape != expected)
        {
            velocity.Dispose();
            throw new InvalidOperationException($"The denoiser returned {velocity.DType} {velocity.Shape}; expected F32 {expected}.");
        }
        return velocity;
    }

    private IReadOnlyList<byte[]> Decode(Tensor latents, int frames)
    {
        Tensor rgb;
        using (SanaWmStage<ISanaWmVideoDecoder> stage = _components.OpenVideoDecoder())
            rgb = stage.Component.Decode(latents);
        using (rgb)
        {
            TensorShape expected = new([1, 3, frames, Height, Width]);
            if (rgb.DType != DType.F32 || rgb.Shape != expected)
                throw new InvalidOperationException($"The video decoder returned {rgb.DType} {rgb.Shape}; expected F32 {expected}.");
            ReadOnlySpan<float> values = rgb.AsReadOnlySpan<float>();
            byte[][] result = new byte[frames][];
            for (int f = 0; f < frames; f++)
                result[f] = ToRgb24(values, frames, f, Height, Width);
            return result;
        }
    }

    // The largest schedule sigma strictly below sigma − 1e-6, or 0 when there is none (diffusers' per-token lower_sigmas).
    private static float NextSigma(float sigma, ReadOnlySpan<float> sigmas)
    {
        float threshold = sigma - SigmaEpsilon;
        float next = 0f;
        foreach (float candidate in sigmas)
        {
            if (candidate < threshold && candidate > next)
                next = candidate;
        }
        return next;
    }

    private static void Validate(SanaWmClipRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.ImageRgb);
        ArgumentNullException.ThrowIfNull(request.Prompt);
        ArgumentNullException.ThrowIfNull(request.NegativePrompt);
        ArgumentNullException.ThrowIfNull(request.CameraToWorld);
        ArgumentNullException.ThrowIfNull(request.Intrinsics);
        int stride = SanaWmCameraConditioning.TemporalStride;
        if (request.Frames < 1 || (request.Frames - 1) % stride != 0)
            throw new ArgumentException($"SANA-WM renders {stride}n+1 frames; {request.Frames} requested.", nameof(request));
        if (request.CameraToWorld.Count != request.Frames)
            throw new ArgumentException($"The camera path has {request.CameraToWorld.Count} poses for {request.Frames} frames.", nameof(request));
        if (request.Intrinsics.Count == 0)
            throw new ArgumentException("SANA-WM needs intrinsics for its camera path.", nameof(request));
        if (request.Steps < 1)
            throw new ArgumentException($"Steps must be positive; {request.Steps} requested.", nameof(request));
        if (!double.IsFinite(request.Guidance) || request.Guidance < 0)
            throw new ArgumentException($"Guidance must be a finite non-negative scale; {request.Guidance} requested.", nameof(request));
        if (!double.IsFinite(request.FlowShift) || request.FlowShift <= 0)
            throw new ArgumentException($"Flow shift must be positive; {request.FlowShift} requested.", nameof(request));
        if (request.ImageWidth < 1 || request.ImageHeight < 1 || request.ImageRgb.Length != request.ImageWidth * request.ImageHeight * 3)
            throw new ArgumentException($"The seed image is not {request.ImageWidth}x{request.ImageHeight} RGB24.", nameof(request));
    }

    private static float Bf16(float value) => BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(value));
}
