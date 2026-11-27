using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.World.Camera;

/// <summary>The SANA-WM DiT camera inputs for one clip, built as upstream <c>prepare_camera</c> and
/// <c>_pack_camera_conditions</c> build them: poses relativized to frame 0, intrinsics fitted over time, mapped into the
/// cover-cropped image and scaled to latent pixels, then <see cref="Conditions"/> sampled every
/// <see cref="TemporalStride"/> frames and <see cref="ChunkPlucker"/> holding the Plücker rays of each latent frame's
/// raw frames. Owns <see cref="ChunkPlucker"/>.</summary>
public sealed class SanaWmCameraConditioning : IDisposable
{
    /// <summary>Raw frames per latent frame (the VAE's temporal stride).</summary>
    public const int TemporalStride = 8;

    /// <summary>Pixels per latent pixel (the VAE's spatial stride).</summary>
    public const int SpatialStride = 32;

    /// <summary>Floats per latent frame in <see cref="Conditions"/>: 16 camera-to-world values, then fx, fy, cx, cy.</summary>
    public const int ConditionWidth = 20;

    /// <summary>Channels of <see cref="ChunkPlucker"/>: six Plücker channels for each raw frame of a latent frame.</summary>
    public const int PluckerChannels = TemporalStride * PluckerEmbedding.Channels;

    private Tensor? _chunkPlucker;

    private SanaWmCameraConditioning(float[][] relativePoses, PinholeIntrinsics[] cropIntrinsics,
        PinholeIntrinsics[] latentIntrinsics, float[] conditions, Tensor chunkPlucker, int latentFrames)
    {
        RelativePoses = relativePoses;
        CropIntrinsics = cropIntrinsics;
        LatentIntrinsics = latentIntrinsics;
        Conditions = conditions;
        _chunkPlucker = chunkPlucker;
        LatentFrames = latentFrames;
    }

    /// <summary>Camera-to-world poses relative to frame 0 (identity first), one per raw frame, 16 row-major floats each.</summary>
    public IReadOnlyList<float[]> RelativePoses { get; }

    /// <summary>Intrinsics per raw frame in pixels of the cropped image.</summary>
    public IReadOnlyList<PinholeIntrinsics> CropIntrinsics { get; }

    /// <summary>Intrinsics per raw frame in latent pixels.</summary>
    public IReadOnlyList<PinholeIntrinsics> LatentIntrinsics { get; }

    /// <summary>Per latent frame, the relative pose (16 floats) then the latent intrinsics: <c>LatentFrames · 20</c> floats.</summary>
    public float[] Conditions { get; }

    /// <summary>Host F32 <c>[1, 48, LatentFrames, H, W]</c>: channel <c>6·k + c</c> is Plücker channel <c>c</c>
    /// (direction then moment) of the latent frame's <c>k</c>-th raw frame.</summary>
    public Tensor ChunkPlucker => _chunkPlucker ?? throw new ObjectDisposedException(nameof(SanaWmCameraConditioning));

    /// <summary>Latent frames: <c>(frames − 1) / 8 + 1</c>.</summary>
    public int LatentFrames { get; }

    /// <summary>Builds the camera inputs for a clip of <c>cameraToWorld.Count</c> frames.</summary>
    /// <param name="cameraToWorld">One camera-to-world pose per raw frame (16 row-major floats), frame 0 first.</param>
    /// <param name="sourceIntrinsics">Intrinsics in source-image pixels: one entry for every frame, or a per-frame
    /// sequence fitted to the frame count by <see cref="FitOverTime"/>.</param>
    /// <param name="crop">The cover-fit and center-crop that took the source image to the model's input size.</param>
    /// <param name="width">Width of the cropped image, a multiple of <see cref="SpatialStride"/>.</param>
    /// <param name="height">Height of the cropped image, a multiple of <see cref="SpatialStride"/>.</param>
    public static SanaWmCameraConditioning Prepare(IReadOnlyList<float[]> cameraToWorld, IReadOnlyList<PinholeIntrinsics> sourceIntrinsics,
        CoverCropMapping crop, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(cameraToWorld);
        ArgumentNullException.ThrowIfNull(sourceIntrinsics);
        int frames = cameraToWorld.Count;
        if (width < SpatialStride || height < SpatialStride || width % SpatialStride != 0 || height % SpatialStride != 0)
            throw new ArgumentException($"The cropped size {width}x{height} is not a positive multiple of {SpatialStride}.", nameof(width));

        PinholeIntrinsics[] fitted = FitOverTime(sourceIntrinsics, frames);
        PinholeIntrinsics[] cropped = new PinholeIntrinsics[frames];
        PinholeIntrinsics[] latent = new PinholeIntrinsics[frames];
        int latentWidth = width / SpatialStride, latentHeight = height / SpatialStride;
        for (int i = 0; i < frames; i++)
        {
            (float fx, float fy, float cx, float cy) = crop.MapIntrinsics(fitted[i].Fx, fitted[i].Fy, fitted[i].Cx, fitted[i].Cy);
            cropped[i] = new PinholeIntrinsics(fx, fy, cx, cy);
            latent[i] = ToLatent(cropped[i], width, height, latentWidth, latentHeight);
        }
        float[][] relative = Relativize(cameraToWorld);
        (float[] conditions, Tensor plucker) = Pack(relative, latent, latentHeight, latentWidth);
        return new SanaWmCameraConditioning(relative, cropped, latent, conditions, plucker, conditions.Length / ConditionWidth);
    }

    /// <summary>Poses relative to the first: the identity, then <c>P₀⁻¹·Pᵢ</c> for every later pose. Computed in double
    /// precision and rounded to single.</summary>
    public static float[][] Relativize(IReadOnlyList<float[]> cameraToWorld)
    {
        ArgumentNullException.ThrowIfNull(cameraToWorld);
        if (cameraToWorld.Count == 0)
            throw new ArgumentException("The camera path has no poses.", nameof(cameraToWorld));
        foreach (float[] pose in cameraToWorld)
        {
            if (pose is null || pose.Length != 16)
                throw new ArgumentException("Each camera pose is 16 row-major floats.", nameof(cameraToWorld));
        }

        float[] first = cameraToWorld[0];
        Span<double> inverse = stackalloc double[16];
        // Rigid inverse [Rᵀ | −Rᵀt], as upstream get_pose_inverse builds it.
        for (int r = 0; r < 3; r++)
        {
            double t = 0.0;
            for (int c = 0; c < 3; c++)
            {
                inverse[r * 4 + c] = first[c * 4 + r];
                t += (double)first[c * 4 + r] * first[c * 4 + 3];
            }
            inverse[r * 4 + 3] = -t;
        }
        inverse[12] = 0; inverse[13] = 0; inverse[14] = 0; inverse[15] = 1;

        float[][] relative = new float[cameraToWorld.Count][];
        relative[0] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        for (int i = 1; i < cameraToWorld.Count; i++)
        {
            float[] pose = cameraToWorld[i];
            float[] result = new float[16];
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < 4; k++)
                        sum += inverse[r * 4 + k] * pose[k * 4 + c];
                    result[r * 4 + c] = (float)sum;
                }
            }
            relative[i] = result;
        }
        return relative;
    }

    /// <summary>Fits a per-frame intrinsics sequence to <paramref name="frames"/> frames as upstream
    /// <c>load_intrinsics</c> does: one entry is broadcast, an equal count is copied, a longer sequence is truncated, and a
    /// shorter one is resampled linearly over normalized time (single-precision time grids, double-precision interpolation).</summary>
    public static PinholeIntrinsics[] FitOverTime(IReadOnlyList<PinholeIntrinsics> source, int frames)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentOutOfRangeException.ThrowIfLessThan(frames, 1);
        if (source.Count == 0)
            throw new ArgumentException("No intrinsics were given.", nameof(source));
        PinholeIntrinsics[] fitted = new PinholeIntrinsics[frames];
        if (source.Count == 1 || source.Count >= frames)
        {
            for (int i = 0; i < frames; i++)
                fitted[i] = source[source.Count == 1 ? 0 : i];
            return fitted;
        }

        double[] oldTime = LinspaceSingle(source.Count);
        double[] newTime = LinspaceSingle(frames);
        for (int i = 0; i < frames; i++)
        {
            fitted[i] = new PinholeIntrinsics(
                Interpolate(newTime[i], oldTime, source, static k => k.Fx),
                Interpolate(newTime[i], oldTime, source, static k => k.Fy),
                Interpolate(newTime[i], oldTime, source, static k => k.Cx),
                Interpolate(newTime[i], oldTime, source, static k => k.Cy));
        }
        return fitted;
    }

    /// <summary>Scales pixel intrinsics to a latent grid: x terms by <c>latentWidth / pixelWidth</c>, y terms by
    /// <c>latentHeight / pixelHeight</c>.</summary>
    public static PinholeIntrinsics ToLatent(PinholeIntrinsics pixel, int pixelWidth, int pixelHeight, int latentWidth, int latentHeight)
    {
        float sx = (float)(latentWidth / (double)pixelWidth);
        float sy = (float)(latentHeight / (double)pixelHeight);
        return new PinholeIntrinsics(pixel.Fx * sx, pixel.Fy * sy, pixel.Cx * sx, pixel.Cy * sy);
    }

    /// <summary>Packs relative poses and latent intrinsics (one per raw frame) into the DiT camera inputs. Latent frame
    /// <c>i</c> samples raw frame <c>8i</c> for its condition row and raw frames <c>max(0, 8i − 7)</c> onward (eight of
    /// them, the last repeated when the clip runs out) for its Plücker channels, with rays at integer pixel positions.</summary>
    /// <returns>The conditions (<c>T·20</c> floats) and the host F32 chunk Plücker <c>[1, 48, T, H, W]</c>, owned by the caller.</returns>
    public static (float[] Conditions, Tensor ChunkPlucker) Pack(IReadOnlyList<float[]> relativePoses,
        IReadOnlyList<PinholeIntrinsics> latentIntrinsics, int latentHeight, int latentWidth)
    {
        ArgumentNullException.ThrowIfNull(relativePoses);
        ArgumentNullException.ThrowIfNull(latentIntrinsics);
        int frames = relativePoses.Count;
        if (frames == 0 || latentIntrinsics.Count != frames)
            throw new ArgumentException($"Need one intrinsics entry per pose; got {latentIntrinsics.Count} for {frames} poses.",
                nameof(latentIntrinsics));
        return PackRange(relativePoses, latentIntrinsics, latentHeight, latentWidth, 0, (frames - 1) / TemporalStride + 1);
    }

    /// <summary><see cref="Pack"/> for latent frames <c>[firstLatentFrame, firstLatentFrame + latentFrames)</c> only: the camera
    /// inputs of one chunk of a stream, whose path need only reach the chunk's last raw frame.</summary>
    /// <param name="relativePoses">Poses relative to frame 0, one per raw frame, reaching at least raw frame
    /// <c>8·(firstLatentFrame + latentFrames − 1)</c>.</param>
    /// <param name="latentIntrinsics">One entry for every raw frame, or one per raw frame covering the same range.</param>
    /// <returns>The conditions (<c>latentFrames·20</c> floats) and the host F32 chunk Plücker
    /// <c>[1, 48, latentFrames, H, W]</c>, owned by the caller.</returns>
    public static (float[] Conditions, Tensor ChunkPlucker) PackRange(IReadOnlyList<float[]> relativePoses,
        IReadOnlyList<PinholeIntrinsics> latentIntrinsics, int latentHeight, int latentWidth, int firstLatentFrame, int latentFrames)
    {
        ArgumentNullException.ThrowIfNull(relativePoses);
        ArgumentNullException.ThrowIfNull(latentIntrinsics);
        ArgumentOutOfRangeException.ThrowIfLessThan(latentHeight, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(latentWidth, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(firstLatentFrame);
        ArgumentOutOfRangeException.ThrowIfLessThan(latentFrames, 1);
        int frames = relativePoses.Count;
        int lastRaw = (firstLatentFrame + latentFrames - 1) * TemporalStride;
        if (lastRaw >= frames)
            throw new ArgumentException($"Latent frame {firstLatentFrame + latentFrames - 1} samples raw frame {lastRaw}; the path "
                + $"has {frames} poses.", nameof(relativePoses));
        if (latentIntrinsics.Count != 1 && latentIntrinsics.Count <= lastRaw)
            throw new ArgumentException($"Need one intrinsics entry, or one per raw frame through {lastRaw}; got "
                + $"{latentIntrinsics.Count}.", nameof(latentIntrinsics));

        float[] conditions = new float[latentFrames * ConditionWidth];
        int plane = latentHeight * latentWidth;
        Tensor plucker = new(new TensorShape([1, PluckerChannels, latentFrames, latentHeight, latentWidth]), DType.F32, DeviceKind.Cpu);
        bool built = false;
        try
        {
            Span<float> destination = plucker.AsSpan<float>();
            float[] rays = new float[plane * PluckerEmbedding.Channels];
            for (int t = 0; t < latentFrames; t++)
            {
                int index = (firstLatentFrame + t) * TemporalStride;
                relativePoses[index].AsSpan(0, 16).CopyTo(conditions.AsSpan(t * ConditionWidth));
                PinholeIntrinsics k = IntrinsicsAt(latentIntrinsics, index);
                conditions[t * ConditionWidth + 16] = k.Fx;
                conditions[t * ConditionWidth + 17] = k.Fy;
                conditions[t * ConditionWidth + 18] = k.Cx;
                conditions[t * ConditionWidth + 19] = k.Cy;

                int start = Math.Max(0, index - (TemporalStride - 1));
                for (int slot = 0; slot < TemporalStride; slot++)
                {
                    int raw = Math.Min(start + slot, frames - 1);
                    PinholeIntrinsics rawK = IntrinsicsAt(latentIntrinsics, raw);
                    PluckerEmbedding.Compute(relativePoses[raw], rawK.Fx, rawK.Fy, rawK.Cx, rawK.Cy, latentHeight, latentWidth, rays,
                        0f, PluckerLayout.DirectionMoment);
                    for (int c = 0; c < PluckerEmbedding.Channels; c++)
                    {
                        int channel = slot * PluckerEmbedding.Channels + c;
                        Span<float> target = destination.Slice((channel * latentFrames + t) * plane, plane);
                        for (int p = 0; p < plane; p++)
                            target[p] = rays[p * PluckerEmbedding.Channels + c];
                    }
                }
            }
            built = true;
            return (conditions, plucker);
        }
        finally
        {
            if (!built)
                plucker.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Dispose() => Interlocked.Exchange(ref _chunkPlucker, null)?.Dispose();

    private static PinholeIntrinsics IntrinsicsAt(IReadOnlyList<PinholeIntrinsics> intrinsics, int rawFrame) =>
        intrinsics[intrinsics.Count == 1 ? 0 : rawFrame];

    // np.linspace(0, 1, n, dtype=float32): i/(n−1) in double, the last point exactly 1, rounded to single.
    private static double[] LinspaceSingle(int count)
    {
        double[] values = new double[count];
        double step = count > 1 ? 1.0 / (count - 1) : 0.0;
        for (int i = 0; i < count; i++)
            values[i] = (float)(i * step);
        values[count - 1] = 1.0;
        return values;
    }

    // np.interp on one intrinsics component: left-closed segments, the last sample returned at the right end.
    private static float Interpolate(double x, double[] xp, IReadOnlyList<PinholeIntrinsics> fp, Func<PinholeIntrinsics, float> component)
    {
        int last = xp.Length - 1;
        if (x >= xp[last])
            return component(fp[last]);
        if (x <= xp[0])
            return component(fp[0]);
        int j = 0;
        while (j + 1 <= last && xp[j + 1] <= x)
            j++;
        double y0 = component(fp[j]);
        if (xp[j] == x)
            return (float)y0;
        double y1 = component(fp[j + 1]);
        double slope = (y1 - y0) / (xp[j + 1] - xp[j]);
        return (float)(slope * (x - xp[j]) + y0);
    }
}
