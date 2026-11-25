using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.DepthPro2;
using Xunit.Abstractions;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>
/// How often Depth Pro 2 moves tensors between the host and the CUDA device while it runs one warm image, three joint views and an unbuffered video,
/// on the S weights with synthetic frames: device-to-host syncs (<c>GetD2hSyncCount</c>) and residency-cache misses (<c>GetGpuCacheStats</c>; each miss
/// is a host-to-device upload). It asserts only that the regions run, and prints one <c>D2H_SYNCS</c> line per region
/// so a change can be compared with a baseline run of the same test. Needs <c>HARTSY_DEPTHPRO2_WEIGHTS</c>, <c>HARTSY_DEPTHPRO2_FOCAL_WEIGHTS</c> and a
/// CUDA device; otherwise it is reported as skipped (a skip is not a measurement).
/// </summary>
[Trait("Category", "GpuIntegration")]
[Collection(DepthPro2GpuKnobsCollection.Name)]
public sealed class DepthPro2DeviceTrafficTests(ITestOutputHelper output)
{
    private const string WeightsVariable = "HARTSY_DEPTHPRO2_WEIGHTS";
    private const string FocalVariable = "HARTSY_DEPTHPRO2_FOCAL_WEIGHTS";
    private static readonly Lazy<string?> CudaFailure = new(Probe);

    [DeviceTrafficFact]
    public void CountDeviceToHostSyncs()
    {
        using SafeTensorsLoader depthLoader = Load(WeightsVariable);
        using SafeTensorsLoader focalLoader = Load(FocalVariable);
        DepthPro2Model model = new(new DepthPro2Network(DepthPro2Preset.Small, depthLoader.GetAllTensors()));
        DepthPro2FocalEstimator estimator = new(focalLoader.GetAllTensors());
        using CudaBackend device = new(0, Ernie5ModelGpuParityTests.PtxDirectory());

        // One warm image at 1024x576: the first run uploads the weights, the counted one is steady state.
        const int imageW = 1024, imageH = 576;
        byte[] image = Frame(imageW, imageH, 0);
        model.Predict(device, [image], imageW, imageH, null, estimator);
        (long imageSyncs, long imageUploads, double imageMs) = CountedValue(device, () => model.Predict(device, [image], imageW, imageH, null, estimator));

        // Three joint 384x216 views.
        const int viewW = 384, viewH = 216;
        byte[][] views = [Frame(viewW, viewH, 1), Frame(viewW, viewH, 2), Frame(viewW, viewH, 3)];
        model.Predict(device, views, viewW, viewH, null, estimator);
        (long viewSyncs, long viewUploads, double viewMs) = CountedValue(device, () => model.Predict(device, views, viewW, viewH, null, estimator));

        // An unbuffered video of 6 frames at 384x216 with a window of 4: frames 0 to 2 fill the ring, 3 to 5 run with a full ring.
        const int window = 4, frames = 6;
        long[] perFrame = new long[frames], perFrameUploads = new long[frames];
        double[] perFrameMs = new double[frames];
        using (DepthPro2VideoStream warm = new(model.Network, device, viewW, viewH, 600, window, 0, DepthPro2VideoMode.Unbuffered))
            for (int i = 0; i < frames; i++) warm.Step(Frame(viewW, viewH, 10 + i));
        using (DepthPro2VideoStream stream = new(model.Network, device, viewW, viewH, 600, window, 0, DepthPro2VideoMode.Unbuffered))
            for (int i = 0; i < frames; i++)
            {
                byte[] frame = Frame(viewW, viewH, 10 + i);
                (perFrame[i], perFrameUploads[i], perFrameMs[i]) = CountedValue(device, () => stream.Step(frame));
            }

        output.WriteLine($"D2H_SYNCS image {imageW}x{imageH}: {imageSyncs} syncs, {imageUploads} uploads, {imageMs:F0} ms");
        output.WriteLine($"D2H_SYNCS views 3x{viewW}x{viewH}: {viewSyncs} syncs, {viewUploads} uploads, {viewMs:F0} ms");
        output.WriteLine($"D2H_SYNCS unbuffered window {window} {viewW}x{viewH} per frame: [{string.Join(", ", perFrame)}] syncs, "
            + $"[{string.Join(", ", perFrameUploads)}] uploads, [{string.Join(", ", perFrameMs.Select(ms => ms.ToString("F0")))}] ms");
        Assert.True(imageSyncs >= 0 && viewSyncs >= 0 && perFrame.All(count => count >= 0));
    }

    private static (long Syncs, long Uploads, double Milliseconds) Counted(CudaBackend device, Action work)
    {
        device.ResetD2hSyncCount();
        long missesBefore = device.GetGpuCacheStats().misses;
        Stopwatch clock = Stopwatch.StartNew();
        work();
        return (device.GetD2hSyncCount(), device.GetGpuCacheStats().misses - missesBefore, clock.Elapsed.TotalMilliseconds);
    }

    private static (long Syncs, long Uploads, double Milliseconds) CountedValue<T>(CudaBackend device, Func<T> work) => Counted(device, () => { _ = work(); });

    // A smooth gradient with a few sinusoids and mild noise, the same for a given seed.
    private static byte[] Frame(int width, int height, int seed)
    {
        Random rng = new(1000 + seed);
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                double u = x / (double)width, v = y / (double)height;
                double r = 0.4 + 0.4 * Math.Sin(6 * u + seed) * Math.Cos(4 * v), g = 0.5 - 0.3 * u + 0.2 * v, b = 0.3 + 0.5 * Math.Sin(5 * v + 0.5 * seed);
                int o = (y * width + x) * 3;
                rgb[o] = Clamp(r + rng.NextDouble() * 0.04);
                rgb[o + 1] = Clamp(g + rng.NextDouble() * 0.04);
                rgb[o + 2] = Clamp(b + rng.NextDouble() * 0.04);
            }

        return rgb;
    }

    private static byte Clamp(double value) => (byte)Math.Round(Math.Clamp(value, 0, 1) * 255);

    private static SafeTensorsLoader Load(string variable)
    {
        SafeTensorsLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable(variable)!);
        return loader;
    }

    private static string? Probe()
    {
        try
        {
            using IBackend backend = new CudaBackend(0, Ernie5ModelGpuParityTests.PtxDirectory());
            return null;
        }
        catch (Exception ex)
        {
            return $"{ex.GetType().Name}: {ex.Message}";
        }
    }

    /// <summary>A fact reported as skipped when the weights or a CUDA device are missing.</summary>
    private sealed class DeviceTrafficFactAttribute : FactAttribute
    {
        public DeviceTrafficFactAttribute()
        {
            foreach (string name in new[] { WeightsVariable, FocalVariable })
            {
                string? path = Environment.GetEnvironmentVariable(name);
                if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                {
                    Skip = $"SKIPPED: {name} is unset or not an existing file; nothing was measured.";
                    return;
                }
            }

            if (CudaFailure.Value is { } failure) Skip = $"SKIPPED: no usable CUDA device ({failure}); nothing was measured.";
        }
    }
}
