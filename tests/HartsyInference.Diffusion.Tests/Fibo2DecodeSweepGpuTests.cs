using System.Security.Cryptography;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Features;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>VAE decode peak sweep for the Fibo 2.0 B3 follow-up. Decodes one fixed 1024 x 1024 latent at tile sizes 64, 32 and 16 with the
/// real FLUX.2 VAE and logs the device used and the pool reserved after each decode. It needs the real VAE, so it skips unless the
/// environment variable HARTSY_FLUX2_VAE names the file (the catalog build, Comfy-Org/Fibo-2.0 vae/flux2-vae.safetensors, sha256
/// 868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3).
/// Pass: every tile size returns a 1024 x 1024 image, and tile 32 and tile 64 agree within 2 levels per channel. The peaks are recorded,
/// not gated: the tile size choice is a follow-up.</summary>
public sealed class Fibo2DecodeSweepGpuTests(ITestOutputHelper output)
{
    private const long MiB = 1L << 20;

    /// <summary>The fact the sweep runs under: reported as skipped, never passed, when HARTSY_FLUX2_VAE is unset or not a file, or when no CUDA
    /// device opens.</summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    public sealed class Flux2VaeCudaFactAttribute : FactAttribute
    {
        public Flux2VaeCudaFactAttribute()
        {
            string? path = Environment.GetEnvironmentVariable("HARTSY_FLUX2_VAE");
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                Skip = "SKIPPED: HARTSY_FLUX2_VAE is unset or not an existing file; the decode sweep did not run.";
            else if (GpuDevices.CudaFailure is { } failure)
                Skip = $"SKIPPED: no usable CUDA device ({failure}); the decode sweep did not run.";
        }
    }

    [Flux2VaeCudaFact]
    [Trait("Category", "GpuIntegration")]
    public void DecodeTileSweep_Run()
    {
        string vaePath = Environment.GetEnvironmentVariable("HARTSY_FLUX2_VAE")!;

        using CudaBackend backend = new(0, PtxDirectory());
        using Flux2VaeComponents vae = Flux2VaeComponents.Load(vaePath, "fibo2");
        Tensor latent = SeedGenerator.CreateNoise(new TensorShape(1, 32, 128, 128), 7);
        try
        {
            byte[]? reference = null;
            foreach (int tile in new[] { 64, 32, 16 })
            {
                backend.TrimMemoryPool();
                DeviceMemoryUsage before = backend.GetDeviceMemoryUsage();
                using Tensor image = vae.Decoder.DecodeTiled(backend, latent, tile, 8);
                DeviceMemoryUsage after = backend.GetDeviceMemoryUsage();
                byte[] rgb = ImagePostProcessor.TensorToRgbBytes(image);
                output.WriteLine($"tile {tile}: image {image.Shape}, used before {(before.TotalBytes - before.FreeBytes) / MiB} MiB, " +
                    $"after {(after.TotalBytes - after.FreeBytes) / MiB} MiB, pool reserved {after.PoolReservedBytes / MiB} MiB, " +
                    $"sha256 {Convert.ToHexString(SHA256.HashData(rgb)).ToLowerInvariant()[..16]}");

                Assert.Equal(1024, image.Shape[2]);
                Assert.Equal(1024, image.Shape[3]);
                if (tile == 64)
                    reference = rgb;
                else if (tile == 32)
                {
                    int worst = 0;
                    for (int i = 0; i < rgb.Length; i++)
                        worst = Math.Max(worst, Math.Abs(rgb[i] - reference![i]));
                    output.WriteLine($"tile 32 vs tile 64: max level difference {worst}");
                    Assert.True(worst <= 2, $"tile 32 differs from tile 64 by {worst} levels");
                }
            }
        }
        finally
        {
            latent.Dispose();
        }
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("HartsyInference.Cuda/Ptx was not found above the test binary.");
    }
}
