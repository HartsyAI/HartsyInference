using HartsyInference.Cuda;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>QA gate item: "tile-seam visibility audit (un-tiled F32 reference vs. tiled, a few resolutions)".
/// Decodes the same real latent through <see cref="VaeDecoder.Decode"/> (single full-res pass, ground truth) and
/// <see cref="VaeDecoder.DecodeTiled"/> (production tile loop), diffs pixel-by-pixel, and specifically checks
/// error concentrated at tile boundaries vs. tile interiors — a real seam artifact shows up as a boundary-biased
/// diff, not a uniform one. The encoder-side tiled path (<see cref="VaeTiledEncoder"/>, Tier 1.2) is NOT covered
/// here: it's still disconnected from production (see the comment at <c>SdxlPipeline.BuildInitialLatent</c> —
/// routing through it reproducibly segfaults inside libcuda.so, unresolved as of this session) — nothing to
/// audit for seams on a path nothing calls yet.</summary>
[Trait("Category", "Integration")]
public sealed class VaeTileSeamAuditTests
{
    private readonly ITestOutputHelper _output;
    public VaeTileSeamAuditTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void DecodeTiled_MatchesUntiledReference_NoBoundaryBiasedSeam()
    {
        if (!File.Exists(TestPaths.Sdxl.SingleFile))
        {
            _output.WriteLine($"SKIPPED: SDXL checkpoint not found at {TestPaths.Sdxl.SingleFile}.");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }

        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) = SdxlCheckpointConverter.LoadAndConvert(TestPaths.Sdxl.SingleFile);
        using (loader)
        using (CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir))
        {
            // This checkpoint's raw VAE weights are F16 (confirmed via the safetensors header) — the real
            // production path never runs those raw (VaePrecisionHelper force-upcasts, Tier 1.1). Mirror that
            // here: an unfixed F16 decode collapses to solid black regardless of tiling, which would make this
            // audit vacuous (both paths equally black tells you nothing about seams).
            DType vaeDtype = VaePrecisionHelper.PreferredVaeDtype(backend);
            Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(converted.Vae, vaeDtype);
            VaeDecoder vae = new VaeDecoder(VaeConfig.Sdxl);
            vae.LoadWeights(vaeWeights);
            VaeEncoder encoder = new VaeEncoder(VaeConfig.Sdxl);
            encoder.LoadWeights(vaeWeights);
            _output.WriteLine($"VAE weight dtype: raw={converted.Vae.Values.First().DType.Name}, used={vaeDtype.Name} (VaePrecisionHelper policy).");

            // 1536x1536 image -> 192x192 latent (8x downsample). Well above the 64-latent tile size used by
            // DecodeTiled below, forcing a real multi-tile grid (not the single-tile short-circuit). The latent
            // comes from a real VAE-encoded image, not a hand-crafted tensor — a synthetic latent is out of the
            // decoder's training distribution and decodes to near-solid-black (tried first, useless to eyeball).
            const int size = 1536;
            const int tileLatentSize = 64;
            const int overlapLatent = 8;
            using Tensor sourceImage = ImagePostProcessor.RgbBytesToTensor(PhotoLikeRgb(size), size, size);
            Tensor latent = encoder.Encode(backend, sourceImage);

            Tensor refDecode = vae.Decode(backend, latent);
            Tensor tiledDecode = vae.DecodeTiled(backend, latent, tileLatentSize, overlapLatent);

            Assert.Equal(refDecode.Shape[2], tiledDecode.Shape[2]);
            Assert.Equal(refDecode.Shape[3], tiledDecode.Shape[3]);
            int height = (int)refDecode.Shape[2];
            int width = (int)refDecode.Shape[3];

            byte[] refRgb = ImagePostProcessor.TensorToRgbBytes(refDecode);
            byte[] tiledRgb = ImagePostProcessor.TensorToRgbBytes(tiledDecode);

            // Tile boundaries in pixel space: each latent tile edge * 8 (VAE upsample factor).
            int latentSize = size / 8;
            HashSet<int> boundaryPx = new HashSet<int>();
            for (int t = tileLatentSize; t < latentSize; t += tileLatentSize)
            {
                boundaryPx.Add(t * 8);
            }

            double boundarySum = 0; long boundaryCount = 0;
            double interiorSum = 0; long interiorCount = 0;
            const int band = 3; // pixels within this distance of a boundary line count as "boundary"
            for (int y = 0; y < height; y++)
            {
                bool yNearBoundary = boundaryPx.Any(b => Math.Abs(y - b) <= band);
                for (int x = 0; x < width; x++)
                {
                    bool xNearBoundary = boundaryPx.Any(b => Math.Abs(x - b) <= band);
                    int i = (y * width + x) * 3;
                    int diff = Math.Abs(refRgb[i] - tiledRgb[i]) + Math.Abs(refRgb[i + 1] - tiledRgb[i + 1]) + Math.Abs(refRgb[i + 2] - tiledRgb[i + 2]);
                    if (yNearBoundary || xNearBoundary)
                    {
                        boundarySum += diff; boundaryCount++;
                    }
                    else
                    {
                        interiorSum += diff; interiorCount++;
                    }
                }
            }
            double boundaryMean = boundarySum / Math.Max(1, boundaryCount) / 3.0;
            double interiorMean = interiorSum / Math.Max(1, interiorCount) / 3.0;
            _output.WriteLine($"Boundary-band mean abs diff: {boundaryMean:F3} (n={boundaryCount}). Interior mean abs diff: {interiorMean:F3} (n={interiorCount}).");

            Directory.CreateDirectory(TestPaths.OutputDir);
            File.WriteAllBytes(Path.Combine(TestPaths.OutputDir, "vae_tileseam_untiled_reference.rgb"), refRgb);
            File.WriteAllBytes(Path.Combine(TestPaths.OutputDir, "vae_tileseam_tiled.rgb"), tiledRgb);
            _output.WriteLine($"Wrote both {width}x{height} RGB24 outputs for visual inspection.");

            refDecode.Dispose();
            tiledDecode.Dispose();
            latent.Dispose();

            // The real failure mode this catches: a seam shows up as boundary error many times larger than
            // interior error. Some non-zero diff everywhere is expected (tiled decode is not required to be
            // bit-exact — it's a different receptive-field/blend path) so this checks bias, not zero-diff.
            Assert.True(boundaryMean < interiorMean * 3.0 + 2.0,
                $"Tile-boundary error ({boundaryMean:F3}) is disproportionately larger than interior error ({interiorMean:F3}) — looks like a visible seam.");
        }
    }

    /// <summary>A gradient-plus-shapes RGB image — real photo-adjacent structure (not flat, not full noise) so
    /// the VAE encoder produces an in-distribution latent, unlike a hand-crafted latent tensor (tried first: it
    /// decodes to near-solid-black — out of the decoder's training distribution, useless to eyeball).</summary>
    private static byte[] PhotoLikeRgb(int size)
    {
        byte[] rgb = new byte[size * size * 3];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int i = (y * size + x) * 3;
                float gx = x / (float)size;
                float gy = y / (float)size;
                rgb[i] = (byte)(gx * 255);
                rgb[i + 1] = (byte)(gy * 255);
                rgb[i + 2] = (byte)(128 + 100 * Math.Sin(gx * 12) * Math.Cos(gy * 12));
                // A few high-contrast blobs so both tiles and tile boundaries land on real edges/detail.
                for (int cx = size / 6; cx <= size; cx += size / 4)
                {
                    for (int cy = size / 6; cy <= size; cy += size / 4)
                    {
                        int dx = x - cx, dy = y - cy;
                        if (dx * dx + dy * dy < (size / 16) * (size / 16))
                        {
                            rgb[i] = 255; rgb[i + 1] = 40; rgb[i + 2] = 40;
                        }
                    }
                }
            }
        }
        return rgb;
    }
}
