using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight verification for Tier 3.4's first gate (per the plan: build ONLY
/// <see cref="LtxVideoVaeEncoder"/>, verify it stand-alone via an encode→decode round trip through the existing,
/// already-shipped <see cref="LtxVideoVaeDecoder"/>, before any transformer/pipeline conditioning work). No new
/// image-to-video machinery is exercised here — this is purely "does the new encoder produce a latent the
/// existing decoder can turn back into a recognizable image."
/// <para>Checkpoint: the real production single-file <c>ltx-video-2b-v0.9.safetensors</c> (base 0.9, NOT the
/// standalone <c>Models/VAE/LTXV/ltxv_vae.safetensors</c> file — that one turned out to be a different/orphan
/// variant not used by any current code path; the bundled checkpoint is what <c>LtxVideoRecipe</c> actually
/// loads). Both encoder and decoder read the SAME <c>LtxVideoCheckpointConverter</c> output, so this also
/// exercises the converter's encoder-key regrouping (carried since the converter was written, per its own doc
/// comment, but never consumed by anything until now).</para>
/// <para><b>Real finding from reading the actual diffusers source before writing the encoder</b> (not assumed by
/// decoder symmetry): <c>AutoencoderKLLTXVideo</c> defaults <c>encoder_causal=True</c>, <c>decoder_causal=False</c>
/// — the two halves of the SAME checkpoint use different causal padding. Assuming symmetry would have silently
/// produced wrong padding on the encoder side.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class LtxVideoVaeEncoderRoundTripRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public LtxVideoVaeEncoderRoundTripRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void LtxVideoVaeEncoder_EncodeThenDecode_ReconstructsARecognizableImage()
    {
        string checkpointPath = TestPaths.LtxVideo.SingleFile;
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: LTX-Video 0.9 checkpoint not found at {checkpointPath}.");
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

        (LtxVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader loader) =
            LtxVideoCheckpointConverter.LoadAndConvert(checkpointPath);
        try
        {
            Assert.True(conv.Vae.Keys.Any(k => k.StartsWith("encoder.", StringComparison.Ordinal)),
                "Converted VAE weights have no encoder.* keys — the converter's encoder regrouping may have regressed.");

            using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
            Dictionary<string, Core.Tensors.Tensor> vaeF32 = HartsyInference.Engine.Features.VaePrecisionHelper.CastVaeWeights(conv.Vae, Core.Tensors.DType.F32);

            LtxVideoVaeEncoder encoder = new LtxVideoVaeEncoder();
            encoder.LoadWeights(vaeF32);
            LtxVideoVaeDecoder decoder = new LtxVideoVaeDecoder();
            decoder.LoadWeights(vaeF32);

            // Advisor-flagged failure mode this round-trip test alone cannot catch: MapVae only renames
            // per_channel_statistics.* -> latents_mean/latents_std on original-naming checkpoints. If that
            // detection ever regressed and the stats went missing, Normalize (encoder) and Denormalize (decoder)
            // would BOTH silently become no-ops -- the same missing scale cancels on both sides, so a round trip
            // through this SAME encoder+decoder pair would still look perfect while feeding an un-normalized-space
            // latent to anything else (e.g. a transformer) that expects the normalized one. Assert the stats were
            // actually found, not just that the round trip looks right.
            Assert.True(encoder.HasLatentStats, "Encoder found no latents_mean/latents_std -- normalization is silently a no-op (see LtxVideoVaeEncoder.HasLatentStats doc).");
            Assert.True(decoder.HasLatentStats, "Decoder found no latents_mean/latents_std -- denormalization is silently a no-op (see LtxVideoVaeDecoder.HasLatentStats doc).");

            const int size = 256; // divisible by patch(4)*2^3=32
            byte[] original = SyntheticPattern(size, size);
            Core.Tensors.Tensor rgbIn = RgbToTensor5d(original, size, size);

            Core.Tensors.Tensor latent = encoder.Encode(backend, rgbIn);
            rgbIn.Dispose();
            _output.WriteLine($"Latent shape: [{latent.Shape[0]},{latent.Shape[1]},{latent.Shape[2]},{latent.Shape[3]},{latent.Shape[4]}].");
            (float mean, float std) = LatentMeanStd(latent);
            _output.WriteLine($"Latent per-element mean={mean:F4}, std={std:F4} (normalized space should be roughly unit-scale, not near-zero/near-input-valued).");

            Core.Tensors.Tensor decoded = decoder.Decode(backend, latent);
            byte[] roundTripped = Tensor5dToRgb(decoded, out int outW, out int outH);
            decoded.Dispose();

            Assert.Equal(size, outW);
            Assert.Equal(size, outH);

            // Advisor-flagged second check: a mean-abs-diff-of-1.09 round trip is also consistent with a
            // pass-through bug (encoder/decoder bypassing the latent entirely). Perturb the SAME latent (zero it)
            // and decode again -- if the decode is actually load-bearing on the latent's content, the output must
            // change substantially, not reproduce the same quadrant pattern.
            Core.Tensors.Tensor zeroed = new Core.Tensors.Tensor(latent.Shape, Core.Tensors.DType.F32); // zero-initialized
            Core.Tensors.Tensor decodedZero = decoder.Decode(backend, zeroed);
            byte[] fromZero = Tensor5dToRgb(decodedZero, out _, out _);
            zeroed.Dispose();
            decodedZero.Dispose();
            latent.Dispose();

            long zeroDiffSum = 0;
            for (int i = 0; i < roundTripped.Length; i++) zeroDiffSum += Math.Abs(roundTripped[i] - fromZero[i]);
            double meanAbsDiffFromZero = zeroDiffSum / (double)roundTripped.Length;
            _output.WriteLine($"Mean absolute per-byte difference (real latent vs zeroed latent, both decoded): {meanAbsDiffFromZero:F2}.");
            Assert.True(meanAbsDiffFromZero > 20.0, $"Decoding a zeroed latent produced nearly the same image as the real latent (diff {meanAbsDiffFromZero:F2}) -- the latent may not be load-bearing (encoder/decoder bypass).");

            File.WriteAllBytes(Path.Combine(RepoRoot.Path, "ltx_vae_roundtrip_original.rgb"), original);
            File.WriteAllBytes(Path.Combine(RepoRoot.Path, "ltx_vae_roundtrip_decoded.rgb"), roundTripped);
            _output.WriteLine($"Wrote {size}x{size} original + round-tripped RGB for visual inspection.");

            long diffSum = 0;
            for (int i = 0; i < original.Length; i++) diffSum += Math.Abs(original[i] - roundTripped[i]);
            double meanAbsDiff = diffSum / (double)original.Length;
            _output.WriteLine($"Mean absolute per-byte difference (original vs round-tripped): {meanAbsDiff:F2}.");

            // A VAE round trip is lossy by construction (128-channel bottleneck at 1/32 spatial resolution) — this
            // is not a near-zero-diff check. It only rules out the two failure modes a wrong-but-shape-compatible
            // encoder would produce: a flat/collapsed output (encoder emits ~constant garbage) or full-scale static
            // (encoder emits noise the decoder can't structure). Real pass/fail is the visual inspection below.
            Assert.True(meanAbsDiff < 80.0, $"Round-trip mean abs diff ({meanAbsDiff:F2}) is too large for a lossy-but-structured VAE reconstruction — likely a wrong encoder, not just compression loss.");
        }
        finally
        {
            loader.Dispose();
        }
    }

    /// <summary>Four color quadrants plus a diagonal gradient — distinct enough that blur, color-channel swaps,
    /// or spatial misalignment are all visually obvious on inspection, unlike a solid color.</summary>
    private static byte[] SyntheticPattern(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = (y * width + x) * 3;
                bool left = x < width / 2, top = y < height / 2;
                byte r, g, b;
                if (left && top) { r = 220; g = 40; b = 40; }        // red
                else if (!left && top) { r = 40; g = 200; b = 60; }  // green
                else if (left && !top) { r = 40; g = 80; b = 220; }  // blue
                else { r = 230; g = 210; b = 40; }                    // yellow
                byte grad = (byte)(255 * (x + y) / (width + height));
                rgb[i] = (byte)((r + grad) / 2);
                rgb[i + 1] = (byte)((g + grad) / 2);
                rgb[i + 2] = (byte)((b + grad) / 2);
            }
        }
        return rgb;
    }

    private static unsafe (float Mean, float Std) LatentMeanStd(Core.Tensors.Tensor latent)
    {
        long n = latent.Shape.ElementCount;
        float* p = (float*)latent.DataPointer;
        double sum = 0, sumSq = 0;
        for (long i = 0; i < n; i++) { double v = p[i]; sum += v; sumSq += v * v; }
        double mean = sum / n;
        double variance = sumSq / n - mean * mean;
        return ((float)mean, (float)Math.Sqrt(Math.Max(0, variance)));
    }

    private static unsafe Core.Tensors.Tensor RgbToTensor5d(byte[] rgb, int width, int height)
    {
        Core.Tensors.Tensor t = new Core.Tensors.Tensor(new Core.Tensors.TensorShape([1, 3, 1, height, width]), Core.Tensors.DType.F32);
        float* p = (float*)t.DataPointer;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int i = (y * width + x) * 3;
                for (int c = 0; c < 3; c++)
                    p[c * height * width + y * width + x] = (rgb[i + c] / 255.0f) * 2.0f - 1.0f;
            }
        return t;
    }

    private static unsafe byte[] Tensor5dToRgb(Core.Tensors.Tensor t, out int width, out int height)
    {
        int f = (int)t.Shape[2];
        height = (int)t.Shape[3];
        width = (int)t.Shape[4];
        float* p = (float*)t.DataPointer;
        byte[] rgb = new byte[height * width * 3];
        // Middle frame (or the only one) — the round trip here is a single still image.
        int fi = f / 2;
        long frame = (long)height * width;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int i = (y * width + x) * 3;
                for (int c = 0; c < 3; c++)
                {
                    float v = p[((long)c * f + fi) * frame + y * width + x];
                    v = Math.Clamp((v + 1.0f) * 0.5f, 0f, 1f);
                    rgb[i + c] = (byte)(v * 255.0f + 0.5f);
                }
            }
        return rgb;
    }
}
