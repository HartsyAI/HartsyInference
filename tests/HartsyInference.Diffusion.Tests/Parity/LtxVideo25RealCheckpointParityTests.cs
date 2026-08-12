using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Loads the real LTX-2.5 video VAE and decodes the smallest legal latent through
/// <see cref="LtxVideo25DiffusionDecoder"/>. The point is the key mapping and the geometry against the actual
/// checkpoint: every <c>decoder.*</c> tensor must be consumed except <c>decoder.type_emb</c>, which neither reference
/// implementation reads. A full-resolution decode is out of reach here — the managed <see cref="IBackend.Na3d"/> is a
/// numerical reference, not a performance path.</summary>
public sealed unsafe class LtxVideo25RealCheckpointParityTests
{
    [Fact]
    [Trait("Category", "Integration")]
    public void DecodesTheSmallestLegalLatentFromTheRealCheckpoint()
    {
        string path = TestPaths.LtxVideo2.VideoVae25;
        if (!File.Exists(path)) return;   // tier-lint: guarded

        (LtxVideo2CheckpointConverter.ConvertedWeights weights, SafeTensorsLoader loader) =
            LtxVideo2CheckpointConverter.LoadAndConvert(path);
        using (loader)
        {
            Assert.True(LtxVideo2CheckpointConverter.IsDiffusionVideoVae(loader.Descriptors.Keys));
            Assert.Equal(310, weights.VaeDiffusionDecoder.Count);

            using LtxVideo25DiffusionDecoder decoder = new LtxVideo25DiffusionDecoder(
                config: null, latentsMean: Channels(weights.Vae, "latents_mean"), latentsStd: Channels(weights.Vae, "latents_std"));
            decoder.LoadWeights(weights.VaeDiffusionDecoder);

            string[] unconsumed = [.. weights.VaeDiffusionDecoder.Keys.Except(decoder.ConsumedKeys).Order()];
            Assert.Equal(["decoder.type_emb"], unconsumed);

            LtxVideo25DiffusionDecoderConfig config = decoder.Config;
            const int latentFrames = 1, latentHeight = 2, latentWidth = 2;
            using Tensor latent = Fill(new TensorShape([1, config.InChannels, latentFrames, latentHeight, latentWidth]), 12345);
            using Tensor noise = Fill(decoder.NoiseShape(latentFrames, latentHeight, latentWidth), 987);

            IBackend backend = new CpuBackend();
            using Tensor pixels = decoder.Decode(backend, latent, noise);

            Assert.Equal(new TensorShape([1, 3, config.OutputFrames(latentFrames),
                latentHeight * config.SpatialUpscale, latentWidth * config.SpatialUpscale]), pixels.Shape);
            float* values = (float*)pixels.DataPointer;
            for (long i = 0; i < pixels.ElementCount; i++)
                Assert.True(float.IsFinite(values[i]), $"pixel {i} is {values[i]}");
        }
    }

    private static float[]? Channels(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? tensor)) return null;
        using Tensor f32 = tensor.DType == DType.F32 ? tensor.To(tensor.Device) : tensor.CastTo(DType.F32);
        float[] values = new float[f32.ElementCount];
        new Span<float>((void*)f32.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    /// <summary>Deterministic pseudo-random N(0,1)-ish fill: the decoder needs real noise (a constant would make the
    /// finiteness check vacuous), and the shared-noise rule forbids trying to reproduce a torch seed.</summary>
    private static Tensor Fill(TensorShape shape, int seed)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        Random random = new Random(seed);
        float* values = (float*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            values[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return tensor;
    }
}
