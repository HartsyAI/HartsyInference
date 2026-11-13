using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Tests.Services;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The MiniMax-H3 image VAE decodes its tiles block-major (each block's weights resident once for every tile)
/// and can be switched back to the tile-major order with <c>vae.blockMajorTiles=false</c>. The two orders run the same
/// operations on the same tiles, so their pixels must be identical. A component check on the real VAE, not a generation.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MiniMaxH3VaeTileOrderGpuTests
{
    private const string ModelsDirVariable = "HARTSYINFERENCE_MODELS_DIR";
    private const string VaeRelative = "VAE/MiniMaxH3/minimax_h3_image_vae_fp16.safetensors";

    [MiniMaxH3ImageGpuFact]
    public void BlockMajorTilesDecodeToTheSamePixelsAsTileMajor()
    {
        string models = Environment.GetEnvironmentVariable(ModelsDirVariable)!;
        using Core.Backends.IBackend backend = new HartsyInference.Cuda.CudaBackend(0, Ernie5ModelGpuParityTests.PtxDirectory());
        List<IDisposable> loaders = [];
        try
        {
            MiniMaxH3VideoVaeDecoder decoder = LoadTiledVae(Path.Combine(models, VaeRelative), loaders);
            // A 512 px latent is 3 x 3 tiles of 256 px with 64 px overlap, so both the row and the column blends run.
            using Tensor latent = new(new TensorShape([1L, 24L, 1L, 32L, 32L]), DType.F32);
            Random random = new(42);
            Span<float> values = latent.AsSpan<float>();
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }

            KnobStore.Set(EngineKnobs.VaeBlockMajorTiles, false);
            float[] tileMajor;
            using (Tensor reference = decoder.Decode(backend, latent)) tileMajor = reference.AsSpan<float>().ToArray();
            KnobStore.Set(EngineKnobs.VaeBlockMajorTiles, true);
            float[] blockMajor;
            using (Tensor candidate = decoder.Decode(backend, latent)) blockMajor = candidate.AsSpan<float>().ToArray();

            Assert.Equal(tileMajor.Length, blockMajor.Length);
            int mismatches = 0;
            for (int i = 0; i < tileMajor.Length; i++)
            {
                if (tileMajor[i] != blockMajor[i]) mismatches++;
            }
            Assert.True(mismatches == 0, $"block-major tiles differ from tile-major at {mismatches} of {tileMajor.Length} values");
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.VaeBlockMajorTiles);
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
        }
    }

    /// <summary>The live image decoder: the detected config as-is, which keeps tiling on.</summary>
    private static MiniMaxH3VideoVaeDecoder LoadTiledVae(string file, List<IDisposable> loaders)
    {
        CheckpointSource source = CheckpointSource.Open(file);
        loaders.Add(source);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(source.Weights, StringComparer.Ordinal);
        MiniMaxH3VideoVaeConfig config = MiniMaxH3VideoVaeConfig.Detect(weights);
        MiniMaxH3VideoVaeDecoder decoder = new MiniMaxH3VideoVaeDecoder(config);
        decoder.LoadWeights(weights);
        return decoder;
    }
}
