using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The FLUX.2 DiT run block-streamed and resident, on the tiny fixture. The stream is the shared
/// <see cref="BlockStreamingController"/>, with a stub upload cache: the CPU backend has no device to copy to, so an upload is a no-op.
/// What is checked is the orchestration (every block brought in before its forward, the outputs identical to the resident run) and that
/// the block split covers every weight exactly once. The real transfer is CUDA-only and is measured on the real-weight run.</summary>
public sealed class Flux2StreamingTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "flux2_edit_tiny");
    private const int HPacked = 2, WPacked = 3;

    [Fact]
    public void AStreamedVelocityMatchesTheResidentOneAndEveryBlockIsBroughtIn()
    {
        // The parity fixture's F32 DiT, the same profile the reference-forward parity test runs under.
        using IDisposable f32 = KnobProfile.Create("ovb-parity").With(EngineKnobs.DitF16, false).Push();
        using SafeTensorsLoader weights = Weights();
        using Flux2Transformer dit = Transformer(weights);
        using SafeTensorsLoader reference = Reference();
        using CpuBackend backend = new();
        using Tensor noise = reference.GetTensor("noise");
        using Tensor packed = Flux2LatentOps.Pack(noise);
        using Tensor cond = reference.GetTensor("cond");

        float[] resident = Velocity(backend, dit, packed, cond);

        IStreamingBlock[] blocks = new IStreamingBlock[dit.BlockCount];
        for (int b = 0; b < blocks.Length; b++)
            blocks[b] = dit.GetBlock(b);
        CountingCache cache = new();
        using BlockStreamingController streamer = new(cache, blocks, prefetchAhead: 1, retainBehind: 0, backend: null);
        float[] streamed;
        try
        {
            dit.BeforeBlockForward = streamer.BeforeBlockForward;
            streamer.Prime();
            streamed = Velocity(backend, dit, packed, cond);
        }
        finally
        {
            dit.BeforeBlockForward = null;
            streamer.EvictAll();
        }

        Assert.Equal(resident, streamed);
        for (int b = 0; b < blocks.Length; b++)
            Assert.Contains(blocks[b].EnumerateWeights(), cache.Uploaded.Contains);
        Assert.True(cache.Uploads >= blocks.Length, $"expected at least one upload per block, got {cache.Uploads}");
    }

    [Fact]
    public void TheBlockSplitCoversEveryWeightExactlyOnce()
    {
        using SafeTensorsLoader weights = Weights();
        using Flux2Transformer dit = Transformer(weights);
        Flux2Config config = IrisZTestData.Config(modalities: 0);

        Assert.Equal(config.Depth + config.DepthSingleBlocks, dit.BlockCount);

        List<Tensor> all = dit.EnumerateWeights().ToList();
        List<Tensor> shared = dit.EnumerateSharedWeights().ToList();
        List<Tensor> blockWeights = Enumerable.Range(0, dit.BlockCount)
            .SelectMany(b => dit.GetBlock(b).EnumerateWeights()).ToList();

        Assert.Equal(all.Count, shared.Count + blockWeights.Count);
        HashSet<Tensor> distinct = new(all, ReferenceEqualityComparer.Instance);
        Assert.Equal(all.Count, distinct.Count);
        Assert.All(shared.Concat(blockWeights), w => Assert.Contains(w, distinct));
    }

    private static float[] Velocity(CpuBackend backend, Flux2Transformer dit, Tensor packed, Tensor cond)
    {
        using Tensor velocity = dit.ForwardWithReferences(backend, packed, cond, 0.6f, HPacked, WPacked, [0f]);
        return velocity.AsSpan<float>().ToArray();
    }

    private static Flux2Transformer Transformer(SafeTensorsLoader weights)
    {
        Dictionary<string, Tensor> converted = new Flux2CheckpointConverter(IrisZTestData.Hidden, IrisZTestData.Mlp)
            .ConvertTransformer(weights.GetAllTensors());
        Flux2Transformer transformer = new(IrisZTestData.Config(modalities: 0));
        transformer.LoadWeights(converted);
        return transformer;
    }

    private static SafeTensorsLoader Weights()
    {
        SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "weights.safetensors"));
        return loader;
    }

    private static SafeTensorsLoader Reference()
    {
        SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "reference.safetensors"));
        return loader;
    }

    /// <summary>A stand-in upload cache: every upload is an immediate no-op, and the uploads are counted for the test.</summary>
    private sealed class CountingCache : IStreamingWeightCache
    {
        public int Uploads { get; private set; }

        public HashSet<Tensor> Uploaded { get; } = new(ReferenceEqualityComparer.Instance);

        public StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights)
        {
            Uploads++;
            foreach (Tensor weight in weights)
                Uploaded.Add(weight);
            return new StreamingUploadToken(1, this);
        }

        public void AwaitWeights(StreamingUploadToken token)
        {
        }

        public void EvictAsync(IEnumerable<Tensor> weights)
        {
        }

        public long QueryAvailableWeightCacheBytes(long activationReserve) => long.MaxValue / 2;

        public void DrainAndReleasePool()
        {
        }
    }
}
