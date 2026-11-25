using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat DiT run block-streamed and resident, on the tiny random checkpoint. The stream is driven through the same
/// <see cref="BlockStreamingController"/> the other families use, with a stub upload cache: the CPU backend has no device to copy to, so the
/// upload is a no-op. What the test checks is the orchestration: every block is brought in before its forward, in order, and the outputs are
/// identical to the resident run. The real transfer is CUDA-only and is measured at the real-weight run (B8).</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatStreamingTests
{
    private const int Frames = 3, Height = 4, Width = 4, Text = 5;

    [Fact]
    public void AStreamedDenoiseMatchesTheResidentOne()
    {
        LongCatConfig config = LongCatTestData.Tiny;
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, seed: 23);
        try
        {
            using LongCatTransformer dit = new(config);
            dit.LoadWeights(weights);
            using CpuBackend backend = new();
            LongCatVideoPipeline pipeline = new(backend, dit);
            LongCatRun run = TinyRun(config);

            float[] resident = pipeline.Denoise(run with { RecordTrace = false }).Latents;

            IStreamingBlock[] blocks = new IStreamingBlock[dit.BlockCount];
            for (int b = 0; b < blocks.Length; b++)
                blocks[b] = dit.GetBlock(b);
            CountingCache cache = new();
            using BlockStreamingController streamer = new(cache, blocks, prefetchAhead: 1, retainBehind: 0, backend: null);
            dit.BeforeBlockForward = streamer.BeforeBlockForward;
            float[] streamed;
            try
            {
                streamer.Prime();
                streamed = pipeline.Denoise(run with { RecordTrace = false }).Latents;
            }
            finally
            {
                dit.BeforeBlockForward = null;
                streamer.EvictAll();
            }

            Assert.Equal(resident, streamed);
            // Each denoise step visits every block once, so every block is uploaded at least once per step.
            Assert.True(cache.Uploads >= blocks.Length * run.Steps, $"{cache.Uploads} uploads for {blocks.Length} blocks over {run.Steps} steps.");
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private static LongCatRun TinyRun(LongCatConfig config)
    {
        float[] noise = Wave(config.InChannels * Frames * Height * Width, 0.5f);
        return new LongCatRun(config.InChannels, Frames, Height, Width, noise, Prompt(config, 1), Prompt(config, 2), Steps: 3,
            Distill: false, Shift: 3f, Guidance: 4f);
    }

    private static LongCatPrompt Prompt(LongCatConfig config, int seed) => new(Wave(Text * config.CaptionChannels, 0.1f * seed), Text, Text);

    private static float[] Wave(int count, float phase)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = 0.5f * MathF.Sin(0.37f * i + phase);
        return values;
    }

    /// <summary>A stand-in upload cache: every upload is an immediate no-op, and the count is kept for the test.</summary>
    private sealed class CountingCache : IStreamingWeightCache
    {
        public int Uploads { get; private set; }

        public StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights)
        {
            foreach (Tensor _ in weights)
                Uploads++;
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
