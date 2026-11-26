using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.FastFrames;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The KV refiner's layer repair on CUDA must not read device results back to the host: the per-head renorm runs as the
/// guided Euler step (#1209), so the repair's device-to-host sync count is zero. This is a measurement with a hard assertion; it runs
/// on a CUDA device only, and the count is printed so a failure says how many syncs happened.</summary>
public sealed class FastFramesRefinerGpuSyncTests
{
    private const int Tokens = 24;

    private static readonly FastFramesKvRefinerConfig Tiny = new(Dim: 64, Heads: 4, Layers: 2, AdapterDim: 32,
        StemBlocks: 1, StemHeads: 4, StemMlpMultiplier: 2, HeadMlpMultiplier: 2, FrequencyDim: 32);

    [CudaDeviceFact]
    public void RepairLayerOnCudaReadsNothingBackToTheHost()
    {
        Dictionary<string, Tensor> weights = RandomWeights(seed: 31);
        try
        {
            using IBackend cuda = HartsyInference.Engine.BackendFactory.CreateCuda(0);
            using FastFramesKvRefiner refiner = new(Tiny);
            refiner.LoadWeights(weights);
            using FastFramesKvRefiner.Conditioning conditioning = refiner.Prepare(cuda, 833f);

            using Tensor noisyK = RandomTensor([Tokens, Tiny.Dim], seed: 32);
            using Tensor noisyV = RandomTensor([Tokens, Tiny.Dim], seed: 33);
            using Tensor phi = RandomTensor([Tokens, Tiny.AdapterDim], seed: 34);

            cuda.ResetD2hSyncCount();
            (Tensor refinedK, Tensor refinedV) = refiner.RepairLayer(cuda, 0, noisyK, noisyV, phi, conditioning);
            long syncs = cuda.GetD2hSyncCount();
            refinedK.Dispose();
            refinedV.Dispose();

            Assert.True(syncs == 0, $"RepairLayer on CUDA performed {syncs} device-to-host syncs; expected 0.");
        }
        finally
        {
            foreach (Tensor t in weights.Values)
                t.Dispose();
        }
    }

    private static Tensor RandomTensor(long[] shape, int seed)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        Random rng = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        return tensor;
    }

    private static Dictionary<string, Tensor> RandomWeights(int seed)
    {
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        foreach ((string name, long[] shape) in FastFramesKvRefiner.ExpectedShapes(Tiny))
            weights[name] = RandomTensor(shape, seed + weights.Count);
        return weights;
    }
}
