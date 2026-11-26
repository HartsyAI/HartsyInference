using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The host-offloaded KV cache: its cached pass is the device cache's pass exactly, each block's keys and values are staged to the device for
/// that block's forward and released after it, and the device footprint estimate is one block's pair.</summary>
public sealed class LongCatKvOffloadTests : IDisposable
{
    private const int Cond = 2, Noisy = 3, Height = 4, Width = 6, TextRows = 5;

    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _weights;
    private readonly CpuBackend _cpu = new();
    private readonly LongCatTransformer _transformer;
    private readonly LongCatConditioning _conditioning = new(NumCondLatents: Cond);

    public LongCatKvOffloadTests()
    {
        _weights = LongCatTestData.Weights(_config, 81);
        _transformer = new LongCatTransformer(_config);
        _transformer.LoadWeights(_weights);
    }

    public void Dispose()
    {
        _transformer.Dispose();
        LongCatTestData.DisposeAll(_weights);
        _cpu.Dispose();
    }

    [Fact]
    public void AnOffloadedCachedPassIsTheDeviceCachedPassExactly()
    {
        using Tensor cond = LongCatTestData.Latent(_config, Cond, Height, Width, 82);
        using Tensor noisy = LongCatTestData.Latent(_config, Noisy, Height, Width, 83);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 84);
        using LongCatKvCache device = _transformer.BuildKvCache(_cpu, cond, _conditioning);
        using LongCatKvCache offloaded = _transformer.BuildKvCache(_cpu, cond, _conditioning, offload: true);
        Assert.False(device.Offloaded);
        Assert.True(offloaded.Offloaded);
        Assert.Equal(device.Bytes, offloaded.Bytes);

        using Tensor expected = _transformer.ForwardCached(_cpu, noisy, text, [300f], _conditioning, device);
        using Tensor actual = _transformer.ForwardCached(_cpu, noisy, text, [300f], _conditioning, offloaded);
        float[] a = expected.AsReadOnlySpan<float>().ToArray(), b = actual.AsReadOnlySpan<float>().ToArray();
        Assert.True(a.Any(x => Math.Abs(x) > 0.01f), "The output is all zeros; nothing was compared.");
        Assert.Equal(a, b);
    }

    [Fact]
    public void EachBlockIsStagedOnceAndReleasedAfterIt()
    {
        using Tensor cond = LongCatTestData.Latent(_config, Cond, Height, Width, 85);
        (IBackend backend, StagingRecorder recorder) = Wrap(_cpu);
        using LongCatKvCache offloaded = _transformer.BuildKvCache(backend, cond, _conditioning, offload: true);
        recorder.Log.Clear();

        for (int layer = 0; layer < _config.Depth; layer++)
        {
            using IDisposable? staged = offloaded.Stage(layer);
            Assert.NotNull(staged);
        }

        Assert.Equal(_config.Depth, recorder.Preloads.Count);
        Assert.Equal(_config.Depth, recorder.Frees.Count);
        for (int layer = 0; layer < _config.Depth; layer++)
        {
            Assert.Equal(Pair(offloaded, layer), recorder.Preloads[layer]);
            Assert.Equal(Pair(offloaded, layer), recorder.Frees[layer]);
        }
        Assert.Equal(Enumerable.Range(0, _config.Depth).SelectMany(i => new[] { $"preload{i}", $"free{i}" }).ToList(), recorder.Log);
    }

    [Fact]
    public void ADeviceCacheIsNeverStaged()
    {
        using Tensor cond = LongCatTestData.Latent(_config, Cond, Height, Width, 88);
        (IBackend backend, StagingRecorder recorder) = Wrap(_cpu);
        using LongCatKvCache device = _transformer.BuildKvCache(backend, cond, _conditioning);
        recorder.Log.Clear();

        Assert.Null(device.Stage(0));
        Assert.Empty(recorder.Preloads);
        Assert.Empty(recorder.Frees);
    }

    [Fact]
    public void TheOffloadedEstimateIsOneBlockOfTheDeviceEstimate()
    {
        long device = LongCatKvCache.EstimateBytes(_config, Cond, Height, Width, offloaded: false);
        long offloaded = LongCatKvCache.EstimateBytes(_config, Cond, Height, Width, offloaded: true);
        Assert.Equal(device / _config.Depth, offloaded);
    }

    /// <summary>The proxy as the backend a forward takes, and the same object as its recorder.</summary>
    private static (IBackend Backend, StagingRecorder Recorder) Wrap(IBackend inner)
    {
        IBackend backend = DispatchProxy.Create<IBackend, StagingRecorder>();
        StagingRecorder recorder = (StagingRecorder)(object)backend;
        recorder.Inner = inner;
        return (backend, recorder);
    }

    private static List<Tensor> Pair(LongCatKvCache cache, int layer) => [cache.Keys(layer), cache.Values(layer)];

    /// <summary>Forwards to the CPU backend and records the preloads and frees of the staging, each by the tensors it names, in call order.</summary>
    public class StagingRecorder : DispatchProxy
    {
        public IBackend? Inner { get; set; }
        public List<List<Tensor>> Preloads { get; } = [];
        public List<List<Tensor>> Frees { get; } = [];
        public List<string> Log { get; } = [];

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            if (targetMethod.Name is nameof(IBackend.PreloadWeights) or nameof(IBackend.FreeWeights))
            {
                List<Tensor> tensors = [.. (IEnumerable<Tensor>)args![0]!];
                if (targetMethod.Name == nameof(IBackend.PreloadWeights))
                {
                    Preloads.Add(tensors);
                    Log.Add($"preload{Preloads.Count - 1}");
                }
                else
                {
                    Frees.Add(tensors);
                    Log.Add($"free{Frees.Count - 1}");
                }
            }
            try
            {
                return targetMethod.Invoke(Inner, args);
            }
            catch (TargetInvocationException ex) when (ex.InnerException is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Throw(ex.InnerException);
                throw;
            }
        }
    }
}
