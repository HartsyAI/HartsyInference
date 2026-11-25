using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A pinned prefix released for a re-size restores the shared weights before the planner reads free memory: the release frees each block,
/// and a freed block also frees the LoRA adapter operands it expands to, which live with the shared weights.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class BlockStreamingReleaseTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "longcat-release-" + Guid.NewGuid().ToString("N"));
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _dense;

    public BlockStreamingReleaseTests()
    {
        Directory.CreateDirectory(_directory);
        _dense = LongCatTestData.Weights(_config, 67);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_dense);
        Directory.Delete(_directory, recursive: true);
    }

    [Fact]
    public void AReleasedPinRestoresTheSharedAdapterOperandsBeforeTheFreeMemoryRead()
    {
        using CpuBackend cpu = new();
        using LongCatTransformer dit = GatedTransformer(cpu);
        IBackend backend = RecordingBackend.Create(cpu, new NoStreamingCache(), out RecordingBackend recorder);
        IEnumerable<Tensor> operands = ((LongCatBlock)dit.GetBlock(0)).EnumerateAdapterOperands().ToList();
        Assert.NotEmpty(operands);

        // The first generation sizes the prefix and keeps it resident.
        ResidentPrefixPin pin = new();
        using (BlockStreamingScope.Open(Options(backend, dit, pin, tokenLoad: 100)))
            pin.Resident = true;
        recorder.Calls.Clear();

        // A longer generation releases the pinned prefix to re-size it.
        using (BlockStreamingScope.Open(Options(backend, dit, pin, tokenLoad: 200)))
        {
        }

        int freed = recorder.Calls.FindIndex(c => c.Kind == "FreeWeights" && c.Tensors.Contains(dit.GetBlock(0).EnumerateWeights().First()));
        Assert.True(freed >= 0, "the release must free block 0");
        int restored = recorder.Calls.FindIndex(freed + 1, c => c.Kind == "PreloadWeights" && operands.All(o => c.Tensors.Contains(o)));
        Assert.True(restored > freed, "the shared adapter operands must be preloaded again after the block free");
    }

    private BlockStreamingOptions Options(IBackend backend, LongCatTransformer dit, ResidentPrefixPin pin, long tokenLoad) => new()
    {
        Backend = backend, Denoiser = dit, ModelName = "release-test", HeadroomBytes = 0, TokenLoad = tokenLoad, Pin = pin,
    };

    private LongCatTransformer GatedTransformer(CpuBackend cpu)
    {
        using LongCatLoraFixture fixture = new(Path.Combine(_directory, "a.safetensors"), _config, 9);
        using LoraFile file = LoraFile.Load(fixture.Path);
        using LoraStack stack = new();
        stack.AddGated(file, new AdjunctGate());
        Dictionary<string, Tensor> patched = new(_dense, StringComparer.Ordinal);
        stack.ApplyTo(patched, LoraTarget.Transformer, cpu);
        LongCatTransformer dit = new(_config);
        dit.LoadWeights(patched);
        return dit;
    }

    /// <summary>Forwards every call to the CPU backend and records the tensors each preload and free was given. The free memory is reported as ample,
    /// so the sizing keeps every block resident.</summary>
    public class RecordingBackend : DispatchProxy
    {
        private IBackend? _inner;
        private IStreamingWeightCache? _cache;

        public List<(string Kind, List<Tensor> Tensors)> Calls { get; } = new();

        public static IBackend Create(IBackend inner, IStreamingWeightCache cache, out RecordingBackend recorder)
        {
            IBackend proxy = DispatchProxy.Create<IBackend, RecordingBackend>();
            recorder = (RecordingBackend)(object)proxy;
            recorder._inner = inner;
            recorder._cache = cache;
            return proxy;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            switch (targetMethod.Name)
            {
                case "get_StreamingCache":
                    return _cache;
                case "FreeMemoryBytes":
                    return 1L << 40;
                case "PreloadWeights":
                case "FreeWeights":
                    List<Tensor> tensors = ((IEnumerable<Tensor>)args![0]!).ToList();
                    Calls.Add((targetMethod.Name, tensors));
                    return targetMethod.Invoke(_inner, [tensors, .. args.Skip(1)]);
                default:
                    return targetMethod.Invoke(_inner, args);
            }
        }
    }

    /// <summary>A streaming cache with no room to upload: the release path only needs the cache to exist.</summary>
    private sealed class NoStreamingCache : IStreamingWeightCache
    {
        public StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights) => new(1, this);

        public void AwaitWeights(StreamingUploadToken token)
        {
        }

        public void EvictAsync(IEnumerable<Tensor> weights)
        {
        }

        public long QueryAvailableWeightCacheBytes(long activationReserve) => long.MaxValue;

        public void DrainAndReleasePool()
        {
        }
    }
}
