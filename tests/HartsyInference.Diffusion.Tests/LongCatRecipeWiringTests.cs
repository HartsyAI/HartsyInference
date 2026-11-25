using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The recipe's streamed denoise, through the shared helper the LongCat pipelines call: every finished step ends the stream, the stream's
/// hook is cleared once the loop is over, and the DiT's weights are freed on success, on a cancel and on an exception. A fake backend records the
/// frees and supplies a streaming cache, so the streamed path runs on the CPU backend.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatRecipeWiringTests
{
    private const int Channels = 4;

    [Fact]
    public void EveryFinishedStepEndsTheStreamAndTheDiTIsFreedAfterTheClip()
    {
        using Harness h = new();

        float[] latents = h.Run(onStep =>
        {
            for (int i = 1; i <= 3; i++)
                onStep(new GenerationProgress(i, 3, 0));
            return [1f, 2f];
        });

        Assert.Equal([1f, 2f], latents);
        Assert.True(h.Scope!.Streaming, "the wiring test must exercise the streamed path");
        Assert.Equal(3, h.Scope.StepsEnded);
        Assert.Null(h.Dit.BeforeBlockForward);
        h.AssertEveryDiTWeightFreed();
    }

    [Fact]
    public void ACancelMidDenoiseStillFreesTheDiTAndClearsTheStreamHook()
    {
        using Harness h = new();

        Assert.Throws<OperationCanceledException>(() => h.Run(onStep =>
        {
            onStep(new GenerationProgress(1, 3, 0));
            throw new OperationCanceledException();
        }));

        Assert.Equal(1, h.Scope!.StepsEnded);
        Assert.Null(h.Dit.BeforeBlockForward);
        h.AssertEveryDiTWeightFreed();
    }

    [Fact]
    public void AnExceptionMidDenoiseStillFreesTheDiT()
    {
        using Harness h = new();

        Assert.Throws<InvalidOperationException>(() => h.Run(_ => throw new InvalidOperationException("step failed")));

        Assert.Equal(0, h.Scope!.StepsEnded);
        Assert.Null(h.Dit.BeforeBlockForward);
        h.AssertEveryDiTWeightFreed();
    }

    /// <summary>A transformer with its tiny random weights, a recording backend over the CPU, and the recipe's streamed denoise.</summary>
    private sealed class Harness : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = LongCatTestData.Weights(LongCatTestData.Tiny, seed: 41);
        private readonly CpuBackend _cpu = new();
        private readonly CountingCache _cache = new();
        private readonly IBackend _backend;
        private readonly RecordingBackend _recorder;

        public Harness()
        {
            Dit = new LongCatTransformer(LongCatTestData.Tiny);
            Dit.LoadWeights(_weights);
            _backend = RecordingBackend.Create(_cpu, _cache, out _recorder);
        }

        public LongCatTransformer Dit { get; }

        /// <summary>The scope the recipe opened in the last run.</summary>
        public BlockStreamingScope? Scope { get; private set; }

        /// <summary>Runs the helper exactly as the recipe pipelines do: the scope is opened through the recipe's own factory.</summary>
        public float[] Run(Func<Action<GenerationProgress>, float[]> denoise)
        {
            LongCatParts parts = new(Dit, Channels, LongCatTestData.Tiny.CaptionChannels, HasAudio: false, Decoder: null!, Encoder: null!,
                Umt5: null!, Tokenizer: null!);
            return LongCatRecipeSupport.RunDenoiseStreamed(_backend, Dit,
                () => Scope = LongCatRecipeSupport.OpenDenoiseStream(_backend, parts, "longcat-test"),
                denoise, progress: null);
        }

        /// <summary>Every weight the DiT owns was freed through the backend.</summary>
        public void AssertEveryDiTWeightFreed()
        {
            foreach (Tensor weight in Dit.EnumerateWeights())
                Assert.Contains(weight, _recorder.Freed);
        }

        public void Dispose()
        {
            Dit.Dispose();
            LongCatTestData.DisposeAll(_weights);
            _cpu.Dispose();
        }
    }

    /// <summary>A streaming cache that reports no room, so every block is streamed; uploads and evictions are no-ops on the CPU.</summary>
    private sealed class CountingCache : IStreamingWeightCache
    {
        public StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights) => new(1, this);

        public void AwaitWeights(StreamingUploadToken token)
        {
        }

        public void EvictAsync(IEnumerable<Tensor> weights)
        {
        }

        public long QueryAvailableWeightCacheBytes(long activationReserve) => 0;

        public void DrainAndReleasePool()
        {
        }
    }

    /// <summary>Forwards every backend call to the CPU backend, except the streaming cache (supplied) and the frees (recorded, then forwarded).</summary>
    public class RecordingBackend : DispatchProxy
    {
        private IBackend? _inner;
        private IStreamingWeightCache? _cache;

        /// <summary>Every tensor passed to <c>FreeWeights</c>.</summary>
        public HashSet<Tensor> Freed { get; } = new(ReferenceEqualityComparer.Instance);

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
            if (targetMethod.Name == "get_StreamingCache")
                return _cache;
            if (targetMethod.Name == "FreeWeights" && args is { Length: > 0 } && args[0] is IEnumerable<Tensor> weights)
            {
                foreach (Tensor weight in weights)
                    Freed.Add(weight);
            }
            return targetMethod.Invoke(_inner, args);
        }
    }
}
