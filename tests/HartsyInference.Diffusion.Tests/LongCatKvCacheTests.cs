using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat KV cache: the cached denoising pass against the forward over the whole clip, the queries it runs, and what the cache pins,
/// frees and weighs.</summary>
[Collection("AttentionEvidence")]
public sealed class LongCatKvCacheTests : IDisposable
{
    private const int Cond = 2, Noisy = 3, Height = 4, Width = 6, Hw = Height / 2 * (Width / 2), TextRows = 5;

    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _weights;
    private readonly CpuBackend _cpu = new();
    private readonly IBackend _backend;
    private readonly Recorder _recorder;
    private readonly LongCatTransformer _transformer;
    private readonly LongCatConditioning _conditioning = new(NumCondLatents: Cond);

    public LongCatKvCacheTests()
    {
        _weights = LongCatTestData.Weights(_config, 61);
        _backend = DispatchProxy.Create<IBackend, Recorder>();
        _recorder = (Recorder)_backend;
        _recorder.Inner = _cpu;
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
    public void TheCachedPass_EqualsTheNoisyFramesOfTheWholeForward()
    {
        using Tensor clip = LongCatTestData.Latent(_config, Cond + Noisy, Height, Width, 62);
        using Tensor cond = Frames(clip, 0, Cond);
        using Tensor noisy = Frames(clip, Cond, Noisy);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 63);

        using Tensor whole = _transformer.Forward(_cpu, clip, text, _conditioning.FrameTimesteps(300f, Cond + Noisy), in _conditioning);
        using LongCatKvCache cache = _transformer.BuildKvCache(_cpu, cond, _conditioning);
        using Tensor cached = _transformer.ForwardCached(_cpu, noisy, text, [300f], _conditioning, cache);

        using Tensor expected = Frames(whole, Cond, Noisy);
        ReadOnlySpan<float> a = expected.AsReadOnlySpan<float>(), b = cached.AsReadOnlySpan<float>();
        Assert.Equal(a.Length, b.Length);
        float worst = 0f, largest = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
            largest = Math.Max(largest, Math.Abs(a[i]));
        }
        Assert.True(largest > 0.01f, "The output is all zeros; nothing was compared.");
        Assert.True(worst <= 1e-5f * Math.Max(1f, largest), $"The cached pass is {worst:E2} from the whole forward (largest value {largest:E2}).");
    }

    [Fact]
    public void TheCachedPass_RunsQueriesForTheNoisyRowsOnly()
    {
        using Tensor cond = LongCatTestData.Latent(_config, Cond, Height, Width, 64);
        using Tensor noisy = LongCatTestData.Latent(_config, Noisy, Height, Width, 65);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 66);
        using LongCatKvCache cache = _transformer.BuildKvCache(_cpu, cond, _conditioning);
        // The session is process-wide; the component names this thread's calls.
        using AttentionEvidenceSession session = AttentionEvidence.Begin();
        using (AttentionEvidence.Context(nameof(LongCatKvCacheTests)))
        {
            using Tensor cached = _transformer.ForwardCached(_cpu, noisy, text, [300f], _conditioning, cache);
        }

        (int Queries, int Keys)[] self = [.. session.Snapshot()
            .Where(static e => e.Kind == AttentionEvidenceKind.Execution && e.Component == nameof(LongCatKvCacheTests) && e.KeyCapacity != TextRows)
            .Select(static e => (e.QueryTokens, e.KeyCapacity))];
        Assert.Equal(_config.Depth, self.Length);
        Assert.All(self, static a => Assert.Equal((Noisy * Hw, (Cond + Noisy) * Hw), a));
    }

    [Fact]
    public void TheCache_PinsEveryKeyAndValue_UntilItIsDisposed()
    {
        using Tensor cond = LongCatTestData.Latent(_config, Cond, Height, Width, 67);
        LongCatKvCache cache = _transformer.BuildKvCache(_backend, cond, _conditioning);

        Assert.Equal(2 * _config.Depth, _recorder.Pinned.Count);
        for (int layer = 0; layer < _config.Depth; layer++)
        {
            Assert.Equal(new TensorShape(1, _config.NumHeads, Cond * Hw, _config.HeadDim), cache.Keys(layer).Shape);
            Assert.Contains(cache.Keys(layer), _recorder.Pinned);
            Assert.Contains(cache.Values(layer), _recorder.Pinned);
        }
        Tensor keys = cache.Keys(0);

        cache.Dispose();
        cache.Dispose();
        Assert.Empty(_recorder.Pinned);
        Assert.Throws<ObjectDisposedException>(() => keys.AsReadOnlySpan<float>().Length);
    }

    [Fact]
    public void ABuildThatFailsInALaterBlock_FreesWhatTheEarlierOnesStored()
    {
        using Tensor cond = LongCatTestData.Latent(_config, Cond, Height, Width, 68);
        _transformer.BeforeBlockForward = block =>
        {
            if (block == 1)
            {
                Assert.Equal(2, _recorder.Pinned.Count);
                throw new InvalidOperationException("block 1 failed");
            }
        };

        Assert.Equal("block 1 failed",
            Assert.Throws<InvalidOperationException>(() => _transformer.BuildKvCache(_backend, cond, _conditioning)).Message);

        Assert.Empty(_recorder.Pinned);
    }

    [Fact]
    public void EstimateBytes_IsWhatTheBuiltCacheTakes()
    {
        using Tensor cond = LongCatTestData.Latent(_config, Cond, Height, Width, 69);
        using LongCatKvCache cache = _transformer.BuildKvCache(_cpu, cond, _conditioning);

        Assert.Equal(cache.Bytes, LongCatKvCache.EstimateBytes(_config, Cond, Height, Width));
        Assert.Equal(Cond * Hw, cache.Tokens);
        Assert.Equal(_config.Depth, cache.Layers);
    }

    [Fact]
    public void EstimateBytes_RefusesWhatABuildRefuses()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => LongCatKvCache.EstimateBytes(_config, 0, Height, Width));
        Assert.Throws<ArgumentOutOfRangeException>(() => LongCatKvCache.EstimateBytes(_config, -1, Height, Width));
        Assert.Throws<ArgumentException>(() => LongCatKvCache.EstimateBytes(_config, Cond, Height + 1, Width));
    }

    [Fact]
    public void ACacheOfAnotherGridOrBackend_IsRefused()
    {
        using Tensor cond = LongCatTestData.Latent(_config, Cond, Height, Width, 70);
        using Tensor sameRows = LongCatTestData.Latent(_config, Noisy, Width, Height, 71);
        using Tensor noisy = LongCatTestData.Latent(_config, Noisy, Height, Width, 72);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 73);
        using LongCatKvCache cache = _transformer.BuildKvCache(_cpu, cond, _conditioning);
        using CpuBackend other = new();

        Assert.Contains("2×3 tokens", Assert.Throws<ArgumentException>(
            () => _transformer.ForwardCached(_cpu, sameRows, text, [300f], _conditioning, cache)).Message);
        Assert.Contains("another backend", Assert.Throws<ArgumentException>(
            () => _transformer.ForwardCached(other, noisy, text, [300f], _conditioning, cache)).Message);
    }

    private static Tensor Frames(Tensor clip, int from, int count)
    {
        int channels = (int)clip.Shape[1], frames = (int)clip.Shape[2], height = (int)clip.Shape[3], width = (int)clip.Shape[4], hw = height * width;
        Tensor part = new(new TensorShape([1, channels, count, height, width]), DType.F32);
        for (int c = 0; c < channels; c++)
            clip.AsReadOnlySpan<float>().Slice((c * frames + from) * hw, count * hw).CopyTo(part.AsSpan<float>().Slice(c * count * hw, count * hw));
        return part;
    }

    /// <summary>Forwards every call to the CPU backend and records the pins. It cannot forward a call that takes a span, such as Concat.</summary>
    public class Recorder : DispatchProxy
    {
        public IBackend? Inner { get; set; }

        public HashSet<Tensor> Pinned { get; } = new(ReferenceEqualityComparer.Instance);

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            switch (targetMethod.Name)
            {
                case nameof(IBackend.PinActivation):
                    Assert.True(Pinned.Add((Tensor)args![0]!));
                    break;
                case nameof(IBackend.UnpinActivation):
                    Assert.True(Pinned.Remove((Tensor)args![0]!));
                    break;
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
