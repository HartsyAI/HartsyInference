using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>The default emotion vector on <see cref="IndexTts2Reference"/> costs a w2v-bert pass, so it is built on first
/// need, once, and owned by the reference: a reference only used with its own emotion clip never pays for it.</summary>
public sealed class IndexTts2ReferenceLazyEmoVecTests
{
    [Fact]
    public void DefaultEmoVec_IsBuiltOnFirstNeedOnce_AndDisposedWithTheReference()
    {
        int calls = 0;
        Tensor? produced = null;
        using CpuBackend backend = new();
        IndexTts2Reference reference = CreateReference(b =>
        {
            Assert.Same(backend, b);
            calls++;
            produced = Filled(1, 1280, 0.5f);
            return produced;
        });

        Assert.Equal(0, calls);
        Tensor first = reference.GetEmoVec(backend);
        Tensor second = reference.GetEmoVec(backend);
        Assert.Equal(1, calls);
        Assert.Same(produced, first);
        Assert.Same(first, second);
        Assert.Equal(0.5f, first.AsSpan<float>()[1279]);

        reference.Dispose();
        Assert.Throws<ObjectDisposedException>(() => reference.GetEmoVec(backend));
        Assert.Throws<ObjectDisposedException>(() => { _ = first.AsSpan<float>().Length; });
        Assert.Equal(1, calls);
    }

    [Fact]
    public void DefaultEmoVec_NeverNeeded_IsNeverBuilt()
    {
        int calls = 0;
        IndexTts2Reference reference = CreateReference(_ =>
        {
            calls++;
            return Filled(1, 1280, 0f);
        });
        reference.Dispose();
        Assert.Equal(0, calls);
    }

    [Fact]
    public void ConcurrentFirstCalls_BuildOnce()
    {
        int calls = 0;
        using CpuBackend backend = new();
        using IndexTts2Reference reference = CreateReference(_ =>
        {
            Interlocked.Increment(ref calls);
            Thread.Yield();
            return Filled(1, 1280, 1f);
        });
        Tensor[] seen = new Tensor[8];
        Parallel.For(0, seen.Length, new ParallelOptions { MaxDegreeOfParallelism = seen.Length }, i => seen[i] = reference.GetEmoVec(backend));
        Assert.Equal(1, calls);
        Assert.All(seen, t => Assert.Same(seen[0], t));
    }

    private static IndexTts2Reference CreateReference(Func<IBackend, Tensor> emoVecFactory) =>
        new(owner: null!, Zeros(1, 4, 1024), 4, Zeros(1, 80, 4), 4, Zeros(1, 192), Zeros(1, 4, 1024), Zeros(1, 32, 1280),
            Zeros(1, 1280), emoVecFactory, new float[16]);

    private static Tensor Zeros(params long[] shape) => Filled(shape, 0f);

    private static Tensor Filled(long rows, long columns, float value) => Filled([rows, columns], value);

    private static Tensor Filled(long[] shape, float value)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }
}
