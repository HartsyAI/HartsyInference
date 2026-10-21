using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The UNet's two-slot (backend, context) constant K/V cache: which slot a backend owns, what a hit returns, what an
/// eviction disposes, and that two threads on two backends sharing one sub-block never see each other's tensors.</summary>
public sealed unsafe class ConstantKvCacheTests
{
    private const int Channels = 16, Heads = 2, CtxDim = 12, Tokens = 6, CtxTokens = 5;

    private static Tensor Mh() => new(new TensorShape(1, Heads, CtxTokens, Channels / Heads), DType.F32);

    private static Tensor Ctx() => new(new TensorShape(1, CtxTokens, CtxDim), DType.F32);

    private static bool IsDisposed(Tensor t)
    {
        try
        {
            _ = (nint)t.DataPointer;
            return false;
        }
        catch (ObjectDisposedException)
        {
            return true;
        }
    }

    [Fact]
    public void AMissStoresAndTheSameBackendAndContextHitsWithTheStoredTensors()
    {
        using CpuBackend backend = new();
        ConstantKvCache cache = new();
        Tensor context = Ctx(), key = Mh(), value = Mh();

        Assert.False(cache.TryGet(backend, context, out _, out _));
        Assert.True(cache.TryStore(backend, context, key, value));

        Assert.True(cache.TryGet(backend, context, out Tensor hitKey, out Tensor hitValue));
        Assert.Same(key, hitKey);
        Assert.Same(value, hitValue);
        Assert.False(IsDisposed(key));
    }

    [Fact]
    public void AHitNeedsTheSameBackendAndTheSameContextReference()
    {
        using CpuBackend backend = new();
        using CpuBackend other = new();
        ConstantKvCache cache = new();
        Tensor context = Ctx();
        cache.TryStore(backend, context, Mh(), Mh());

        Assert.False(cache.TryGet(other, context, out _, out _));
        Assert.False(cache.TryGet(backend, Ctx(), out _, out _));
    }

    [Fact]
    public void ARestoreOnTheSameBackendReplacesItsSlotAndDisposesTheOldTensors()
    {
        using CpuBackend backend = new();
        ConstantKvCache cache = new();
        Tensor first = Ctx(), second = Ctx(), oldKey = Mh(), oldValue = Mh(), newKey = Mh(), newValue = Mh();
        cache.TryStore(backend, first, oldKey, oldValue);

        cache.TryStore(backend, second, newKey, newValue);

        Assert.True(IsDisposed(oldKey));
        Assert.True(IsDisposed(oldValue));
        Assert.False(cache.TryGet(backend, first, out _, out _));
        Assert.True(cache.TryGet(backend, second, out Tensor hit, out _));
        Assert.Same(newKey, hit);
    }

    [Fact]
    public void TwoBackendsHoldTwoSlotsAndNeitherEvictsTheOther()
    {
        using CpuBackend a = new();
        using CpuBackend b = new();
        ConstantKvCache cache = new();
        Tensor ctxA = Ctx(), ctxB = Ctx(), keyA = Mh(), valueA = Mh(), keyB = Mh(), valueB = Mh();

        cache.TryStore(a, ctxA, keyA, valueA);
        cache.TryStore(b, ctxB, keyB, valueB);

        Assert.True(cache.TryGet(a, ctxA, out Tensor hitA, out _));
        Assert.True(cache.TryGet(b, ctxB, out Tensor hitB, out _));
        Assert.Same(keyA, hitA);
        Assert.Same(keyB, hitB);
        Assert.False(cache.TryGet(a, ctxB, out _, out _));
        Assert.False(cache.TryGet(b, ctxA, out _, out _));
        Assert.All((Tensor[])[keyA, valueA, keyB, valueB], t => Assert.False(IsDisposed(t)));
    }

    [Fact]
    public void AThirdBackendEvictsSlotZeroAndLeavesSlotOne()
    {
        using CpuBackend a = new();
        using CpuBackend b = new();
        using CpuBackend c = new();
        ConstantKvCache cache = new();
        Tensor ctxA = Ctx(), ctxB = Ctx(), ctxC = Ctx(), keyA = Mh(), valueA = Mh(), keyB = Mh(), valueB = Mh(), keyC = Mh(), valueC = Mh();
        cache.TryStore(a, ctxA, keyA, valueA);
        cache.TryStore(b, ctxB, keyB, valueB);

        cache.TryStore(c, ctxC, keyC, valueC);

        Assert.True(IsDisposed(keyA));
        Assert.True(IsDisposed(valueA));
        Assert.False(IsDisposed(keyB));
        Assert.False(cache.TryGet(a, ctxA, out _, out _));
        Assert.True(cache.TryGet(b, ctxB, out _, out _));
        Assert.True(cache.TryGet(c, ctxC, out Tensor hitC, out _));
        Assert.Same(keyC, hitC);
    }

    [Fact]
    public void TwoThreadsOnTwoBackendsEachKeepTheirOwnSlot()
    {
        using CpuBackend a = new();
        using CpuBackend b = new();
        ConstantKvCache cache = new();
        Tensor ctxA = Ctx(), ctxB = Ctx();
        Tensor?[] firstKey = new Tensor?[2];
        using Barrier barrier = new(2);

        void Run(int index, IBackend backend, Tensor context)
        {
            barrier.SignalAndWait();
            for (int step = 0; step < 200; step++)
            {
                if (cache.TryGet(backend, context, out Tensor key, out Tensor value))
                {
                    Assert.Same(firstKey[index], key);
                    Assert.False(IsDisposed(key));
                    Assert.False(IsDisposed(value));
                }
                else
                {
                    Assert.Null(firstKey[index]);
                    firstKey[index] = Mh();
                    cache.TryStore(backend, context, firstKey[index]!, Mh());
                }
            }
        }

        Exception?[] failures = new Exception?[2];
        Thread[] threads =
        [
            new(() => Guard(0, () => Run(0, a, ctxA))),
            new(() => Guard(1, () => Run(1, b, ctxB))),
        ];
        void Guard(int index, Action action)
        {
            try { action(); } catch (Exception ex) { failures[index] = ex; }
        }

        foreach (Thread t in threads) t.Start();
        foreach (Thread t in threads) t.Join();

        Assert.Null(failures[0]);
        Assert.Null(failures[1]);
        Assert.NotSame(firstKey[0], firstKey[1]);
        Assert.True(cache.TryGet(a, ctxA, out Tensor hitA, out _));
        Assert.True(cache.TryGet(b, ctxB, out Tensor hitB, out _));
        Assert.Same(firstKey[0], hitA);
        Assert.Same(firstKey[1], hitB);
    }

    [Fact]
    public void OneSubBlockSharedByTwoThreadsOnTwoBackendsMatchesItsSingleThreadedOutputs()
    {
        Dictionary<string, Tensor> weights = SubBlockWeights();
        TransformerSubBlock shared = new(Channels, Heads, CtxDim);
        shared.LoadWeights(weights, "attn2", "norm2");
        TransformerSubBlock reference = new(Channels, Heads, CtxDim);
        reference.LoadWeights(weights, "attn2", "norm2");
        using CpuBackend a = new();
        using CpuBackend b = new();
        using CpuBackend single = new();
        Tensor[] contexts = [Seeded(new TensorShape(1, CtxTokens, CtxDim), 61), Seeded(new TensorShape(1, CtxTokens, CtxDim), 62)];
        Tensor[] hidden = [Seeded(new TensorShape(1, Tokens, Channels), 63), Seeded(new TensorShape(1, Tokens, Channels), 64)];
        float[][] expected = new float[2][];
        for (int i = 0; i < 2; i++) expected[i] = Read(reference.Forward(single, hidden[i], contexts[i]));
        Assert.NotEqual(expected[0], expected[1]);
        using Barrier barrier = new(2);
        Exception?[] failures = new Exception?[2];

        void Run(int index, IBackend backend)
        {
            barrier.SignalAndWait();
            for (int step = 0; step < 50; step++)
                Assert.Equal(expected[index], Read(shared.Forward(backend, hidden[index], contexts[index])));
        }

        Thread[] threads =
        [
            new(() => { try { Run(0, a); } catch (Exception ex) { failures[0] = ex; } }),
            new(() => { try { Run(1, b); } catch (Exception ex) { failures[1] = ex; } }),
        ];
        foreach (Thread t in threads) t.Start();
        foreach (Thread t in threads) t.Join();

        Assert.Null(failures[0]);
        Assert.Null(failures[1]);
    }

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>(t.DataPointer, checked((int)t.ElementCount)).ToArray();

    private static Dictionary<string, Tensor> SubBlockWeights() => new()
    {
        ["norm2.weight"] = Seeded(new TensorShape(Channels), 1, 0.25f),
        ["norm2.bias"] = Seeded(new TensorShape(Channels), 2, 0.1f),
        ["attn2.to_q.weight"] = Seeded(new TensorShape(Channels, Channels), 3, 0.3f),
        ["attn2.to_k.weight"] = Seeded(new TensorShape(Channels, CtxDim), 4, 0.3f),
        ["attn2.to_v.weight"] = Seeded(new TensorShape(Channels, CtxDim), 5, 0.3f),
        ["attn2.to_out.0.weight"] = Seeded(new TensorShape(Channels, Channels), 6, 0.3f),
        ["attn2.to_out.0.bias"] = Seeded(new TensorShape(Channels), 7, 0.1f),
    };

    private static Tensor Seeded(TensorShape shape, uint seed, float scale = 1f)
    {
        Tensor tensor = new(shape, DType.F32);
        float* data = (float*)tensor.DataPointer;
        uint state = seed * 2654435761u + 0x9E3779B9u;
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            state = state * 1664525u + 1013904223u;
            data[i] = (((state >> 8) & 0xFFFF) / 32768f - 1f) * scale;
        }
        return tensor;
    }
}
