using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The shared key and value store: ownership on every path, segment order, the token axis, and what it refuses.</summary>
public sealed class KvSegmentStoreTests : IDisposable
{
    private readonly CpuBackend _backend = new();

    public void Dispose() => _backend.Dispose();

    [Fact]
    public void Join_ConcatenatesSegmentsOfDifferentLengthsInOrder()
    {
        using KvSegmentStore store = new(_backend, layers: 2, tokenAxis: 0);
        store.Append(1, Filled(1, 2, 3), Filled(-1, 2, 3));
        store.Append(1, Filled(2, 1, 3), Filled(-2, 1, 3));
        store.Append(1, Filled(3, 3, 3), Filled(-3, 3, 3));

        (Tensor keys, Tensor values) = store.Join(1, 1, 2);
        using (keys)
        using (values)
        {
            Assert.Equal(new TensorShape(4, 3), keys.Shape);
            Assert.Equal([2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3], keys.AsReadOnlySpan<float>().ToArray());
            Assert.Equal(-3f, values.AsReadOnlySpan<float>()[11]);
        }
        Assert.Equal(6, store.Tokens(1));
        Assert.Equal(0, store.Tokens(0));
    }

    [Fact]
    public void JoinOfManySegmentsKeepsEveryTokenInOrderOnAHeadMajorAxis()
    {
        // Five segments: more than two, so the join is the strided path. Two heads, dim 3; each segment has its own value and length.
        int[] lengths = [1, 2, 3, 1, 2];
        using KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 2);
        for (int s = 0; s < lengths.Length; s++)
            store.Append(0, Filled(s + 1, 1, 2, lengths[s], 3), Filled(-(s + 1), 1, 2, lengths[s], 3));

        (Tensor keys, Tensor values) = store.Join(0, 0, lengths.Length);
        using (keys)
        using (values)
        {
            int total = lengths.Sum();
            Assert.Equal(new TensorShape(1, 2, total, 3), keys.Shape);
            List<float> expectedKeys = [], expectedValues = [];
            for (int head = 0; head < 2; head++)
                for (int s = 0; s < lengths.Length; s++)
                    for (int t = 0; t < lengths[s] * 3; t++)
                    {
                        expectedKeys.Add(s + 1);
                        expectedValues.Add(-(s + 1));
                    }
            Assert.Equal(expectedKeys, keys.AsReadOnlySpan<float>().ToArray());
            Assert.Equal(expectedValues, values.AsReadOnlySpan<float>().ToArray());
        }
    }

    [Fact]
    public void JoinOfManySegmentsKeepsEveryTokenInOrderOnTheLeadingAxis()
    {
        int[] lengths = [2, 1, 3, 1];
        using KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        for (int s = 0; s < lengths.Length; s++)
            store.Append(0, Filled(s + 1, lengths[s], 2), Filled(-(s + 1), lengths[s], 2));

        (Tensor keys, Tensor values) = store.Join(0, 0, lengths.Length);
        using (keys)
        using (values)
        {
            Assert.Equal(new TensorShape(7, 2), keys.Shape);
            List<float> expected = [];
            for (int s = 0; s < lengths.Length; s++)
                for (int i = 0; i < lengths[s] * 2; i++)
                    expected.Add(s + 1);
            Assert.Equal(expected, keys.AsReadOnlySpan<float>().ToArray());
            Assert.Equal(-4f, values.AsReadOnlySpan<float>()[13]);
        }
    }

    [Fact]
    public void Join_UsesTheTokenAxisOfAHeadMajorStore()
    {
        using KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 2);
        store.Append(0, Filled(1, 1, 2, 1, 2), Filled(1, 1, 2, 1, 2));
        store.Append(0, Filled(2, 1, 2, 2, 2), Filled(2, 1, 2, 2, 2));

        (Tensor keys, Tensor values) = store.Join(0, 0, 2);
        using (keys)
        using (values)
        {
            Assert.Equal(new TensorShape(1, 2, 3, 2), keys.Shape);
            Assert.Equal([1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2], keys.AsReadOnlySpan<float>().ToArray());
        }
    }

    [Fact]
    public void Set_TakesOwnership_AndFreesWhatItReplaces()
    {
        KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        Tensor first = Filled(1, 2, 2), firstValues = Filled(1, 2, 2), second = Filled(2, 3, 2), secondValues = Filled(2, 3, 2);
        store.Set(0, 0, first, firstValues);
        Assert.Same(first, store.Keys(0, 0));

        store.Set(0, 0, second, secondValues);
        Assert.True(IsDisposed(first));
        Assert.True(IsDisposed(firstValues));
        Assert.Same(second, store.Keys(0, 0));
        Assert.Same(secondValues, store.Values(0, 0));

        store.Dispose();
        Assert.True(IsDisposed(second));
        Assert.True(IsDisposed(secondValues));
    }

    [Fact]
    public void Set_DisposesTheOperandsWhenItRefusesThem_AndKeepsTheSlot()
    {
        using KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        Tensor kept = Filled(1, 2, 2);
        store.Set(0, 0, kept, Filled(1, 2, 2));

        void Refused(Tensor keys, Tensor values)
        {
            Assert.Throws<ArgumentException>(() => store.Set(0, 0, keys, values));
            Assert.True(IsDisposed(keys));
            Assert.True(IsDisposed(values));
        }

        Refused(Filled(1, 2, 2), Filled(1, 3, 2));
        Refused(Filled(1, 2, 3), Filled(1, 2, 3));
        Refused(Filled(1, 1, 2, 2), Filled(1, 1, 2, 2));
        Refused(new Tensor(new TensorShape(2, 2), DType.F16), new Tensor(new TensorShape(2, 2), DType.F16));
        Assert.Same(kept, store.Keys(0, 0));
        Assert.False(IsDisposed(kept));
    }

    [Fact]
    public void AFirstSegmentThatIsRefused_DoesNotFixTheShape()
    {
        using KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        Assert.Throws<ArgumentException>(() => store.Set(0, 0, Filled(1, 2, 2), Filled(1, 3, 2)));
        store.Set(0, 0, Filled(1, 2, 7), Filled(1, 2, 7));
        Assert.Equal(2, store.Tokens(0));
    }

    [Fact]
    public void Set_AfterDispose_IsRefusedAndDisposesTheOperands()
    {
        KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        store.Dispose();
        store.Dispose();
        Tensor keys = Filled(1, 2, 2), values = Filled(1, 2, 2);

        Assert.Throws<ObjectDisposedException>(() => store.Set(0, 0, keys, values));

        Assert.True(IsDisposed(keys));
        Assert.True(IsDisposed(values));
    }

    [Fact]
    public void AnUnwrittenSlot_IsEmpty_AndAJoinOverItIsRefused()
    {
        using KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        store.Set(0, 2, Filled(1, 2, 2), Filled(1, 2, 2));

        Assert.Equal(3, store.Count(0));
        Assert.False(store.Has(0, 0));
        Assert.True(store.Has(0, 2));
        Assert.False(store.Has(0, 9));
        Assert.Throws<InvalidOperationException>(() => store.Keys(0, 1));
        Assert.Throws<InvalidOperationException>(() => store.Join(0, 0, 3));
        Assert.Throws<InvalidOperationException>(() => store.Join(0, 2, 2));
    }

    [Fact]
    public void RemoveRange_FreesTheDropped_AndTheRestMoveDown()
    {
        using KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        Tensor[] keys = [.. Enumerable.Range(1, 4).Select(i => Filled(i, 1, 2))];
        foreach (Tensor k in keys) store.Append(0, k, Filled(0, 1, 2));

        store.RemoveRange(0, 1, 2);

        Assert.Equal(2, store.Count(0));
        Assert.Same(keys[0], store.Keys(0, 0));
        Assert.Same(keys[3], store.Keys(0, 1));
        Assert.True(IsDisposed(keys[1]));
        Assert.True(IsDisposed(keys[2]));
        Assert.Throws<ArgumentOutOfRangeException>(() => store.RemoveRange(0, 1, 2));
    }

    [Fact]
    public void Compact_LeavesOneSegmentWithTheSameTokens_AndFreesTheOldOnes()
    {
        using KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        Tensor first = Filled(1, 2, 2), second = Filled(2, 1, 2);
        store.Append(0, first, Filled(-1, 2, 2));
        store.Append(0, second, Filled(-2, 1, 2));
        long bytes = store.Bytes;

        store.Compact(0);

        Assert.Equal(1, store.Count(0));
        Assert.Equal([1, 1, 1, 1, 2, 2], store.Keys(0, 0).AsReadOnlySpan<float>().ToArray());
        Assert.Equal([-1, -1, -1, -1, -2, -2], store.Values(0, 0).AsReadOnlySpan<float>().ToArray());
        Assert.True(IsDisposed(first));
        Assert.True(IsDisposed(second));
        Assert.Equal(bytes, store.Bytes);
        Tensor only = store.Keys(0, 0);
        store.Compact(0);
        Assert.Same(only, store.Keys(0, 0));
    }

    [Fact]
    public void EstimateBytes_IsWhatAFullStoreHolds()
    {
        using KvSegmentStore store = new(_backend, layers: 3, tokenAxis: 0);
        for (int layer = 0; layer < 3; layer++)
            for (int i = 0; i < 2; i++)
                store.Append(layer, Filled(0, 5, 4), Filled(0, 5, 4));

        Assert.Equal(store.Bytes, KvSegmentStore.EstimateBytes(layers: 3, tokens: 10, width: 4, DType.F32));
    }

    [Fact]
    public void WritingASlotWithTensorsItHolds_KeepsThem_AndFreesOnlyTheOther()
    {
        using KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        Tensor keys = Filled(1, 2, 2), values = Filled(2, 2, 2), newValues = Filled(3, 2, 2);
        store.Set(0, 0, keys, values);
        store.Set(0, 0, keys, values);
        Assert.False(IsDisposed(keys));
        Assert.False(IsDisposed(values));

        store.Set(0, 0, keys, newValues);
        Assert.False(IsDisposed(keys));
        Assert.True(IsDisposed(values));
        Assert.Same(newValues, store.Values(0, 0));
    }

    [Fact]
    public void ARefusedRewriteKeepsATensorTheSlotHolds()
    {
        IBackend backend = DispatchProxy.Create<IBackend, PinRecorder>();
        PinRecorder pins = (PinRecorder)backend;
        using KvSegmentStore store = new(backend, layers: 1, tokenAxis: 0);
        Tensor keys = Filled(1, 2, 2), values = Filled(1, 2, 2);
        store.Append(0, keys, values);

        Tensor wrong = Filled(2, 3, 2);
        Assert.Throws<ArgumentException>(() => store.Set(0, 0, keys, wrong));
        Tensor unpinnable = Filled(3, 2, 2);
        pins.Refuse = unpinnable;
        Assert.Throws<InvalidOperationException>(() => store.Set(0, 0, keys, unpinnable));

        Assert.True(IsDisposed(wrong));
        Assert.True(IsDisposed(unpinnable));
        Assert.Same(keys, store.Keys(0, 0));
        Assert.Same(values, store.Values(0, 0));
        Assert.False(IsDisposed(keys));
        Assert.Contains(keys, pins.Pinned);
        Assert.Equal(2, pins.Pinned.Count);
    }

    [Fact]
    public void OneTensorForBothIsRefused_AndDisposed()
    {
        using KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        Tensor both = Filled(1, 2, 2);
        Assert.Throws<ArgumentException>(() => store.Set(0, 0, both, both));
        Assert.True(IsDisposed(both));
    }

    [Fact]
    public void AFailedPin_ChangesNothingButDisposesTheOperands()
    {
        IBackend backend = DispatchProxy.Create<IBackend, PinRecorder>();
        PinRecorder pins = (PinRecorder)backend;
        KvSegmentStore store = new(backend, layers: 1, tokenAxis: 0);
        store.Append(0, Filled(1, 2, 2), Filled(1, 2, 2));
        Tensor keys = Filled(2, 2, 2), values = Filled(2, 2, 2);
        pins.Refuse = values;

        Assert.Throws<InvalidOperationException>(() => store.Append(0, keys, values));

        Assert.Equal(1, store.Count(0));
        Assert.True(IsDisposed(keys));
        Assert.True(IsDisposed(values));
        Assert.Equal(2, pins.Pinned.Count);
        store.Dispose();
        Assert.Empty(pins.Pinned);
    }

    [Fact]
    public void AnEmptiedStore_TakesAnotherShape()
    {
        using KvSegmentStore store = new(_backend, layers: 2, tokenAxis: 0);
        store.Append(0, Filled(1, 2, 2), Filled(1, 2, 2));
        store.Append(1, Filled(1, 2, 2), Filled(1, 2, 2));
        store.Clear(0);
        Assert.Throws<ArgumentException>(() => store.Append(0, Filled(1, 2, 5), Filled(1, 2, 5)));
        store.Clear(1);
        store.Append(0, Filled(1, 2, 5), Filled(1, 2, 5));
        Assert.Equal(2, store.Tokens(0));
    }

    [Fact]
    public void ALayerOutOfRange_AndReadsAfterDispose_AreRefused()
    {
        KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        Tensor keys = Filled(1, 2, 2), values = Filled(1, 2, 2);
        Assert.Throws<ArgumentOutOfRangeException>(() => store.Append(3, keys, values));
        Assert.True(IsDisposed(keys));
        Assert.Throws<ArgumentOutOfRangeException>(() => store.Keys(-1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => store.Count(1));
        Assert.False(store.Has(0, -1));
        Assert.False(store.Has(7, 0));
        store.Append(0, Filled(1, 2, 2), Filled(1, 2, 2));
        store.Dispose();
        Assert.Throws<ObjectDisposedException>(() => store.Keys(0, 0));
        Assert.Throws<ObjectDisposedException>(() => store.Join(0, 0, 1));
        Assert.Throws<ObjectDisposedException>(() => store.Compact(0));
    }

    [Fact]
    public void EveryStoredTensorIsPinned_AndEveryFreedOneIsUnpinnedFirst()
    {
        IBackend backend = DispatchProxy.Create<IBackend, PinRecorder>();
        PinRecorder pins = (PinRecorder)backend;
        KvSegmentStore store = new(backend, layers: 1, tokenAxis: 0);
        Tensor first = Filled(1, 2, 2), second = Filled(2, 2, 2), third = Filled(3, 2, 2);

        store.Append(0, first, Filled(1, 2, 2));
        store.Append(0, second, Filled(2, 2, 2));
        Assert.Equal(4, pins.Pinned.Count);

        store.Set(0, 0, third, Filled(3, 2, 2));
        Assert.Equal(4, pins.Pinned.Count);
        Assert.DoesNotContain(first, pins.Pinned);
        Assert.Throws<ArgumentException>(() => store.Set(0, 0, Filled(1, 2, 2), Filled(1, 3, 2)));
        Assert.Equal(4, pins.Pinned.Count);

        store.RemoveRange(0, 0, 1);
        Assert.Equal(2, pins.Pinned.Count);
        Assert.Contains(second, pins.Pinned);
        store.Dispose();
        Assert.Empty(pins.Pinned);
        Assert.False(pins.UnpinnedAfterDispose);
    }

    /// <summary>Tracks pins; the store calls nothing else on these paths.</summary>
    public class PinRecorder : DispatchProxy
    {
        public HashSet<Tensor> Pinned { get; } = new(ReferenceEqualityComparer.Instance);

        public bool UnpinnedAfterDispose { get; private set; }

        /// <summary>The tensor whose pin fails.</summary>
        public Tensor? Refuse { get; set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            Tensor tensor = (Tensor)args![0]!;
            switch (targetMethod!.Name)
            {
                case nameof(IBackend.PinActivation):
                    if (ReferenceEquals(tensor, Refuse)) throw new InvalidOperationException("the pin was refused");
                    Assert.True(Pinned.Add(tensor));
                    break;
                case nameof(IBackend.UnpinActivation):
                    UnpinnedAfterDispose |= IsDisposed(tensor);
                    Assert.True(Pinned.Remove(tensor));
                    break;
                default:
                    throw new NotSupportedException(targetMethod.Name);
            }
            return null;
        }
    }

    private static Tensor Filled(float value, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    private static bool IsDisposed(Tensor tensor)
    {
        try
        {
            _ = tensor.AsReadOnlySpan<byte>().Length;
            return false;
        }
        catch (ObjectDisposedException)
        {
            return true;
        }
    }

    [Fact]
    public void MoveLayerToHost_KeepsBf16SegmentsBitExact()
    {
        using KvSegmentStore store = new(_backend, layers: 1, tokenAxis: 0);
        Tensor keys = BFloat16(1, 2, 4), values = BFloat16(-3, 2, 4);
        byte[] keyBytes = keys.AsReadOnlySpan<byte>().ToArray(), valueBytes = values.AsReadOnlySpan<byte>().ToArray();
        store.Append(0, keys, values);

        store.MoveLayerToHost(0);

        Assert.Equal(new TensorShape(2, 4), store.Keys(0, 0).Shape);
        Assert.Equal(DType.BF16, store.Keys(0, 0).DType);
        Assert.Equal(keyBytes, store.Keys(0, 0).AsReadOnlySpan<byte>().ToArray());
        Assert.Equal(valueBytes, store.Values(0, 0).AsReadOnlySpan<byte>().ToArray());
    }

    private static Tensor BFloat16(float start, int rows, int cols)
    {
        Tensor tensor = new(new TensorShape(rows, cols), DType.BF16);
        Span<byte> bytes = tensor.AsSpan<byte>();
        for (int i = 0; i < rows * cols; i++)
        {
            // The high 16 bits of the float: a valid BF16 value, distinct per element.
            uint bits = BitConverter.SingleToUInt32Bits(start + i * 0.5f) >> 16;
            bytes[2 * i] = (byte)bits;
            bytes[2 * i + 1] = (byte)(bits >> 8);
        }
        return tensor;
    }
}
