using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>
/// State-machine tests for <see cref="BlockStreamingController"/>. The controller's
/// correctness is observable as the sequence of calls it issues to the cache, so a
/// fake cache that records calls is the natural test fixture — no GPU required.
///
/// <para>Validates: prime → forward window movement → eviction window movement,
/// across the four interesting prefetch / retain configurations (synchronous,
/// 1-deep prefetch, 2-deep prefetch, with retain). Also exercises edge cases:
/// model smaller than the prefetch window, calling forward in non-monotonic
/// order, double-prime, double-evict.</para>
/// </summary>
public sealed class BlockStreamingControllerTests
{
    /// <summary>Fake cache that records the order of every method invocation as
    /// human-readable strings. Reference equality on tokens is preserved by handing
    /// out monotonically increasing handles.</summary>
    private sealed class RecordingCache : IStreamingWeightCache
    {
        public List<string> Calls { get; } = new();
        private nint _nextHandle = 1;

        public StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights)
        {
            string blockId = ExtractBlockId(weights);
            Calls.Add($"upload:{blockId}");
            nint h = _nextHandle++;
            return new StreamingUploadToken(h, this);
        }

        public void AwaitWeights(StreamingUploadToken token)
        {
            if (token.IsEmpty)
            {
                Calls.Add("await:empty");
                return;
            }
            Calls.Add($"await:#{token.Handle}");
        }

        public void EvictAsync(IEnumerable<Tensor> weights)
        {
            Calls.Add($"evict:{ExtractBlockId(weights)}");
        }

        public long QueryAvailableWeightCacheBytes(long activationReserve) => long.MaxValue;

        public void DrainAndReleasePool() => Calls.Add("drain");

        private static string ExtractBlockId(IEnumerable<Tensor> weights)
        {
            // The fake block tags its weights via TaggedBlock.BlockId; we surface that
            // in call traces so tests can assert "upload:3" instead of opaque pointers.
            using IEnumerator<Tensor> e = weights.GetEnumerator();
            if (!e.MoveNext()) return "?";
            return TaggedBlock.IdFor(e.Current);
        }
    }

    /// <summary>Minimal IStreamingBlock that owns one tensor whose identity can be
    /// reverse-mapped to the originating block via a static side-table. Lets the
    /// recording cache produce readable trace lines without per-test plumbing.</summary>
    private sealed class TaggedBlock : IStreamingBlock
    {
        private static readonly Dictionary<Tensor, string> _tags = new();
        private readonly Tensor _tensor;
        public TaggedBlock(string id, long bytes)
        {
            _tensor = new Tensor(new TensorShape(1), DType.F32);
            _tags[_tensor] = id;
            EstimatedWeightBytes = bytes;
        }
        public long EstimatedWeightBytes { get; }
        public IEnumerable<Tensor> EnumerateWeights() { yield return _tensor; }
        public static string IdFor(Tensor t) => _tags.TryGetValue(t, out string? id) ? id : "?";
    }

    /// <summary>Backend stub that counts <see cref="IBackend.TrimMemoryPool"/> calls; every compute member is unreachable
    /// from the controller and throws if that ever stops being true.</summary>
    private sealed class TrimCountingBackend : IBackend
    {
        public int TrimCalls { get; private set; }

        public void TrimMemoryPool() => TrimCalls++;

        public DeviceKind Device => DeviceKind.Cpu;

        public BackendCapabilities Capabilities { get; } = new BackendCapabilities { Name = "trim-counting-stub" };

        public void Dispose() { }

        #region Unused compute surface

        private static Exception Unused([CallerMemberName] string member = "") =>
            new NotSupportedException($"{member} is not reachable from BlockStreamingController.");

        public void MatMul(Tensor output, Tensor a, Tensor b) => throw Unused();
        public void BatchedMatMul(Tensor output, Tensor a, Tensor b) => throw Unused();
        public void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias) => throw Unused();
        public void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW) => throw Unused();
        public void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias, int stride, int padding, int dilation, int groups, int outputPadding) => throw Unused();
        public void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias, int stride, int padding, int dilation, int groups, int outputPadding) => throw Unused();
        public void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps) => throw Unused();
        public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps) => throw Unused();
        public void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps) => throw Unused();
        public void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps) => throw Unused();
        public void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool allowF16 = false) => throw Unused();
        public void Gelu(Tensor output, Tensor input) => throw Unused();
        public void Silu(Tensor output, Tensor input) => throw Unused();
        public void Sigmoid(Tensor output, Tensor input) => throw Unused();
        public void Tanh(Tensor output, Tensor input) => throw Unused();
        public void Elu(Tensor output, Tensor input, float alpha) => throw Unused();
        public void LeakyRelu(Tensor output, Tensor input, float slope) => throw Unused();
        public void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta) => throw Unused();
        public void Add(Tensor output, Tensor a, Tensor b) => throw Unused();
        public void Mul(Tensor output, Tensor a, Tensor b) => throw Unused();
        public void Scale(Tensor output, Tensor input, float scalar) => throw Unused();
        public void Clamp(Tensor output, Tensor input, float min, float max) => throw Unused();
        public void Transpose2D(Tensor output, Tensor input, int d1, int d2) => throw Unused();
        public void Permute0213(Tensor output, Tensor input, int s, int h, int d) => throw Unused();
        public void GeGlu(Tensor output, Tensor input) => throw Unused();
        public void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial) => throw Unused();
        public void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim) => throw Unused();
        public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim) => throw Unused();
        public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW) => throw Unused();
        public void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW) => throw Unused();
        public void CopyTo(Tensor destination, Tensor source) => throw Unused();
        public void Fill(Tensor tensor, float value) => throw Unused();
        public void Fft(Tensor output, Tensor input) => throw Unused();
        public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window) => throw Unused();
        public void MelFilterbank(Tensor output, Tensor input, Tensor filters) => throw Unused();

        #endregion
    }

    private static (BlockStreamingController controller, RecordingCache cache, IReadOnlyList<TaggedBlock> blocks)
        Setup(int blockCount, int prefetchAhead, int retainBehind = 0, IBackend? backend = null)
    {
        TaggedBlock[] blocks = Enumerable.Range(0, blockCount)
            .Select(i => new TaggedBlock($"b{i}", bytes: 1024))
            .ToArray();
        RecordingCache cache = new RecordingCache();
        BlockStreamingController controller = new BlockStreamingController(cache, blocks, prefetchAhead, retainBehind, backend);
        return (controller, cache, blocks);
    }

    // ── Prime ────────────────────────────────────────────────────────────

    [Fact]
    public void Prime_With_PrefetchAhead_1_Uploads_Two_Blocks()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 5, prefetchAhead: 1);
        ctrl.Prime();
        // Should upload blocks [0..1] inclusive (block 0 itself + 1 ahead).
        Assert.Equal(new[] { "upload:b0", "upload:b1" }, cache.Calls);
    }

    [Fact]
    public void Prime_With_PrefetchAhead_0_Uploads_Only_First_Block()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 5, prefetchAhead: 0);
        ctrl.Prime();
        Assert.Equal(new[] { "upload:b0" }, cache.Calls);
    }

    [Fact]
    public void Prime_When_Model_Smaller_Than_Window_Stops_At_End()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 2, prefetchAhead: 5);
        ctrl.Prime();
        // Only blocks 0 and 1 exist — controller must not try to upload block 2+.
        Assert.Equal(new[] { "upload:b0", "upload:b1" }, cache.Calls);
    }

    [Fact]
    public void Prime_Is_Idempotent()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 3, prefetchAhead: 1);
        ctrl.Prime();
        ctrl.Prime();
        // Second call should be all no-ops — blocks 0, 1 are already Uploading.
        Assert.Equal(new[] { "upload:b0", "upload:b1" }, cache.Calls);
    }

    // ── BeforeBlockForward — main happy path ─────────────────────────────

    [Fact]
    public void Forward_Loop_With_PrefetchAhead_1_Retain_0_Issues_Expected_Trace()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 4, prefetchAhead: 1, retainBehind: 0);
        ctrl.Prime();
        for (int i = 0; i < 4; i++)
        {
            ctrl.BeforeBlockForward(i);
        }
        Assert.Equal(new[]
        {
            // Prime: blocks 0 and 1 are uploaded ahead (prefetchAhead=1 means [0..1] inclusive)
            "upload:b0", "upload:b1",
            // BeforeBlockForward(0): await 0; prefetch (0+1)=1 already Uploading, no-op; evict -1 oor
            "await:#1",
            // BeforeBlockForward(1): await 1; prefetch (1+1)=2 NEW upload; evict 0
            "await:#2", "upload:b2", "evict:b0",
            // BeforeBlockForward(2): await 2; prefetch 3 NEW; evict 1
            "await:#3", "upload:b3", "evict:b1",
            // BeforeBlockForward(3): await 3; prefetch 4 oor no-op; evict 2
            "await:#4", "evict:b2",
        }, cache.Calls);
    }

    [Fact]
    public void Forward_Loop_With_PrefetchAhead_0_Is_Synchronous()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 3, prefetchAhead: 0, retainBehind: 0);
        ctrl.Prime();
        for (int i = 0; i < 3; i++) ctrl.BeforeBlockForward(i);
        Assert.Equal(new[]
        {
            // Prime: only block 0
            "upload:b0",
            // Block 0: await 0, no prefetch, evict -1 (no-op)
            "await:#1",
            // Block 1: cold-path upload+await 1, no prefetch, evict 0
            "upload:b1", "await:#2", "evict:b0",
            // Block 2: cold upload+await 2, no prefetch, evict 1
            "upload:b2", "await:#3", "evict:b1",
        }, cache.Calls);
    }

    [Fact]
    public void Forward_Loop_With_PrefetchAhead_2_Keeps_Two_Uploads_In_Flight()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 5, prefetchAhead: 2, retainBehind: 0);
        ctrl.Prime();
        for (int i = 0; i < 5; i++) ctrl.BeforeBlockForward(i);
        Assert.Equal(new[]
        {
            // Prime uploads [0..2] (prefetchAhead=2 means block 0 + 2 ahead inclusive)
            "upload:b0", "upload:b1", "upload:b2",
            // Block 0: await 0; prefetch 0+2=2 already up; evict -1 oor
            "await:#1",
            // Block 1: await 1; prefetch 3 NEW; evict 0
            "await:#2", "upload:b3", "evict:b0",
            // Block 2: await 2; prefetch 4 NEW; evict 1
            "await:#3", "upload:b4", "evict:b1",
            // Block 3: await 3; prefetch 5 oor; evict 2
            "await:#4", "evict:b2",
            // Block 4: await 4; prefetch 6 oor; evict 3
            "await:#5", "evict:b3",
        }, cache.Calls);
    }

    [Fact]
    public void RetainBehind_1_Keeps_Last_Block_Cached()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 4, prefetchAhead: 1, retainBehind: 1);
        ctrl.Prime();
        for (int i = 0; i < 4; i++) ctrl.BeforeBlockForward(i);
        // Eviction target is i - retainBehind - 1 = i - 2; so first eviction at i=2 evicts block 0.
        Assert.Equal(new[]
        {
            // Prime: blocks [0..1]
            "upload:b0", "upload:b1",
            // i=0: await 0; prefetch 1 already up; evict -2 oor
            "await:#1",
            // i=1: await 1; prefetch 2 NEW; evict -1 oor
            "await:#2", "upload:b2",
            // i=2: await 2; prefetch 3 NEW; evict 0
            "await:#3", "upload:b3", "evict:b0",
            // i=3: await 3; prefetch 4 oor; evict 1
            "await:#4", "evict:b1",
        }, cache.Calls);
    }

    // ── EvictAll ─────────────────────────────────────────────────────────

    [Fact]
    public void EvictAll_Frees_Every_Resident_And_InFlight_Block()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 3, prefetchAhead: 1);
        ctrl.Prime();
        ctrl.BeforeBlockForward(0);
        ctrl.BeforeBlockForward(1);
        cache.Calls.Clear();

        ctrl.EvictAll();
        // Block 0 is Evicted (already evicted at i=1), block 1 is Resident, block 2 is Uploading.
        // EvictAll iterates 0..2:
        //  - 0: Evicted → no-op
        //  - 1: Resident → evict
        //  - 2: Uploading → await first, then evict
        // Then DrainAndReleasePool is called to flush the cache's stream-ordered pool.
        Assert.Equal(new[] { "evict:b1", "await:#3", "evict:b2", "drain" }, cache.Calls);
    }

    [Fact]
    public void EvictAll_Is_Idempotent()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 2, prefetchAhead: 1);
        ctrl.Prime();
        ctrl.EvictAll();
        cache.Calls.Clear();
        ctrl.EvictAll();
        // Per-block evictions are no-ops on the second call (everything's Evicted),
        // but the drain still runs — it's a transition signal, not bound to per-block state.
        Assert.Equal(new[] { "drain" }, cache.Calls);
    }

    // ── TrimAfterStep ────────────────────────────────────────────────────

    [Fact]
    public void TrimAfterStep_Issues_One_Backend_Trim_Per_Call()
    {
        TrimCountingBackend backend = new TrimCountingBackend();
        (BlockStreamingController ctrl, _, _) = Setup(blockCount: 3, prefetchAhead: 1, backend: backend);
        ctrl.Prime();
        ctrl.TrimAfterStep();
        ctrl.TrimAfterStep();
        Assert.Equal(2, backend.TrimCalls);
    }

    /// <summary>The per-step trim must NOT go through the cache's drain: that syncs the upload stream and would stall on
    /// the prefetch the step just primed.</summary>
    [Fact]
    public void TrimAfterStep_Does_Not_Drain_The_Cache()
    {
        TrimCountingBackend backend = new TrimCountingBackend();
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 3, prefetchAhead: 1, backend: backend);
        ctrl.Prime();
        cache.Calls.Clear();
        ctrl.TrimAfterStep();
        Assert.Empty(cache.Calls);
    }

    [Fact]
    public void TrimAfterStep_Without_Backend_Is_Inert()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 2, prefetchAhead: 1);
        ctrl.Prime();
        cache.Calls.Clear();
        ctrl.TrimAfterStep();
        Assert.Empty(cache.Calls);
    }

    [Fact]
    public void TrimAfterStep_After_Dispose_Throws()
    {
        TrimCountingBackend backend = new TrimCountingBackend();
        (BlockStreamingController ctrl, _, _) = Setup(blockCount: 2, prefetchAhead: 1, backend: backend);
        ctrl.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ctrl.TrimAfterStep());
        Assert.Equal(0, backend.TrimCalls);
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    [Fact]
    public void BeforeBlockForward_OutOfRange_Throws()
    {
        (BlockStreamingController ctrl, _, _) = Setup(blockCount: 3, prefetchAhead: 1);
        Assert.Throws<ArgumentOutOfRangeException>(() => ctrl.BeforeBlockForward(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ctrl.BeforeBlockForward(3));
    }

    [Fact]
    public void Forward_Without_Prime_Cold_Starts_Synchronously()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 3, prefetchAhead: 1);
        // No Prime() call.
        ctrl.BeforeBlockForward(0);
        // Block 0 was NotUploaded → upload+await synchronously, then prefetch 1 async.
        Assert.Equal(new[] { "upload:b0", "await:#1", "upload:b1" }, cache.Calls);
    }

    [Fact]
    public void Forward_Out_Of_Order_Re_Uploads_Evicted_Block()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 3, prefetchAhead: 1, retainBehind: 0);
        ctrl.Prime();
        ctrl.BeforeBlockForward(0);
        ctrl.BeforeBlockForward(1); // evicts block 0
        ctrl.BeforeBlockForward(2); // evicts block 1
        cache.Calls.Clear();

        // Going back to block 0 (which is now Evicted) — must re-upload from scratch
        // (cold path: upload + immediate await), then also prefetch block 1.
        ctrl.BeforeBlockForward(0);
        Assert.Equal(new[] { "upload:b0", "await:#4", "upload:b1" }, cache.Calls);
    }

    [Fact]
    public void Constructor_Validates_Arguments()
    {
        TaggedBlock[] blocks = { new("b0", 1) };
        RecordingCache cache = new RecordingCache();
        Assert.Throws<ArgumentNullException>(() => new BlockStreamingController(null!, blocks));
        Assert.Throws<ArgumentNullException>(() => new BlockStreamingController(cache, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BlockStreamingController(cache, blocks, prefetchAhead: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BlockStreamingController(cache, blocks, retainBehind: -1));
    }

    [Fact]
    public void EstimatedTotalWeightBytes_Sums_All_Blocks()
    {
        TaggedBlock[] blocks =
        {
            new("b0", 1000),
            new("b1", 2500),
            new("b2", 750),
        };
        RecordingCache cache = new RecordingCache();
        using BlockStreamingController ctrl = new BlockStreamingController(cache, blocks);
        Assert.Equal(4250, ctrl.EstimatedTotalWeightBytes);
    }

    [Fact]
    public void Dispose_Evicts_Everything()
    {
        (BlockStreamingController ctrl, RecordingCache cache, _) = Setup(blockCount: 3, prefetchAhead: 1);
        ctrl.Prime();
        ctrl.BeforeBlockForward(0);
        cache.Calls.Clear();

        ctrl.Dispose();
        // Should evict every still-resident block. Order: 0..2 in iteration.
        // After Prime+forward(0): block 0 = Resident, block 1 = Uploading, block 2 = NotUploaded.
        Assert.Contains("evict:b0", cache.Calls);
        Assert.Contains("evict:b1", cache.Calls);
        // Block 2 was never uploaded → no-op evict.
        Assert.DoesNotContain("evict:b2", cache.Calls);
    }

    [Fact]
    public void Dispose_Then_Forward_Throws()
    {
        (BlockStreamingController ctrl, _, _) = Setup(blockCount: 2, prefetchAhead: 1);
        ctrl.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ctrl.BeforeBlockForward(0));
        Assert.Throws<ObjectDisposedException>(() => ctrl.Prime());
    }
}
