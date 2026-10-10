using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>Placement tests for <see cref="BlockStreamingScope"/>. Every decision the scope makes is observable as
/// the sequence of calls it issues to the backend and the streaming cache, so scripted free-VRAM readings plus a
/// recording backend cover it without a GPU.
///
/// <para>The <c>Matches_Legacy</c> cases carry a verbatim transcription of the hand-rolled sizing
/// <c>LtxVideo2Pipeline</c> ran before the migration (<see cref="LegacyLtx2Plan"/>) and assert the scope reproduces
/// its prefix count AND its backend call sequence — the CPU-provable half of the byte-identity gate.</para></summary>
public sealed class BlockStreamingScopeTests
{
    private const long Mb = 1024 * 1024;
    private const long BlockBytes = 384 * Mb;

    /// <summary>Denoiser stub: one shared tensor plus <c>n</c> single-tensor blocks, all readably tagged.</summary>
    private sealed class FakeDenoiser : IStreamableDenoiser
    {
        private readonly TaggedStreamingBlock[] _blocks;
        private readonly TaggedStreamingBlock _shared;

        public FakeDenoiser(int blockCount, long blockBytes = BlockBytes)
        {
            _shared = new TaggedStreamingBlock("shared", 0);
            _blocks = new TaggedStreamingBlock[blockCount];
            for (int i = 0; i < blockCount; i++) _blocks[i] = new TaggedStreamingBlock($"b{i}", blockBytes);
        }

        public int BlockCount => _blocks.Length;

        public IStreamingBlock GetBlock(int idx) => _blocks[idx];

        public IEnumerable<Tensor> EnumerateSharedWeights() => _shared.EnumerateWeights();

        public Action<int>? BeforeBlockForward { get; set; }
    }

    /// <summary>Cache stub with a settable availability answer; upload/evict traffic is not the subject here.</summary>
    private sealed class StubCache : IStreamingWeightCache
    {
        public long Available { get; set; } = long.MaxValue;

        public StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights) => StreamingUploadToken.Empty;

        public void AwaitWeights(StreamingUploadToken token) { }

        public void EvictAsync(IEnumerable<Tensor> weights) { }

        public long QueryAvailableWeightCacheBytes(long activationReserve) => Available;

        public int Drains { get; private set; }

        public void DrainAndReleasePool() => Drains++;
    }

    private static BlockStreamingOptions Options(RecordingStreamingBackend backend, FakeDenoiser denoiser,
        long headroomBytes, long tokenLoad = 0, ResidentPrefixPin? pin = null,
        LowVramMode mode = LowVramMode.Auto, bool perStepTrim = true,
        BlockStreamingPolicy policy = BlockStreamingPolicy.ResidentPrefix)
        => new BlockStreamingOptions
        {
            Backend = backend,
            Denoiser = denoiser,
            ModelName = "Fake",
            HeadroomBytes = headroomBytes,
            TokenLoad = tokenLoad,
            Pin = pin,
            Mode = mode,
            PerStepTrim = perStepTrim,
            Policy = policy,
        };

    // ── Resident-vs-streamed decision ────────────────────────────────────

    [Fact]
    public void Sizes_The_Prefix_From_Free_Vram_Minus_Headroom()
    {
        // 8 blocks; 4 blocks' worth spendable after headroom.
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 4 * BlockBytes + 512 * Mb);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 512 * Mb));

        Assert.Equal(4, scope.ResidentPrefixBlocks);
        Assert.Equal(4, scope.StreamedBlocks);
        Assert.True(scope.Streaming);
    }

    [Fact]
    public void LowVram_Off_Keeps_Everything_Resident_Even_When_It_Cannot_Fit()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 0);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(
            Options(backend, denoiser, 4096 * Mb, mode: LowVramMode.ForceOff));

        Assert.Equal(8, scope.ResidentPrefixBlocks);
        Assert.False(scope.Streaming);
        Assert.Equal(new[] { "preload:shared,b0,b1,b2,b3,b4,b5,b6,b7" }, backend.Calls);
    }

    [Fact]
    public void LowVram_On_Releases_A_Resident_Prefix_And_Drops_Its_Pin()
    {
        ResidentPrefixPin pin = new ResidentPrefixPin { PinnedBlocks = 6, SizedTokens = 1000, Resident = true };
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 6 * BlockBytes);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(
            Options(backend, denoiser, 0, tokenLoad: 1000, pin: pin, mode: LowVramMode.ForceOn));

        Assert.Equal(0, scope.ResidentPrefixBlocks);
        Assert.Contains("free:b0,b1,b2,b3,b4,b5", backend.Calls);
        Assert.Equal(-1, pin.PinnedBlocks);
        Assert.False(pin.Resident);
    }

    // ── Trim ordering and geometry-triggered re-size ─────────────────────

    [Fact]
    public void Trims_The_Pool_Before_Reading_Free_Vram()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 2 * BlockBytes);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0));

        int trim = backend.Calls.IndexOf("trim");
        int read = backend.Calls.FindIndex(c => c.StartsWith("freeBytes:", StringComparison.Ordinal));
        Assert.True(trim >= 0 && read > trim, $"expected a trim before the first free-VRAM read; got [{string.Join(" ", backend.Calls)}]");
    }

    [Fact]
    public void A_Pinned_Prefix_Is_Reused_Without_Re_Sizing_At_The_Same_Geometry()
    {
        ResidentPrefixPin pin = new ResidentPrefixPin();
        RecordingStreamingBackend first = new RecordingStreamingBackend(new StubCache(), 4 * BlockBytes);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using (BlockStreamingScope.Open(Options(first, denoiser, 0, tokenLoad: 1000, pin: pin))) { }
        Assert.Equal(4, pin.PinnedBlocks);
        pin.Resident = true;

        // Free VRAM has since dropped to one block's worth; the pin must survive it.
        RecordingStreamingBackend second = new RecordingStreamingBackend(new StubCache(), 1 * BlockBytes);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(second, denoiser, 0, tokenLoad: 1000, pin: pin));

        Assert.Equal(4, scope.ResidentPrefixBlocks);
        Assert.DoesNotContain(second.Calls, c => c.StartsWith("freeBytes:", StringComparison.Ordinal));
    }

    [Fact]
    public void A_Bigger_Geometry_Releases_And_Re_Sizes_The_Pinned_Prefix()
    {
        ResidentPrefixPin pin = new ResidentPrefixPin { PinnedBlocks = 6, SizedTokens = 1000, Resident = true };
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 3 * BlockBytes);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0, tokenLoad: 2000, pin: pin));

        Assert.Equal(3, scope.ResidentPrefixBlocks);
        Assert.Equal(2000, pin.SizedTokens);
        Assert.Contains("free:b0,b1,b2,b3,b4,b5", backend.Calls);
        Assert.False(pin.Resident);
    }

    [Fact]
    public void A_Non_Resident_Pin_Is_Squeezed_To_What_Fits_But_Keeps_Its_Count()
    {
        // Sized at 6 blocks, then the weights were dropped; this generation only has room for 2.
        ResidentPrefixPin pin = new ResidentPrefixPin { PinnedBlocks = 6, SizedTokens = 1000, Resident = false };
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 2 * BlockBytes);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0, tokenLoad: 1000, pin: pin));

        Assert.Equal(2, scope.ResidentPrefixBlocks);
        Assert.Equal(6, pin.PinnedBlocks);
    }

    // ── The offset hook ──────────────────────────────────────────────────

    // ── The per-step trim cannot be silently skipped ─────────────────────

    [Fact]
    public void EndStep_Trims_The_Pool_Once_Per_Call_On_The_Streamed_Path()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 0);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0));
        int before = backend.Calls.Count(c => c == "trim");

        scope.EndStep();
        scope.EndStep();

        Assert.Equal(2, scope.StepsEnded);
        Assert.Equal(before + 2, backend.Calls.Count(c => c == "trim"));
    }

    [Fact]
    public void Dispose_Drains_The_Cache_Once_On_The_Streamed_Path()
    {
        StubCache cache = new StubCache();
        RecordingStreamingBackend backend = new RecordingStreamingBackend(cache, 0);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0));

        scope.Dispose();
        scope.Dispose();

        Assert.Equal(1, cache.Drains);
    }

    // ── Byte-identity against the hand-rolled LTX-2 sizing ───────────────

    /// <summary>Mutable copy of the three pipeline fields the pre-migration LTX-2 sizing carried across generations.</summary>
    private sealed class LegacyPin
    {
        public bool PrefixResident;
        public bool PrefixTailEvicted;
        public int ResidentPrefixBlocks = -1;
        public long PrefixSizedTokens = -1;
    }

    /// <summary>Verbatim transcription of <c>LtxVideo2Pipeline</c>'s hand-rolled sizing (lines ~279-350 at
    /// <c>882f9100</c>), against the same recording backend the scope runs on.</summary>
    private static int LegacyLtx2Plan(RecordingStreamingBackend backend, FakeDenoiser denoiser, LegacyPin pin,
        long headroomMb, long tokenLoad)
    {
        backend.PreloadWeights(denoiser.EnumerateSharedWeights());
        IStreamingBlock[] blocks = new IStreamingBlock[denoiser.BlockCount];
        for (int b = 0; b < blocks.Length; b++) blocks[b] = denoiser.GetBlock(b);
        long blockBytes = blocks[0].EstimatedWeightBytes;
        IEnumerable<Tensor> BlockRangeWeights(int from, int to)
        {
            for (int b = from; b < to; b++)
                foreach (Tensor t in blocks[b].EnumerateWeights()) yield return t;
        }
        if (pin.PrefixResident && tokenLoad > pin.PrefixSizedTokens)
        {
            backend.FreeWeights(BlockRangeWeights(0, pin.ResidentPrefixBlocks));
            backend.TrimMemoryPool();
            pin.PrefixResident = false;
            pin.ResidentPrefixBlocks = -1;
        }
        if (!pin.PrefixResident || pin.PrefixTailEvicted) { backend.TrimMemoryPool(); pin.PrefixTailEvicted = false; }
        if (pin.ResidentPrefixBlocks < 0 || tokenLoad > pin.PrefixSizedTokens)
        {
            long spendable = backend.FreeMemoryBytes() - headroomMb * 1024 * 1024;
            pin.ResidentPrefixBlocks = (int)Math.Clamp(spendable / Math.Max(blockBytes, 1), 0, blocks.Length);
            pin.PrefixSizedTokens = tokenLoad;
        }
        int residentBlocks = pin.ResidentPrefixBlocks;
        if (!pin.PrefixResident && residentBlocks > 0)
        {
            long spendable = backend.FreeMemoryBytes() - headroomMb * 1024 * 1024;
            int fit = (int)Math.Clamp(spendable / Math.Max(blockBytes, 1), 0, blocks.Length);
            if (fit < residentBlocks) residentBlocks = fit;
        }
        if (residentBlocks > 0) backend.PreloadWeights(BlockRangeWeights(0, residentBlocks));
        return residentBlocks;
    }

    public static IEnumerable<object[]> LegacySweep()
    {
        long[][] readings =
        [
            [0],
            [BlockBytes],
            [3 * BlockBytes + 3072 * Mb],
            [12 * BlockBytes, 2 * BlockBytes],
        ];
        foreach (bool resident in new[] { false, true })
            foreach (bool tailEvicted in new[] { false, true })
                foreach (int pinned in new[] { -1, 5 })
                    foreach (long sizedTokens in new long[] { -1, 1000 })
                        foreach (long tokenLoad in new long[] { 0, 5000 })
                            foreach (long[] free in readings)
                            {
                                yield return [resident, tailEvicted, pinned, sizedTokens, tokenLoad, free];
                            }
    }

    [Theory]
    [MemberData(nameof(LegacySweep))]
    public void Matches_Legacy_Ltx2_Sizing_And_Call_Sequence(
        bool resident, bool tailEvicted, int pinned, long sizedTokens, long tokenLoad, long[] freeReadings)
    {
        const int blockCount = 48;
        const long headroomMb = 3072;
        FakeDenoiser legacyDenoiser = new FakeDenoiser(blockCount);
        RecordingStreamingBackend legacyBackend = new RecordingStreamingBackend(new StubCache(), freeReadings);
        LegacyPin legacyPin = new LegacyPin
        {
            PrefixResident = resident,
            PrefixTailEvicted = tailEvicted,
            ResidentPrefixBlocks = pinned,
            PrefixSizedTokens = sizedTokens,
        };
        int legacyPrefix = LegacyLtx2Plan(legacyBackend, legacyDenoiser, legacyPin, headroomMb, tokenLoad);

        FakeDenoiser denoiser = new FakeDenoiser(blockCount);
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), freeReadings);
        ResidentPrefixPin pin = new ResidentPrefixPin
        {
            Resident = resident,
            TailEvicted = tailEvicted,
            PinnedBlocks = pinned,
            SizedTokens = sizedTokens,
        };
        using BlockStreamingScope scope = BlockStreamingScope.Open(new BlockStreamingOptions
        {
            Backend = backend,
            Denoiser = denoiser,
            ModelName = "LTX-2",
            HeadroomBytes = headroomMb * 1024 * 1024,
            TokenLoad = tokenLoad,
            Pin = pin,
            Mode = LowVramMode.Auto,
            PerStepTrim = false,
        });

        Assert.Equal(legacyPrefix, scope.ResidentPrefixBlocks);
        // Block tags are per-denoiser instances but the ids repeat, so the recorded strings compare directly.
        // graph-reset is excluded deliberately: the hand-rolled LTX-2 plan this pins parity against never invalidated
        // the step graph before freeing the prefix, which is the latent hazard the scope now fixes. Parity is about
        // WHICH weights move and in what order, not about reproducing that omission.
        Assert.Equal(legacyBackend.Calls, backend.Calls.Where(c => c != "graph-reset").ToList());
        Assert.Equal(legacyPin.ResidentPrefixBlocks, pin.PinnedBlocks);
        Assert.Equal(legacyPin.PrefixSizedTokens, pin.SizedTokens);
        Assert.Equal(legacyPin.PrefixResident, pin.Resident);
        Assert.Equal(legacyPin.PrefixTailEvicted, pin.TailEvicted);
    }

    [Theory]
    [InlineData(0, 0, 384, 48, 0)]
    [InlineData(768, 0, 384, 48, 2)]
    [InlineData(1000, 0, 0, 48, 48)]
    [InlineData(long.MaxValue, 0, 384, 48, 48)]
    public void Size_Matches_The_Legacy_Clamp(long free, long headroom, long blockBytes, int blockCount, int expected)
    {
        int legacy = (int)Math.Clamp((free - headroom) / Math.Max(blockBytes, 1), 0, blockCount);
        Assert.Equal(expected, legacy);
        Assert.Equal(legacy, ResidentPrefixSizing.Size(free, headroom, blockBytes, blockCount));
    }

    // ── Forced streaming must displace an already-resident set ───────────

    /// <summary>A warm denoiser's own weights fill the VRAM the planner measures, so a free-VRAM reading below the weight
    /// size must not stream it. Without the pin, warm generations alternate between resident and streamed.</summary>
    [Fact]
    public void AllOrNothing_AWarmPin_StaysResidentThoughFreeVramIsBelowTheWeights()
    {
        // Room for the activations beside the pin, but not for the eight blocks if they were not already there.
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache { Available = 2 * BlockBytes }, 0);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        ResidentPrefixPin pin = new ResidentPrefixPin { Resident = true };

        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, BlockBytes, pin: pin,
            policy: BlockStreamingPolicy.AllOrNothing));

        Assert.Equal(8, scope.ResidentPrefixBlocks);
        Assert.False(scope.Streaming);
        Assert.DoesNotContain(backend.Calls, c => c.StartsWith("free:", StringComparison.Ordinal));
    }

    /// <summary>A pin kept for a small generation must not carry a larger one whose activations no longer fit
    /// beside the resident weights: the blocks are released and the planner streams.</summary>
    [Fact]
    public void AllOrNothing_AWarmPin_IsReleasedWhenTheActivationsNoLongerFit()
    {
        // Nothing left once the larger activation reserve is taken.
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache { Available = 0 }, 0);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        ResidentPrefixPin pin = new ResidentPrefixPin { Resident = true };

        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 4 * BlockBytes, pin: pin,
            policy: BlockStreamingPolicy.AllOrNothing));

        Assert.False(pin.Resident);
        Assert.True(scope.Streaming);
        Assert.Contains(backend.Calls, c => c.StartsWith("free:", StringComparison.Ordinal));
    }

    /// <summary>Without an eviction ahead of the plan, <see cref="VramPlanner.PlanPhase"/> answers Resident for a warm
    /// pin (its availability query cannot see past the weights occupying the space it measures), so a forced stream
    /// silently stays resident — the setting appearing to do nothing on exactly the generations it was set for.</summary>
    [Fact]
    public void AllOrNothing_ForcedStream_EvictsTheWarmPinAndStreams()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), long.MaxValue);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        ResidentPrefixPin pin = new ResidentPrefixPin { Resident = true };

        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0, pin: pin,
            mode: LowVramMode.ForceOn, policy: BlockStreamingPolicy.AllOrNothing));

        Assert.Equal(0, scope.ResidentPrefixBlocks);
        Assert.True(scope.Streaming);
        Assert.False(pin.Resident);
        Assert.Contains(backend.Calls, c => c.StartsWith("free:", StringComparison.Ordinal));
    }

    /// <summary>An unsized pin means the whole set is resident under this policy — freeing a <c>-1</c> range would free
    /// nothing and leave the force inert a second way.</summary>
    /// <summary>A captured graph bakes pointers into the very weights this scope frees (LTX-2 replays one over it), so
    /// the invalidation has to be ordered BEFORE the release, not merely present somewhere in the call sequence.</summary>
    [Fact]
    public void ForcedStream_InvalidatesTheStepGraphBeforeFreeingTheResidentPrefix()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), long.MaxValue)
        {
            StepGraphReady = true,
            StepGraphOwner = new object(),
        };
        FakeDenoiser denoiser = new FakeDenoiser(8);
        ResidentPrefixPin pin = new ResidentPrefixPin { PinnedBlocks = 4, SizedTokens = 1000, Resident = true };

        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0, tokenLoad: 1000,
            pin: pin, mode: LowVramMode.ForceOn));

        int reset = backend.Calls.IndexOf("graph-reset");
        int freed = backend.Calls.FindIndex(c => c.StartsWith("free:", StringComparison.Ordinal));
        Assert.True(reset >= 0, "the step graph was never invalidated");
        Assert.True(freed >= 0, "the resident prefix was never freed");
        Assert.True(reset < freed, $"invalidate must precede free, got {string.Join(",", backend.Calls)}");
        Assert.Null(backend.StepGraphOwner);
    }
}
