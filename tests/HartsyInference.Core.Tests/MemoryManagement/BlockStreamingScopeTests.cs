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
    public void Everything_Resident_When_It_Fits_And_No_Streamer_Is_Attached()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 100L * BlockBytes);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 512 * Mb));

        Assert.Equal(8, scope.ResidentPrefixBlocks);
        Assert.False(scope.Streaming);
        Assert.Null(denoiser.BeforeBlockForward);
    }

    [Fact]
    public void Nothing_Resident_When_Headroom_Exceeds_Free_Vram()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 100 * Mb);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 4096 * Mb));

        Assert.Equal(0, scope.ResidentPrefixBlocks);
        Assert.Equal(8, scope.StreamedBlocks);
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
    public void LowVram_On_Leaves_The_Pin_Untouched_So_The_Next_Auto_Generation_Sizes_From_Scratch()
    {
        ResidentPrefixPin pin = new ResidentPrefixPin();
        FakeDenoiser denoiser = new FakeDenoiser(8);
        RecordingStreamingBackend forced = new RecordingStreamingBackend(new StubCache(), 6 * BlockBytes);
        using (BlockStreamingScope.Open(Options(forced, denoiser, 0, tokenLoad: 1000, pin: pin, mode: LowVramMode.ForceOn))) { }

        // A prefix sized under forced streaming is never uploaded, so recording it would make the next generation
        // squeeze against a count describing VRAM nobody parked in.
        Assert.Equal(-1, pin.PinnedBlocks);
        Assert.Equal(-1, pin.SizedTokens);
        Assert.False(pin.Resident);

        RecordingStreamingBackend auto = new RecordingStreamingBackend(new StubCache(), 3 * BlockBytes);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(auto, denoiser, 0, tokenLoad: 1000, pin: pin));

        Assert.Equal(3, scope.ResidentPrefixBlocks);
        Assert.Equal(3, pin.PinnedBlocks);
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

    [Fact]
    public void LowVram_On_Streams_Every_Block_Even_With_Room_To_Spare()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 100L * BlockBytes);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(
            Options(backend, denoiser, 512 * Mb, mode: LowVramMode.ForceOn));

        Assert.Equal(0, scope.ResidentPrefixBlocks);
        Assert.Equal(8, scope.StreamedBlocks);
    }

    [Fact]
    public void No_Streaming_Cache_Preloads_Everything()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(cache: null, 0);
        FakeDenoiser denoiser = new FakeDenoiser(4);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 512 * Mb));

        Assert.False(scope.Streaming);
        Assert.Equal(new[] { "preload:shared,b0,b1,b2,b3" }, backend.Calls);
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
    public void A_Smaller_Geometry_Does_Not_Disturb_The_Pin()
    {
        ResidentPrefixPin pin = new ResidentPrefixPin { PinnedBlocks = 6, SizedTokens = 2000, Resident = true };
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 1 * BlockBytes);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0, tokenLoad: 500, pin: pin));

        Assert.Equal(6, scope.ResidentPrefixBlocks);
        Assert.Equal(2000, pin.SizedTokens);
        Assert.DoesNotContain(backend.Calls, c => c.StartsWith("free:", StringComparison.Ordinal));
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

    [Fact]
    public void A_Tail_Eviction_Forces_A_Trim_Before_The_Top_Up()
    {
        ResidentPrefixPin pin = new ResidentPrefixPin { PinnedBlocks = 4, SizedTokens = 1000, Resident = true, TailEvicted = true };
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 8 * BlockBytes);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using (BlockStreamingScope.Open(Options(backend, denoiser, 0, tokenLoad: 1000, pin: pin))) { }

        Assert.Contains("trim", backend.Calls);
        Assert.False(pin.TailEvicted);
    }

    // ── The offset hook ──────────────────────────────────────────────────

    [Fact]
    public void The_Hook_Rebases_Block_Indexes_Onto_The_Streamed_Suffix()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 3 * BlockBytes);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0));

        Assert.Equal(3, scope.ResidentPrefixBlocks);
        Assert.NotNull(denoiser.BeforeBlockForward);
        // Resident indexes must not reach the controller at all — it only knows about the 5 streamed blocks, and
        // index 0..2 would otherwise drive the wrong block's upload.
        for (int i = 0; i < 8; i++) denoiser.BeforeBlockForward!(i);
    }

    [Fact]
    public void Dispose_Unhooks_The_Denoiser()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 0);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0));
        Assert.NotNull(denoiser.BeforeBlockForward);
        scope.Dispose();
        Assert.Null(denoiser.BeforeBlockForward);
    }

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
    public void EndStep_Is_Inert_When_The_Caller_Opted_Out()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 0);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0, perStepTrim: false));
        int before = backend.Calls.Count(c => c == "trim");

        scope.EndStep();

        Assert.Equal(1, scope.StepsEnded);
        Assert.Equal(before, backend.Calls.Count(c => c == "trim"));
    }

    [Fact]
    public void A_Streamed_Loop_That_Never_Calls_EndStep_Is_Reported()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 0);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0));

        // The warning is the mechanism; StepsEnded is the assertable half of it, and it must survive Dispose so a
        // test (or a caller) can tell a skipped trim from a resident run that legitimately has nothing to trim.
        scope.Dispose();

        Assert.True(scope.Streaming);
        Assert.Equal(0, scope.StepsEnded);
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

    [Fact]
    public void Dispose_Never_Drains_When_Nothing_Was_Streamed()
    {
        StubCache cache = new StubCache();
        RecordingStreamingBackend backend = new RecordingStreamingBackend(cache, 100L * BlockBytes);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0));
        Assert.False(scope.Streaming);

        scope.Dispose();

        Assert.Equal(0, cache.Drains);
    }

    [Fact]
    public void EndStep_After_Dispose_Throws()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), 0);
        FakeDenoiser denoiser = new FakeDenoiser(8);
        BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0));
        scope.Dispose();

        Assert.Throws<ObjectDisposedException>(() => scope.EndStep());
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
            [512 * Mb],
            [BlockBytes],
            [3 * BlockBytes + 3072 * Mb],
            [48L * BlockBytes + 3072 * Mb],
            [200L * BlockBytes],
            [12 * BlockBytes, 2 * BlockBytes],
            [2 * BlockBytes, 12 * BlockBytes],
        ];
        foreach (bool resident in new[] { false, true })
            foreach (bool tailEvicted in new[] { false, true })
                foreach (int pinned in new[] { -1, 0, 5, 48 })
                    foreach (long sizedTokens in new long[] { -1, 1000 })
                        foreach (long tokenLoad in new long[] { 0, 1000, 5000 })
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
    [InlineData(-1, 0, 384, 48, 0)]
    [InlineData(1000, 2000, 384, 48, 0)]
    [InlineData(768, 0, 384, 48, 2)]
    [InlineData(767, 0, 384, 48, 1)]
    [InlineData(1000, 0, 0, 48, 48)]
    [InlineData(long.MaxValue, 0, 384, 48, 48)]
    public void Size_Matches_The_Legacy_Clamp(long free, long headroom, long blockBytes, int blockCount, int expected)
    {
        int legacy = (int)Math.Clamp((free - headroom) / Math.Max(blockBytes, 1), 0, blockCount);
        Assert.Equal(expected, legacy);
        Assert.Equal(legacy, ResidentPrefixSizing.Size(free, headroom, blockBytes, blockCount));
    }

    // ── Forced streaming must displace an already-resident set ───────────

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
    [Fact]
    public void AllOrNothing_ForcedStream_FreesEveryBlockWhenThePinWasNeverSized()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), long.MaxValue);
        FakeDenoiser denoiser = new FakeDenoiser(4);
        ResidentPrefixPin pin = new ResidentPrefixPin { Resident = true, PinnedBlocks = -1 };

        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0, pin: pin,
            mode: LowVramMode.ForceOn, policy: BlockStreamingPolicy.AllOrNothing));

        string freed = string.Join(",", backend.Calls.Where(c => c.StartsWith("free:", StringComparison.Ordinal)));
        Assert.Contains("b0", freed, StringComparison.Ordinal);
        Assert.Contains("b3", freed, StringComparison.Ordinal);
        Assert.Equal(0, scope.ResidentPrefixBlocks);
    }

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

    /// <summary>A cold pin has nothing to displace, so the force must not manufacture a free call.</summary>
    [Fact]
    public void AllOrNothing_ForcedStream_DoesNotFreeWhenNothingIsResident()
    {
        RecordingStreamingBackend backend = new RecordingStreamingBackend(new StubCache(), long.MaxValue);
        FakeDenoiser denoiser = new FakeDenoiser(4);
        ResidentPrefixPin pin = new ResidentPrefixPin { Resident = false };

        using BlockStreamingScope scope = BlockStreamingScope.Open(Options(backend, denoiser, 0, pin: pin,
            mode: LowVramMode.ForceOn, policy: BlockStreamingPolicy.AllOrNothing));

        Assert.DoesNotContain(backend.Calls, c => c.StartsWith("free:", StringComparison.Ordinal));
        Assert.Equal(0, scope.ResidentPrefixBlocks);
    }
}
