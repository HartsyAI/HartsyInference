using HartsyInference.Core.Backends;
using HartsyInference.Gpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Loads a model, generates, tears it down and loads a different one — repeatedly, in ONE process.
///
/// <para>This is the scenario that produced the crash <c>State.Unregistered</c> exists for: a tensor finalized
/// after its backend was torn down queues GPU cleanup against a state whose <c>ConditionalWeakTable</c> may itself
/// have been finalized, and touching it throws. Nothing tested it. A server swapping models is the ordinary case,
/// not an exotic one, and neither the byte-identity harness nor the rest of this suite reaches it — the harness
/// runs one model per CLI invocation, and the suite's backends come and go without a second model taking their
/// place.</para>
///
/// <para>Prerequisite for migrating the CUDA residency cache onto the shared base: the bugs that migration is most
/// likely to introduce are teardown bugs, and a test that never tears down twice cannot find them.</para>
///
/// <para>Two tests, because the question has two halves. The CUDA one below asks whether a torn-down backend's
/// state is retired and its registry entry gone, which is CUDA's own mechanism. The cross-backend one asks the
/// portable question — whether a backend an engine keeps for its lifetime gives its device memory back when a
/// model is swapped through it — and that one is where a backend whose <c>FreeAllDeviceMemory</c> does nothing
/// shows up, rather than in any single generation.</para></summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed class ModelSwapSoakTests
{
    private const string Prompt = "Write one sentence about a lighthouse.";
    private const int MaxTokens = 24;
    private const int Rounds = 2;

    /// <summary>Headroom for driver-side accounting drift between two probes. Far below the smallest model
    /// here (~1.3 GB), so a leaked model cannot hide under it.</summary>
    private const long LeakToleranceBytes = 256L << 20;

    private readonly ITestOutputHelper _output;

    public ModelSwapSoakTests(ITestOutputHelper output) => _output = output;

    /// <summary>Set to 1 to turn "the models are not on this box" from a skip into a failure.</summary>
    /// <remarks>The suite's convention is to print SKIPPED and return, which xunit records as a PASS — fine for a
    /// capability probe, wrong for the gate on a migration. The gate run sets this; a casual run does not.</remarks>
    private const string RequireVar = "HARTSY_REQUIRE_MODEL_SWAP_SOAK";

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    /// <summary>Free device memory after returning pool reservations to the driver.</summary>
    /// <remarks>The trim is not optional. Every activation free goes through <c>cuMemFreeAsync</c>, which hands
    /// the block back to the stream-ordered mempool, and a pooled block still counts as USED in
    /// <c>cuMemGetInfo</c> until trimmed — measured at 6.9 GB "unreturned" across four swaps here before the trim
    /// was added, which is pool reservation working as designed, not a leak. Measuring without it would have made
    /// this assertion report a leak on every healthy run.</remarks>
    private static long FreeVram()
    {
        using CudaBackend probe = new(0, PtxDir());
        probe.TrimMemoryPool();
        return probe.GetVramInfo().FreeBytes;
    }

    private static void ForceFullGc()
    {
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
    }

    /// <summary>Reports whether the run may proceed, failing instead of skipping when the gate demands it.</summary>
    private bool CanRun(string[] models)
    {
        bool required = Environment.GetEnvironmentVariable(RequireVar) == "1";
        string? reason = !CudaContext.IsAvailable() ? "CUDA unavailable"
            : models.FirstOrDefault(m => !File.Exists(m)) is string missing ? $"model not found: {missing}"
            : null;
        if (reason is null)
        {
            return true;
        }
        Assert.False(required, $"{RequireVar}=1 but the soak cannot run: {reason}");
        _output.WriteLine($"SKIPPED: {reason}");
        return false;
    }

    /// <summary>One load → generate → teardown cycle. Returns the state key the backend used, so the caller can
    /// assert it is gone afterwards.</summary>
    /// <remarks>Deliberately NOT inlined into the test: the model, pipeline and every tensor they created have to
    /// become unreachable before the collection below, and a local in the enclosing scope can stay rooted for the
    /// life of the method.</remarks>
    private nint RunOneModel(string path, bool abandon)
    {
        CudaBackend backend = new(0, PtxDir());
        nint key = backend.TransferState.Key;
        // Deliberately NOT `using`. An orderly Dispose unbinds every tensor and discards the pending-cleanup
        // bucket while the backend is still alive, so nothing survives to fire against a retired state — measured:
        // with both disposed, this test passes even with the retirement guard removed, which means it would be
        // testing nothing. Abandoning both is what leaves a queued GPU cleanup pointing at a state that the GC may
        // already have finalized, and the next model then loads on top of that.
        GgufLanguageModel model = GgufLanguageModel.Load(path, dequantizeToF32: false);
        TextGenerationPipeline pipeline = new(model.Transformer, model.Tokenizer, backend, model.Template);
        GenerationResult result = pipeline.Generate(new GenerationRequest
        {
            Prompt = Prompt,
            MaxTokens = MaxTokens,
            Sampling = SamplingOptions.Default with { Greedy = true },
        });
        Assert.NotEmpty(result.TokenIds);
        _output.WriteLine($"{Path.GetFileName(path)} (state {key}): {result.TokenIds.Count} tokens, "
            + $"{backend.TransferState.WeightCount} weights resident, "
            + $"{backend.TransferState.ActivationCount} activations live, "
            + (abandon ? "abandoned" : "disposed"));
        if (!abandon)
        {
            model.Dispose();
            backend.Dispose();
        }
        return key;
    }

    [Fact]
    public void SwappingModels_RetiresEveryStateAndKeepsGenerating()
    {
        string[] models = [TestPaths.Llm.Llama32_1BQ8, TestPaths.Llm.Qwen3_4BQ4KM];
        if (!CanRun(models))
        {
            return;
        }

        int registryBaseline = GpuTransferHelper.RegisteredStateCount;
        long freeBefore = FreeVram();
        List<nint> keys = [];
        for (int round = 0; round < Rounds; round++)
        {
            foreach (string path in models)
            {
                // Alternate: an orderly dispose is the common path; abandoning is the one that reaches the
                // finalizer-after-teardown crash, because only then is a state still live — or already collected —
                // when the next model starts loading. Both happen in a long-running server.
                bool abandon = round % 2 == 1;
                long reaped = CudaBackend.AbandonedCleanupCompletedCount;
                nint key = RunOneModel(path, abandon);
                keys.Add(key);

                // The crash this guards is a tensor FINALIZED after its backend retired, so the collection has to
                // happen here, between the two models, rather than at the end of the test. Generation abandons
                // activations rather than disposing every one, which is what makes the queue non-empty.
                ForceFullGc();
                if (abandon)
                {
                    Assert.True(CudaBackend.WaitForAbandonedCleanup(reaped + 1, TimeSpan.FromSeconds(60)),
                        $"the reaper never collected abandoned state {key}");
                    ForceFullGc();
                }

                Assert.True(SpinWait.SpinUntil(() => !GpuTransferHelper.IsStateRegistered(key), TimeSpan.FromSeconds(30)),
                    $"state {key} still registered after its backend was released and finalizers ran");
            }
        }

        // Every state is gone, so the registry is back where it started: a swap that leaked one would show here as
        // a count that climbed with Rounds.
        Assert.Equal(registryBaseline, GpuTransferHelper.RegisteredStateCount);

        // And the device memory came back. This is the assertion with teeth for the cache migration: all three
        // bugs the Vulkan half of this work produced were teardown leaks, invisible to a single generation and to
        // any test that builds one backend and stops. The smallest model here holds ~1.3 GB, so a tolerance of a
        // quarter GB cannot hide a leaked model while still absorbing driver-side accounting drift.
        long freeAfter = FreeVram();
        long leaked = freeBefore - freeAfter;
        _output.WriteLine($"free VRAM {freeBefore >> 20} MB -> {freeAfter >> 20} MB after {keys.Count} swaps "
            + $"({leaked >> 20} MB unreturned)");
        Assert.True(leaked < LeakToleranceBytes,
            $"{leaked >> 20} MB of device memory was not returned across {keys.Count} model swaps");
        Assert.Equal(keys.Count, keys.Distinct().Count());
        foreach (nint key in keys)
        {
            Assert.DoesNotContain(key, GpuTransferHelper.RegisteredStateKeysForTests);
        }
    }

    /// <summary>The shape a server actually runs: ONE backend, many models through it. Load, generate, release,
    /// load a different family, generate, release — and the device memory the backend is holding comes back each
    /// time, rather than at whatever later moment the GC reaches the tensors.</summary>
    /// <remarks>The release call is the subject, so it is deliberately the only thing between rounds: with
    /// <c>FreeAllDeviceMemory</c> a no-op — which it was on every backend but CUDA — this fails on the first
    /// swap. Model choice is the smallest pair that still crosses families and quantizations, because a backend
    /// that cannot hold a packed weight dequantizes on load, and the 4B the CUDA test uses costs sixteen
    /// gigabytes and several minutes there.
    ///
    /// <para>Free VRAM is read from the SAME backend throughout. A fresh probe instance would report a Vulkan
    /// device as entirely free whatever the previous one was holding, since that figure is the total minus what
    /// the asking allocator has taken.</para></remarks>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void SwappingModelsThroughOneBackend_ReturnsItsDeviceMemory(string kind)
    {
        string[] models = [TestPaths.Llm.Llama32_1BQ8, TestPaths.Llm.Qwen25_05BQ4KM];
        if (models.FirstOrDefault(path => !File.Exists(path)) is string missing)
        {
            Assert.False(Environment.GetEnvironmentVariable(RequireVar) == "1",
                $"{RequireVar}=1 but the soak cannot run: model not found: {missing}");
            _output.WriteLine($"SKIPPED: model not found: {missing}");
            return;
        }
        if (!BackendGate.TryOpen(kind, _output.WriteLine, out IBackend? opened))
        {
            return;
        }
        using IBackend backend = opened!;

        // One full cycle before the baseline. First touch brings up pipelines, descriptor pools and the first
        // slabs, none of which come back, and measuring from before that reads start-up cost as a leak.
        SwapOneModelThrough(backend, models[0]);
        ForceFullGc();
        long freeBefore = backend.GetVramInfo().FreeBytes;

        for (int round = 0; round < Rounds; round++)
        {
            foreach (string path in models)
            {
                SwapOneModelThrough(backend, path);
                ForceFullGc();
            }
        }

        long freeAfter = backend.GetVramInfo().FreeBytes;
        long unreturned = freeBefore - freeAfter;
        _output.WriteLine($"[{kind}] free VRAM {freeBefore >> 20} MB -> {freeAfter >> 20} MB after "
            + $"{Rounds * models.Length} swaps ({unreturned >> 20} MB unreturned)");
        Assert.True(unreturned < LeakToleranceBytes,
            $"[{kind}] {unreturned >> 20} MB of device memory was not returned across "
            + $"{Rounds * models.Length} model swaps through one backend");
    }

    /// <summary>Loads a model onto this backend, generates, and releases every device allocation it made.</summary>
    /// <remarks>Not inlined, for the same reason as <see cref="RunOneModel"/>: the model and its tensors have to
    /// be unreachable before the caller collects, and a local in the enclosing scope stays rooted.</remarks>
    private void SwapOneModelThrough(IBackend backend, string path)
    {
        // A backend that cannot read a packed weight needs it widened on load — the same question the engine's
        // own loaders ask, asked the same way, so this test degrades exactly as production does.
        GgufLanguageModel model = GgufLanguageModel.Load(path, dequantizeToF32: !backend.Capabilities.SupportsQuantized);
        try
        {
            TextGenerationPipeline pipeline = new(model.Transformer, model.Tokenizer, backend, model.Template);
            GenerationResult result = pipeline.Generate(new GenerationRequest
            {
                Prompt = Prompt,
                MaxTokens = MaxTokens,
                Sampling = SamplingOptions.Default with { Greedy = true },
            });
            Assert.NotEmpty(result.TokenIds);
            _output.WriteLine($"  {Path.GetFileName(path)}: {result.TokenIds.Count} tokens");
        }
        finally
        {
            model.Dispose();
            backend.FreeAllDeviceMemory();
        }
    }
}
