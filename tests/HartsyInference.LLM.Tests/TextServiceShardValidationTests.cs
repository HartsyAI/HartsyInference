using HartsyInference.Core.Exceptions;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Guards the Phase 1 fix for a real bug: a request-level shard composite mixing a CPU stage into an
/// LLM layer-split (e.g. <c>"cuda:0+cpu"</c>) used to reach <c>TextService.LoadSharded</c>, which always loads
/// the GGUF with <c>dequantizeToF32: false</c> — correct for CUDA stages, but a CPU backend can't compute on
/// quantized tensors (CPU requires F32). <c>TextService.ValidateShardDevices</c> now rejects any non-CUDA
/// device in a shard list before any file/backend work happens, so this needs no real checkpoint or GPU: the
/// exception fires on the composite string alone.</summary>
public sealed class TextServiceShardValidationTests
{
    private readonly ITestOutputHelper _output;
    public TextServiceShardValidationTests(ITestOutputHelper output) => _output = output;

    private static ModelSpec FakeSpec() => new()
    {
        Requested = "fake",
        Modality = Modality.Text,
        LocalPath = "/nonexistent/fake-model.gguf",
    };

    private static TextRequest RequestWithDevice(string device) => new()
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = "hi" }],
        MaxTokens = 1,
        Device = device,
    };

    [Fact]
    public async Task ShardedRequest_WithCpuStage_ThrowsBeforeAnyFileAccess()
    {
        using InferenceEngine engine = new("cuda", 0);
        HartsyInferenceException ex = await Assert.ThrowsAsync<HartsyInferenceException>(
            () => engine.Text.GenerateAsync(FakeSpec(), RequestWithDevice("cuda:0+cpu")));
        Assert.Contains("cpu", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("CUDA", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ShardedRequest_AllCudaStages_PassesValidation_FailsLaterOnMissingFile()
    {
        using InferenceEngine engine = new("cuda", 0);
        // A valid all-CUDA composite must clear ValidateShardDevices and fail downstream on the missing
        // checkpoint (FileNotFoundException from GgufLoader) instead — proves the new check isn't rejecting
        // legitimate shard lists too, by confirming it's NOT the shard-validation exception that fires here.
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => engine.Text.GenerateAsync(FakeSpec(), RequestWithDevice("cuda:0+cuda:1")));
    }

    /// <summary>Real-checkpoint companion to the two facts above: an SSM architecture (Mamba, no per-layer-
    /// crossing story for its recurrent state) requesting a composite shard key used to silently resolve to
    /// whatever <c>CreateBackendFor</c>'s naive first-colon parse happened to land on (ordinal 0, NOT
    /// necessarily the first requested device) with zero log signal. It now resolves explicitly to
    /// <c>shardDevices[0]</c> and logs a warning — this just proves the request completes successfully (no
    /// throw, no crash) on the real Mamba checkpoint under a composite key, which is what "falls back cleanly"
    /// means in practice; the exact device-resolution logic is covered structurally by the code path itself.</summary>
    [Fact]
    public async Task ShardedRequest_SsmArchitecture_FallsBackCleanly_NoThrow()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string checkpoint = TestPaths.Llm.Mamba28BQ4K;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        using InferenceEngine engine = new("cuda", 0);
        ModelSpec spec = new() { Requested = "mamba-2.8b", Modality = Modality.Text, LocalPath = checkpoint };
        // A 2-way composite on a box with only 1 GPU still exercises the SSM fallback branch (ResolveShardDevices
        // doesn't require the second device to physically exist to build the shard list) without needing 2 GPUs.
        TextResult result = await engine.Text.GenerateAsync(spec, RequestWithDevice("cuda:0+cuda:1") with { MaxTokens = 4 });
        _output.WriteLine($"Generated {result.CompletionTokens} tokens: \"{result.Text}\"");
        Assert.True(result.CompletionTokens > 0);
    }
}
