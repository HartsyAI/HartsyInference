using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-checkpoint plan-resolution coverage for native PDD acceleration: the official
/// alibaba-pai/MiniMax-H3-Acc-LoRAs FL2VA-Acc-8Step adapter applied on top of the pruned fp8 base checkpoint used
/// by the dense canaries. Proves the profile resolver detects the adapter's actual on-disk format (BF16 banks, no
/// pdd_task metadata — both of which the resolver previously rejected as invalid) and correctly still blocks
/// execution because that base is pruned and no full-width base or Hartsy pruned-rebase conversion is available in
/// this environment. A full accelerated generation needs either the ~34 GB full-width base (which does not fit
/// this box's free disk, and would not fit resident on a single 24 GB card even if it did) or `hartsy convert
/// h3-pdd`, which itself requires that same full-width base as input — so it is out of reach here regardless of
/// this fix. See <c>MiniMaxH3PddAdapterRealFileTests</c> (ModelAssets.Tests) for the companion CPU-only proof that
/// the loader itself (BF16→F32 promotion, hash-bound task binding) works end to end against this exact file.</summary>
[Collection("CudaSerial")]
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class MiniMaxH3PddRealWeightTests
{
    private const string RunEnvVar = "HARTSY_RUN_H3_PDD_REAL";

    private const string AdapterSha256 =
        "0b29be7042d883970eb0c20774a9ba03d95669ed80a721bb4d21be8ea0d0a196";

    private const string Prompt =
        "A joyful golden retriever puppy splashing through a shallow forest stream, cinematic natural light, "
        + "realistic water droplets, steady tracking camera, coherent motion, ambient rushing water and birds";

    private readonly ITestOutputHelper _output;

    public MiniMaxH3PddRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task PddFl2VaAdapter_RealCheckpoint_DetectedButBlockedByPrunedBase()
    {
        if (Environment.GetEnvironmentVariable(RunEnvVar) != "1")
        {
            _output.WriteLine($"SKIPPED: set {RunEnvVar}=1 and {RealWeightGate.RequireEnvVar}=1 "
                + "to run the exact-artifact MiniMax-H3 native PDD plan-resolution canary.");
            return;
        }

        Assert.True(Environment.GetEnvironmentVariable(RealWeightGate.RequireEnvVar) == "1",
            $"{RunEnvVar}=1 must be paired with {RealWeightGate.RequireEnvVar}=1 so missing exact assets fail.");
        Assert.True(CudaContext.IsAvailable(), $"{RunEnvVar}=1 requires a CUDA device.");
        if (!RealWeightGate.Require(_output.WriteLine,
                TestPaths.MiniMaxH3.DitFp8,
                TestPaths.MiniMaxH3.TextEncoder,
                TestPaths.MiniMaxH3.VideoVae,
                TestPaths.MiniMaxH3.AudioVae,
                TestPaths.MiniMaxH3.PddFl2VaAdapter))
        {
            return;
        }

        string adapterHash = await Sha256Async(TestPaths.MiniMaxH3.PddFl2VaAdapter);
        Assert.Equal(AdapterSha256, adapterHash);

        int gpu = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_TEST_GPU"), out int requestedGpu)
            ? requestedGpu : 0;
        Assert.InRange(gpu, 0, CudaContext.GetDeviceCount() - 1);
        ModelSpec spec = ModelResolver.Resolve(
            "minimax-h3", TestPaths.MiniMaxH3.DitFp8, Modality.Video);

        ComponentOverrides components = new()
        {
            Qwen = TestPaths.MiniMaxH3.TextEncoder,
            VideoVae = TestPaths.MiniMaxH3.VideoVae,
            AudioVae = TestPaths.MiniMaxH3.AudioVae,
        };
        VideoRequest request = new()
        {
            Prompt = Prompt,
            Width = 512,
            Height = 288,
            Frames = 39,
            Fps = 24,
            Steps = 8,
            CfgScale = 1f,
            FlowShift = 12f,
            AudioFlowShift = 3f,
            Sampler = "euler",
            Seed = 424242,
            Components = components,
            Loras = new LoraStack
            {
                Entries = [new LoraEntry { Model = TestPaths.MiniMaxH3.PddFl2VaAdapter, Weight = 1.0 }],
            },
        };

        using InferenceEngine engine = new("cuda", gpu);
        VideoPlan plan = await engine.VideoPlanning.PlanAsync(spec, request);
        foreach (VideoPlanIssue issue in plan.Issues)
        {
            _output.WriteLine($"plan {issue.Severity}: {issue.Code}: {issue.Message}");
        }

        // Before the fix this real file failed two structural checks that do not describe the actual published
        // artifact: video.pdd.bank_dtype_invalid (the real banks are BF16, not F32) and video.pdd.task_missing
        // (the real metadata carries no pdd_task/pdd_partition/hartsy.pdd.task key at all). Neither may appear now.
        Assert.DoesNotContain(plan.Issues, issue => issue.Code == "video.pdd.bank_dtype_invalid");
        Assert.DoesNotContain(plan.Issues, issue => issue.Code == "video.pdd.task_missing");
        Assert.DoesNotContain(plan.Issues, issue => issue.Code == "video.pdd.task_invalid");

        // What legitimately remains: this box only has the pruned fp8 base, and official PDD adapters require a
        // full-width base (or a Hartsy pruned-rebase conversion this environment cannot produce — see class doc).
        VideoPlanIssue rebaseIssue = Assert.Single(plan.Issues, issue => issue.Code == "video.pdd.rebase_required");
        Assert.Equal(VideoPlanIssueSeverity.Error, rebaseIssue.Severity);
        Assert.False(plan.IsValid);

        _output.WriteLine("Confirmed: the real official adapter's BF16 banks and metadata-free task now parse "
            + "correctly; the only remaining block is the pruned-base binding, which is correct given this "
            + "checkpoint. A full accelerated generation is out of reach in this environment (see class doc).");
    }

    private static async Task<string> Sha256Async(string path)
    {
        await using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            4 * 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
        using System.Security.Cryptography.SHA256 sha = System.Security.Cryptography.SHA256.Create();
        return Convert.ToHexString(await sha.ComputeHashAsync(stream)).ToLowerInvariant();
    }
}
